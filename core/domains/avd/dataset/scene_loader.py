import os
import json
import argparse
import sys
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from core.domains.avd.dataset.embedding_utils import convert_uint8_to_float

sys.path.append('.')

from core.utils.geometry_utils import rotation_from_camera_normal
from core.domains.avd.dataset.parser_utils import parse_mat_structs, change_image_type
from core.domains.avd.dataset.const import AVDMove, ImageType
from core.domains.avd.dataset.data_classes import Scene, ImageNode


def rotation_from_camera_normal(normal: np.ndarray):
    vertical = np.array([0, 1, 0])
    waxis = normal.reshape(-1)              # w axis is pointing outwards from the camera
    uaxis = np.cross(vertical, waxis)       # u axis is pointing in the negative direction of the image width
    vaxis = np.cross(waxis, uaxis)          # v axis is pointing upwards aligned with the height
    R = np.stack([uaxis, vaxis, waxis])
    return R


class RGBLoader:
    def __init__(self, img_dir, image_names: List[str]):
        self.dir = img_dir
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        return np.array(plt.imread(os.path.join(self.dir, self.image_names[index])))


class DepthLoader:
    def __init__(self, img_dir, image_names: List[str]):
        self.dir = img_dir
        self.image_names = image_names

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        path = os.path.join(self.dir, change_image_type(self.image_names[index], ImageType.DEPTH))
        return np.array(cv2.imread(path, cv2.IMREAD_ANYDEPTH))


class AVDLoadImages(Dataset):
    def __init__(self, scene_path, image_names, img_size: tuple, include_depth=True):
        self.img_size = img_size
        self.image_names = image_names
        self.rgb_dir = os.path.join(scene_path, 'jpg_rgb')
        self.depth_dir = os.path.join(scene_path, 'high_res_depth')
        self.include_depth = include_depth

    def __getitem__(self, index):
        H, W = self.img_size
        image_name = self.image_names[index]
        rgb_img_path = os.path.join(self.rgb_dir, image_name)
        rgb_img = np.array(plt.imread(rgb_img_path))
        if rgb_img.shape != (H, W):
            rgb_img = cv2.resize(rgb_img, (W, H), interpolation=cv2.INTER_NEAREST)
        if self.include_depth:
            depth_img_path = os.path.join(self.depth_dir, change_image_type(image_name, ImageType.DEPTH))
            depth_img = np.array(cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH))
            if depth_img.shape != (H, W):
                depth_img = cv2.resize(depth_img, (W, H), interpolation=cv2.INTER_NEAREST)
            depth_img = depth_img.astype(np.int16)
        else:
            depth_img = None
        return rgb_img, depth_img

    def __len__(self):
        return len(self.image_names)


def avd_load_images(scene_path, image_names, rescale_size: tuple =None, avd_workers=None):
    print("Load as size: ", rescale_size)
    H, W = rescale_size
    rgb_save_path = os.path.join(scene_path, f"images_{H}x{W}.npy")
    depth_save_path = os.path.join(scene_path, f'depths_{H}x{W}.npy')
    if os.path.exists(rgb_save_path) and os.path.exists(depth_save_path):
        rgb_imgs = np.load(rgb_save_path)
        depth_imgs = np.load(depth_save_path)
    else:
        rgb_imgs = np.zeros((len(image_names), H, W, 3), dtype=np.uint8)
        depth_imgs = np.zeros((len(image_names), H, W), dtype=np.uint16)
        loader = DataLoader(AVDLoadImages(scene_path, image_names, (H, W)), batch_size=1, num_workers=avd_workers)
        for i, (rgb_img, depth_img) in enumerate(loader):
            sys.stdout.write(f"\rReading {i + 1} / {len(image_names)} image...")
            sys.stdout.flush()
            rgb_imgs[i] = rgb_img.data.numpy()
            depth_imgs[i] = depth_img.data.numpy().astype(np.uint16)
        if rescale_size != (1080, 1920):
            np.save(rgb_save_path, rgb_imgs)
            np.save(depth_save_path, depth_imgs)
    return rgb_imgs, depth_imgs


class EmbeddingLoader(Dataset):
    def __init__(self, scene_path, image_names, rescale_size, in_ram=False):
        self.image_names = image_names
        self.embeddings = None
        self.in_ram = in_ram
        if in_ram:
            H, W = rescale_size
            self.emb_path = os.path.join(scene_path, f"resnet_{H}x{W}.pt")
            if os.path.exists(self.emb_path):
                print(f"loading embeddings from {self.emb_path}...")
                self.embeddings = torch.load(self.emb_path)

    def __getitem__(self, index):
        assert self.in_ram, "in_ram must be True to load embeddings"
        return self.embeddings[index].float()

    def __len__(self):
        return len(self.image_names)


__scenes = {}


def avd_scene_loader(data, scene_name, resize: tuple, in_ram=False, avd_workers=None, verbose=True, **kwargs):
    """
    constructs a scene data structure (a graph with image nodes and transition edges)
    The returned "Scene" object contains all the information you need about the environment.
    Scene.image_nodes is a dictionary mapping every image_name (str) in a scene to a corresponding
    image_node (ImageNode), which is a data structure that retrieves the RGB and depth images.
    Scene.camera holds the camera parameters if they are available.
    """
    if resize is None: resize = (1080, 1920)
    scene_key = (scene_name, tuple(resize))
    scene = __scenes.get(scene_key)
    if scene: return scene
    if verbose: print(f"Processing scene: {scene_name}")
    scene_path = os.path.join(data, scene_name)
    annotations_path = os.path.join(scene_path, 'annotations.json')
    structs_path = os.path.join(scene_path, 'image_structs')
    camera_params_path = os.path.join(scene_path, 'cameras.txt')
    rgb_dir = os.path.join(scene_path, 'jpg_rgb')
    depth_dir = os.path.join(scene_path, 'high_res_depth')
    # get pandas data frame for camera parameters
    image_structs, scale = parse_mat_structs(structs_path)

    image_names = sorted(list(name for name in os.listdir(rgb_dir) if name.split('.')[-1] in ['jpg', 'png']))
    name_to_ind = {name: i for i, name in enumerate(image_names)}
    if in_ram or resize:
        rgb_imgs, depth_imgs = avd_load_images(scene_path, image_names, resize, avd_workers)
    else:
        rgb_imgs, depth_imgs = RGBLoader(rgb_dir, image_names), DepthLoader(depth_dir, image_names)
    embeddings = EmbeddingLoader(scene_path, image_names, resize, in_ram)

    instance_id_map_path = os.path.join(data, "instance_id_map.txt")
    # create scene instance
    scene = Scene(scene_name, scale, camera_params_path, rgb_imgs, depth_imgs, embeddings,
                  resize, instance_id_map_path, scene_path)
    # opening json file with image annotation
    with open(annotations_path) as f:
        annotations = json.load(f)

    # creating image nodes for all images in the scene
    # get camera parameters
    for i, record in image_structs.iterrows():
        image_name = record['image_name']
        pos, dir, R, t, quat = [np.array(record[k]) for k in ['world_pos', 'direction', 'R', 't', 'quat']]
        if not R.size:
            # R and t are not recorded so we have to calculate these ourselves.
            # assuming that the camera's x axis is horizontal
            R = rotation_from_camera_normal(dir)
            t = pos
        image_node = ImageNode(scene, image_name, name_to_ind[image_name], pos, dir, R, t, quat)
        scene.image_nodes[image_name] = image_node

    # iterating through all image nodes
    for image_name, image_node in scene.image_nodes.items():
        # getting annotation for current image
        image_info = annotations[image_name]

        # adding valid transitions to the image node
        for move in AVDMove:
            next_image_name = image_info[move.name]
            if next_image_name:
                image_node.moves[move] = scene.image_nodes[next_image_name]

        # adding objects detected in image
        for object_info in image_info["bounding_boxes"]:
            bbox, obj_class_id = object_info[:4], int(object_info[4])
            rescaled_bbox = scene.camera.rescale_bbox(bbox)
            obj = scene.object_classes[obj_class_id].append_new(image_node, rescaled_bbox)
            image_node.objects.append(obj)

    __scenes[scene_key] = scene

    return scene


def get_scene_from_commandline(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--scene_name', "-s", required=True, help="scene name")
    parser.add_argument('--data', default="data/avd/src/", help="path to data directory")
    parser.add_argument("--resize", "-sz", help="resize images", type=int, nargs=2, default=None)
    parser.add_argument("--in_ram", "-r", action="store_true", help="store images in ram")
    parser.add_argument("--avd_workers", type=int, default=8, help="number of workers")
    args = parser.parse_args()

    return avd_scene_loader(**vars(args)), args


if __name__ == "__main__":
    scene, args = get_scene_from_commandline()
    print(scene.image_nodes)

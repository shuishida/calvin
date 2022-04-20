import os
import numpy as np
import torch
from typing import Dict, List, Union
from core.domains.avd.dataset.const import SceneType, AVDMove
from core.domains.avd.dataset.parser_utils import parse_scene_name, parse_camera_params
from core.utils.geometry_utils import get_world_coord_from_mesh, sample_free_space, get_undistorted_radius_mapping, \
    undistort_mesh, rotation_from_camera_normal


class Camera:
    scene: 'Scene'
    params_available: bool
    width: int
    height: int
    rescale_width: int
    rescale_height: int
    f_x: float
    f_y: float
    c_x: float
    c_y: float
    distort_coeffs: list
    xs: np.ndarray
    ys: np.ndarray

    def __init__(self, scene: 'Scene', camera_params_path, rescale_size: tuple):
        self.scene = scene
        self.width, self.height, self.f_x, self.f_y, self.c_x, self.c_y, self.distort_coeffs \
            = parse_camera_params(camera_params_path)
        self.rescale_height, self.rescale_width = (self.height, self.width) if rescale_size is None else rescale_size
        self.params_available = True
        self.xs, self.ys = self.get_corrected_camera_coords()

    def get_corrected_camera_coords(self):
        # get (u, v) coordinates in pixel space
        us, vs = np.meshgrid(
            np.linspace(0, self.width, self.rescale_width),
            np.linspace(0, self.height, self.rescale_height))
        # get (x, y, z) coordinates in camera space
        xs = (us - self.c_x) / self.f_x * (1)  # rotation matrix convention goes in the opposite direction
        ys = (vs - self.c_y) / self.f_y * (1)  # rotation matrix convention goes in the opposite direction

        return undistort_mesh(xs, ys, self.distort_coeffs)

    def rescale_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        x_scale = self.rescale_width / self.width
        y_scale = self.rescale_height / self.height
        return tuple(map(int, (x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale)))


class ObjectClass(List['Object']):
    def __init__(self, class_id: int, name: str):
        super(ObjectClass, self).__init__()
        self.id = class_id
        self.name = name

    def __str__(self):
        return f"ObjectClass<{self.name}>"

    def __hash__(self):
        return self.id

    def append_new(self, image_node: 'ImageNode', bbox: tuple) -> 'Object':
        obj = Object(self, image_node, bbox)
        self.append(obj)
        return obj


class Object:
    def __init__(self, object_class: ObjectClass, image_node: 'ImageNode', bbox: tuple):
        self.object_class = object_class
        self.image_node = image_node
        self.bbox = bbox

    def __hash__(self):
        return self.object_class.id, self.image_node.image_name, self.bbox

    def __str__(self):
        return f"Object<{self.object_class.name}, {self.image_node}, {self.bbox}>"

    def rgb(self):
        x1, y1, x2, y2 = self.bbox
        return self.image_node.rgb()[y1:y2, x1:x2]

    def embedding(self):
        x1, y1, x2, y2 = self.bbox
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        embeddings = self.image_node.embedding()
        pad = 5
        embeddings = np.pad(embeddings, [(0,), (pad,), (pad,)], mode='constant')
        return embeddings[:, y:y+2*pad+1, x:x+2*pad+1]

    def depth(self):
        x1, y1, x2, y2 = self.bbox
        return self.image_node.depth()[y1:y2, x1:x2]

    def mask(self):
        mask = np.zeros(self.image_node.depth().shape, dtype=bool)
        x1, y1, x2, y2 = self.bbox
        mask[y1:y2, x1:x2] = 1
        return mask

    @property
    def size(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class ObjectClasses(Dict[int, ObjectClass]):
    def __init__(self, instance_id_map_path: str):
        super(ObjectClasses, self).__init__()
        self.by_name = {}
        with open(instance_id_map_path, 'r') as f:
            for line in f:
                name, class_id = str.split(line)
                class_id = int(class_id)
                instance = ObjectClass(class_id, name)
                self[class_id] = instance
                self.by_name[name] = instance

    def get_by_name(self, name: str) -> ObjectClass:
        return self.by_name[name]


class Scene:
    name: str
    camera: Camera or None
    scale: float or int
    scene_type: SceneType
    scene_number: int
    scan_number: int
    image_nodes: Dict[str, 'ImageNode']
    rgb_images: np.ndarray
    depth_images: np.ndarray
    rescale_size: tuple
    object_classes: ObjectClasses
    scene_path: str

    def __init__(self, scene_name: str, scale: float or int, camera_params_path: str,
                 rgb_images: Union[np.ndarray, List], depth_images: Union[np.ndarray, List], embeddings,
                 rescale_size: tuple, instance_id_map_path: str, scene_path: str):
        self.name = scene_name
        self.scene_type, self.scene_number, self.scan_number = parse_scene_name(scene_name)
        self.image_nodes = {}
        self.scale = scale
        self.camera = Camera(self, camera_params_path, rescale_size) if os.path.exists(camera_params_path) else None
        self.rgb_images = rgb_images
        self.depth_images = depth_images
        self.embeddings = embeddings
        self.object_classes = ObjectClasses(instance_id_map_path)
        self.scene_path = scene_path

    def __repr__(self):
        return self.name

    def coords(self, image_names):
        """
        :return: (3, height, width) where the channels store the world coordinate values (X, Y, Z)
        for each pixel in the image
        """
        camera = self.camera
        coords = []
        for image_name in image_names:
            node = self.image_nodes[image_name]
            coords.append(get_world_coord_from_mesh(camera.xs, camera.ys, node.depth(), node.R, node.position))
        return np.stack(coords)

    def names_to_indices(self, image_names):
        return [self.image_nodes[image_name].image_index for image_name in image_names]


class ImageNode:
    scene: Scene
    image_name: str
    image_index: int
    position: np.ndarray
    camera_direction: np.ndarray
    camera_quaternion: np.ndarray
    moves: Dict[AVDMove, 'ImageNode']
    objects: List['Object']
    world_coords: np.ndarray or None
    is_valid_coords: np.ndarray or None

    def __init__(self, scene: Scene, image_name: str, image_index: int, position, direction, R, t, quaternion=None):
        self.scene = scene
        self.image_name = image_name
        self.image_index = image_index
        self.objects = []
        self.moves = {}
        self.position = position
        self.camera_direction = direction
        self.R = R
        self.t = t
        self.camera_quaternion = quaternion

    def __repr__(self):
        return f"{self.scene}:{self.image_name}"

    def rgb(self):
        return self.scene.rgb_images[self.image_index]

    def depth(self):
        return self.scene.depth_images[self.image_index] / self.scene.scale

    def embedding(self):
        return self.scene.embeddings[self.image_index]

    @property
    def world_coords(self):
        """
        :return: (3, height, width) where the channels store the world coordinate values (X, Y, Z)
        for each pixel in the image
        """
        camera = self.scene.camera
        return get_world_coord_from_mesh(camera.xs, camera.ys, self.depth(), self.R, self.position)

    @property
    def is_valid_coords(self):
        return self.depth() != 0

    def normalised_rgbd(self):
        return np.concatenate([self.rgb() / 255, np.expand_dims(self.depth(), -1)], axis=-1)

    def point_cloud(self):
        is_valid = self.is_valid_coords
        return np.concatenate([self.world_coords[is_valid], self.rgb()[is_valid]], axis=-1).reshape((-1, 6))

    def sample_free_space(self, n_samples_per_pixel=1):
        return sample_free_space(self.world_coords, self.position, n_samples_per_pixel=n_samples_per_pixel)

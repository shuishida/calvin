from mat4py import loadmat
from core.domains.avd.dataset.const import SceneType, ImageType
import os
import pandas as pd


def image_name_parser(name):
    scene_type_str, scene_number, scan_number, image_index, camera_index, image_type_str \
        = name[0:1], int(name[1:4]), int(name[4:5]), int(name[5:11]), int(name[11:13]), name[13:15]
    scene_type = image_type = None
    if scene_type_str == "0": scene_type = SceneType.HOME
    elif scene_type_str == "1": scene_type: SceneType.OFFICE
    else: raise Exception(f"Unknown scene type {scene_type_str} in image name")
    if image_type_str == "01": image_type = ImageType.RGB
    elif image_type_str == "03": image_type: ImageType.DEPTH
    else: raise Exception(f"Unknown image type {image_type_str} in image name")
    return scene_type, scene_number, scan_number, image_index, camera_index, image_type


def change_image_type(image_name, image_type: ImageType):
    if image_type == ImageType.RGB: suffix = "01.jpg"
    elif image_type == ImageType.DEPTH: suffix = "03.png"
    else: raise Exception(f"Unknown image_type: {image_type}")
    return image_name[:13] + suffix


def parse_scene_name(scene_name):
    scene_type_str, scene_number, scan_number = scene_name.split("_")
    scene_number = int(scan_number)
    scan_number = int(scan_number)
    scene_type = None
    if scene_type_str == "Home": scene_type = SceneType.HOME
    elif scene_type_str == "Office": scene_type: SceneType.OFFICE
    else: raise Exception(f"Unknown scene type {scene_type_str} in image name")
    return scene_type, scene_number, scan_number


def parse_mat_structs(structs_path):
    if os.path.exists(structs_path + ".pk") and os.path.exists(structs_path + ".txt"):
        image_structs = pd.read_pickle(structs_path + ".pk")
        with open(structs_path + ".txt", "r") as f:
            scale = int(f.readline().strip())
    else:
        data = loadmat(structs_path + ".mat")
        scale = data['scale']
        image_structs = pd.DataFrame.from_dict(data['image_structs'])
        image_structs.to_pickle(structs_path + ".pk")
        with open(structs_path + ".txt", "w") as f:
            f.write(str(scale))
    return image_structs, scale


def parse_camera_params(camera_params_path):
    with open(camera_params_path, "r") as f:
        params = [float(e) for e in f.read().split("\n")[3].split(" ")[2:]]

    width, height = tuple(map(int, params[:2]))
    f_x, f_y, c_x, c_y = params[2:6]
    distort_coeffs = tuple(params[6:])
    return width, height, f_x, f_y, c_x, c_y, distort_coeffs

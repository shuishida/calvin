import argparse
import os
import sys

import torch
from einops import rearrange
from torch.utils.data import DataLoader

sys.path.append('.')

from core.domains.avd.dataset.scene_loader import AVDLoadImages
from core.models.detector.resnet.resnet_feature_extractor import PretrainedResNetShort


def compute_resnet_embeddings(data=None, device=None, batch_size=None, avd_workers=None, resize=None, **kwargs):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    scene_names = [d for d in os.listdir(data) if os.path.isdir(os.path.join(data, d))]

    resnet = PretrainedResNetShort(freeze=True, device=device)
    resnet.eval()

    for scene_name in scene_names:
        print(f"Processing {scene_name}...")
        scene_path = os.path.join(data, scene_name)
        rgb_dir = os.path.join(scene_path, 'jpg_rgb')
        image_names = sorted(list(name for name in os.listdir(rgb_dir) if name.split('.')[-1] in ['jpg', 'png']))

        loader = DataLoader(AVDLoadImages(scene_path, image_names, resize or (1080, 1920)),
                            batch_size=batch_size, num_workers=avd_workers)
        embeddings = []

        with torch.no_grad():
            batch_sum = 0
            for i, (images, _) in enumerate(loader):
                batch_sum += len(images)
                sys.stdout.write(f"\r--- {batch_sum} / {len(image_names)} images")
                sys.stdout.flush()

                images = images.to(device)
                images = rearrange(images, "b h w f -> b f h w").float() / 255

                embeddings.append(resnet(images))

            embeddings = torch.cat(embeddings)

        print()
        print("Embedding shape: ", embeddings.size())
        _, _, H, W = embeddings.size()

        embeddings_path = os.path.join(scene_path, f"resnet_{H}x{W}.pt")

        torch.save(embeddings.cpu().type(torch.float16), embeddings_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help="path to data directory")
    parser.add_argument('--device', default="cuda:0", help="device")
    parser.add_argument('--batch_size', '-bs', default=32, type=int, help="batch size")
    parser.add_argument("--avd_workers", type=int, default=8, help="store embeddings in ram")
    parser.add_argument("--resize", "-sz", help="resize images", type=int, nargs=2, default=None)
    args = parser.parse_args()

    compute_resnet_embeddings(**vars(args))


"""
python core/domains/avd/dataset/precompute_resnet.py data/avd/src/
"""

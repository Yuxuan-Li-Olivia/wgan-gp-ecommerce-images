import argparse
import os

import torch
from torchvision import utils

from train_gan import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a trained WGAN-GP generator.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/mnt/e/E-commerce Product Images/gan/outputs")
    parser.add_argument("--num_images", type=int, default=64)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    nz = ckpt_args.get("nz", args.nz)
    ngf = ckpt_args.get("ngf", args.ngf)
    image_size = ckpt_args.get("image_size", 64)

    net_g = Generator(nz, ngf, 3, image_size).to(device)
    net_g.load_state_dict(ckpt["net_g"])
    net_g.eval()

    noise = torch.randn(args.num_images, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = net_g(noise).detach().cpu()

    grid_path = os.path.join(args.out_dir, "generated_grid.png")
    utils.save_image(fake, grid_path, nrow=8, normalize=True, value_range=(-1, 1))
    print(f"Saved samples to {grid_path}")


if __name__ == "__main__":
    main()

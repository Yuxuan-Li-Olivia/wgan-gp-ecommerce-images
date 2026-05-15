import argparse
import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from train_gan import Generator, collect_image_paths, seed_everything


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate WGAN-GP with FID and Inception Score.")
    parser.add_argument("--data_root", type=str, default="/mnt/e/E-commerce Product Images/data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_real", type=int, default=1000)
    parser.add_argument("--num_fake", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_path", type=str, default="/mnt/e/E-commerce Product Images/gan/outputs/eval_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_paths = collect_image_paths(args.data_root)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.data_root}")

    if args.num_real > len(image_paths):
        args.num_real = len(image_paths)

    real_paths = random.sample(image_paths, args.num_real)
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    real_dataset = ImagePathDataset(real_paths, transform=transform)
    real_loader = DataLoader(
        real_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    nz = ckpt_args.get("nz", 100)
    ngf = ckpt_args.get("ngf", 64)
    image_size = ckpt_args.get("image_size", args.image_size)

    net_g = Generator(nz, ngf, 3, image_size).to(device)
    net_g.load_state_dict(ckpt["net_g"])
    net_g.eval()

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore(splits=10).to(device)

    with torch.no_grad():
        for real in real_loader:
            real = real.to(device)
            fid.update((real * 255).byte(), real=True)

        fake_generated = 0
        while fake_generated < args.num_fake:
            current_batch = min(args.batch_size, args.num_fake - fake_generated)
            noise = torch.randn(current_batch, nz, 1, 1, device=device)
            fake = net_g(noise)
            fake = (fake + 1) / 2
            fake = fake.clamp(0, 1)
            fake_uint8 = (fake * 255).byte()
            fid.update(fake_uint8, real=False)
            inception.update(fake_uint8)
            fake_generated += current_batch

    fid_value = float(fid.compute().item())
    is_mean, is_std = inception.compute()
    is_mean = float(is_mean.item())
    is_std = float(is_std.item())

    results = {
        "fid": fid_value,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "num_real": args.num_real,
        "num_fake": args.num_fake,
        "image_size": image_size,
    }

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

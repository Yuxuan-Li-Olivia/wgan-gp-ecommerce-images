import argparse
import glob
import os
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
from PIL import Image


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


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, image_size):
        super().__init__()
        if image_size < 64 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size must be a power of 2 and >= 64")

        num_upsamples = int(torch.log2(torch.tensor(image_size)).item()) - 2
        channels = [ngf * 8]
        for _ in range(num_upsamples - 1):
            next_ch = max(ngf // 2, channels[-1] // 2)
            channels.append(next_ch)

        layers = [
            nn.ConvTranspose2d(nz, channels[0], 4, 1, 0, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True),
        ]

        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            ])

        layers.extend([
            nn.ConvTranspose2d(channels[-1], nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        ])

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Critic(nn.Module):
    def __init__(self, ndf, nc, image_size):
        super().__init__()
        if image_size < 64 or image_size & (image_size - 1) != 0:
            raise ValueError("image_size must be a power of 2 and >= 64")

        num_downsamples = int(torch.log2(torch.tensor(image_size)).item()) - 2
        channels = [ndf]
        for _ in range(num_downsamples - 1):
            channels.append(min(ndf * 16, channels[-1] * 2))

        layers = [
            nn.Conv2d(nc, channels[0], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        layers.append(nn.Conv2d(channels[-1], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x).view(-1)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_image_paths(data_root):
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(os.path.join(data_root, pattern), recursive=True))
    return sorted(paths)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a WGAN-GP on the product image dataset.")
    parser.add_argument("--data_root", type=str, default="/mnt/e/E-commerce Product Images/data")
    parser.add_argument("--out_dir", type=str, default="/mnt/e/E-commerce Product Images/gan/outputs")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--gp_lambda", type=float, default=10.0)
    parser.add_argument("--nz", type=int, default=100)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--checkpoint_every", type=int, default=5)
    return parser.parse_args()


def gradient_penalty(critic, real, fake, device):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates)
    grad_outputs = torch.ones_like(critic_interpolates, device=device)
    gradients = autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    seed_everything(args.seed)

    image_paths = collect_image_paths(args.data_root)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.data_root}")

    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = ImagePathDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = 3

    net_g = Generator(args.nz, args.ngf, nc, args.image_size).to(device)
    net_d = Critic(args.ndf, nc, args.image_size).to(device)
    net_g.apply(weights_init)
    net_d.apply(weights_init)

    optimizer_d = optim.Adam(
        net_d.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    optimizer_g = optim.Adam(
        net_g.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    for epoch in range(1, args.epochs + 1):
        net_g.train()
        net_d.train()
        for i, real_images in enumerate(dataloader, 1):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            for _ in range(args.n_critic):
                net_d.zero_grad(set_to_none=True)
                noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
                fake_images = net_g(noise).detach()

                critic_real = net_d(real_images)
                critic_fake = net_d(fake_images)
                wasserstein_distance = critic_real.mean() - critic_fake.mean()
                gp = gradient_penalty(net_d, real_images, fake_images, device)
                loss_d = -wasserstein_distance + args.gp_lambda * gp
                loss_d.backward()
                optimizer_d.step()

            net_g.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake_images = net_g(noise)
            loss_g = -net_d(fake_images).mean()
            loss_g.backward()
            optimizer_g.step()

            if i % 50 == 0 or i == len(dataloader):
                print(
                    f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(dataloader)}] "
                    f"Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} GP: {gp.item():.4f}"
                )

        if epoch % args.sample_every == 0:
            with torch.no_grad():
                fake = net_g(fixed_noise).detach().cpu()
            grid_path = os.path.join(args.out_dir, f"sample_epoch_{epoch:03d}.png")
            utils.save_image(fake, grid_path, nrow=8, normalize=True, value_range=(-1, 1))

        if epoch % args.checkpoint_every == 0:
            ckpt = {
                "epoch": epoch,
                "net_g": net_g.state_dict(),
                "net_d": net_d.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"checkpoint_{epoch:03d}.pt"))

    final_path = os.path.join(args.out_dir, "checkpoint_final.pt")
    torch.save({"net_g": net_g.state_dict(), "args": vars(args)}, final_path)
    print(f"Training complete. Final checkpoint saved to {final_path}")


if __name__ == "__main__":
    main()

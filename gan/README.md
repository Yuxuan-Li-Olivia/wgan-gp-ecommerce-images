# WGAN-GP Product Image Generator

A small WGAN-GP project for generating product images from the e-commerce dataset.

## Project Structure
- train_gan.py: WGAN-GP training script
- generate_samples.py: Sample generation from a saved checkpoint
- eval_metrics.py: FID / Inception Score evaluation
- report.md: Short experiment report
- requirements.txt: Python dependencies
- outputs/: Training samples and checkpoints (auto-created)

## Quick Start
1. Install dependencies:
   python -m pip install -r requirements.txt
2. Train:
   python train_gan.py
3. Generate samples:
   python generate_samples.py --checkpoint /mnt/e/E-commerce Product Images/gan/outputs/checkpoint_final.pt
4. Evaluate FID / IS:
   python eval_metrics.py --checkpoint /mnt/e/E-commerce Product Images/gan/outputs/checkpoint_final.pt

## Notes
- Default data root: /mnt/e/E-commerce Product Images/data
- Output folder: /mnt/e/E-commerce Product Images/gan/outputs
- Defaults target higher quality: 128x128, batch size 32, 60 epochs.
- You can tune --image_size, --batch_size, --n_critic, and --gp_lambda for quality vs speed.

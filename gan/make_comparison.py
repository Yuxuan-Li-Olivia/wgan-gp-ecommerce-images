from PIL import Image, ImageDraw, ImageFont


OUTPUT_64 = "/mnt/e/E-commerce Product Images/gan/outputs_64/generated_grid.png"
OUTPUT_128 = "/mnt/e/E-commerce Product Images/gan/outputs/generated_grid.png"
OUT_PATH = "/mnt/e/E-commerce Product Images/gan/results/compare_64_128.png"


def add_label(image, text):
    canvas = Image.new("RGB", (image.width, image.height + 36), (255, 255, 255))
    canvas.paste(image, (0, 36))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), text, fill=(0, 0, 0), font=font)
    return canvas


def main():
    img_64 = Image.open(OUTPUT_64).convert("RGB")
    img_128 = Image.open(OUTPUT_128).convert("RGB")

    target_height = max(img_64.height, img_128.height)
    if img_64.height != target_height:
        new_width = int(img_64.width * (target_height / img_64.height))
        img_64 = img_64.resize((new_width, target_height), Image.LANCZOS)
    if img_128.height != target_height:
        new_width = int(img_128.width * (target_height / img_128.height))
        img_128 = img_128.resize((new_width, target_height), Image.LANCZOS)

    img_64 = add_label(img_64, "64x64")
    img_128 = add_label(img_128, "128x128")

    combined = Image.new("RGB", (img_64.width + img_128.width, max(img_64.height, img_128.height)), (255, 255, 255))
    combined.paste(img_64, (0, 0))
    combined.paste(img_128, (img_64.width, 0))

    combined.save(OUT_PATH)
    print(f"Saved comparison to {OUT_PATH}")


if __name__ == "__main__":
    main()

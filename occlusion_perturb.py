import cv2
import numpy as np
import argparse
import os
import random
import json
from tqdm import tqdm

def add_random_colored_bars(img, min_bars=1, max_bars=5, thickness_range=(10, 100), length_range=(50, 500)):
    """
    Adds a random number of colored bars to an image.

    Parameters:
    - img: Input image (numpy array)
    - min_bars: Minimum number of bars
    - max_bars: Maximum number of bars
    - thickness_range: Min and max thickness of the bars
    - length_range: Min and max length of the bars

    Returns:
    - img_with_bars: Image with added occlusions
    - occlusions: List of occlusion details (position, thickness, length, color)
    """
    height, width, _ = img.shape
    img_with_bars = img.copy()
    occlusions = []

    num_bars = random.randint(min_bars, max_bars)  # Random number of bars for this image

    for _ in range(num_bars):
        orientation = random.choice(["vertical", "horizontal"])
        thickness = random.randint(*thickness_range)
        length = random.randint(*length_range)
        color = [random.randint(0, 255) for _ in range(3)]  # Random BGR color

        if orientation == "vertical":
            x_pos = random.randint(0, width - thickness)
            y_start = random.randint(0, height - length)
            y_end = y_start + length
            cv2.rectangle(img_with_bars, (x_pos, y_start), (x_pos + thickness, y_end), color, -1)
            occlusions.append({"orientation": "vertical", "x_pos": x_pos, "y_start": y_start, "y_end": y_end, "thickness": thickness, "color": color})
        
        else:  # Horizontal
            y_pos = random.randint(0, height - thickness)
            x_start = random.randint(0, width - length)
            x_end = x_start + length
            cv2.rectangle(img_with_bars, (x_start, y_pos), (x_end, y_pos + thickness), color, -1)
            occlusions.append({"orientation": "horizontal", "y_pos": y_pos, "x_start": x_start, "x_end": x_end, "thickness": thickness, "color": color})

    return img_with_bars, occlusions

def process_images(input_folder, output_folder, log_file, min_bars=1, max_bars=5):
    os.makedirs(output_folder, exist_ok=True)
    log_data = []

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with tqdm(total=len(image_files), desc="Processing Images", unit="img") as pbar:
        for filename in image_files:
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Skipping {filename}: Unable to read image.")
                pbar.update(1)
                continue
            
            img_with_bars, occlusions = add_random_colored_bars(img, min_bars, max_bars)
            cv2.imwrite(output_path, img_with_bars)

            log_data.append({
                "filename": filename,
                "num_bars": len(occlusions),
                "occlusions": occlusions
            })

            pbar.update(1)  # Update progress bar

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"\nProcessed images saved to {output_folder}")
    print(f"Occlusion log saved to {log_file}")

def main():
    parser = argparse.ArgumentParser(description="Add random colored bars (occlusions) to images in a folder.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save occluded images.")
    parser.add_argument("log_file", type=str, help="Path to save the JSON log file of occlusions.")
    parser.add_argument("--min_bars", type=int, default=1, help="Minimum number of bars per image (default: 1).")
    parser.add_argument("--max_bars", type=int, default=5, help="Maximum number of bars per image (default: 5).")

    args = parser.parse_args()
    process_images(args.input_folder, args.output_folder, args.log_file, args.min_bars, args.max_bars)

if __name__ == "__main__":
    main()
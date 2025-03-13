import cv2
import numpy as np
import argparse
import os
import random
import json
from tqdm import tqdm  # Import progress bar library

def change_temperature(img, kelvin_factor):
    kelvin_table = {
        1000: (255, 56, 0),  # Warm
        4000: (255, 209, 163),
        6500: (255, 255, 255),  # Neutral
        9000: (201, 226, 255),  # Cool
    }
    
    r, g, b = kelvin_table.get(kelvin_factor, (255, 255, 255))
    balance_matrix = np.array([b / 255.0, g / 255.0, r / 255.0], dtype=np.float32).reshape((1, 1, 3))
    
    img_float = img.astype(np.float32) / 255.0
    img_adjusted = img_float * balance_matrix  # Element-wise multiplication
    return (np.clip(img_adjusted * 255, 0, 255)).astype(np.uint8), {"kelvin_factor": kelvin_factor}

def day_to_night(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 2] = hsv[..., 2] * 0.4  # Reduce brightness
    img_night = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_night = cv2.addWeighted(img_night, 0.8, np.full_like(img, (255, 150, 100)), 0.2, 0)  # Add blue tint
    return img_night, {}

def adjust_color_tone(img, shift):
    b, g, r = cv2.split(img)
    b = cv2.add(b, shift[0])
    g = cv2.add(g, shift[1])
    r = cv2.add(r, shift[2])
    return cv2.merge((b, g, r)), {"b_shift": shift[0], "g_shift": shift[1], "r_shift": shift[2]}

def apply_tone_mapping(img):
    return cv2.detailEnhance(img, sigma_s=12, sigma_r=0.15), {}

def process_images(input_folder, output_folder, log_file):
    os.makedirs(output_folder, exist_ok=True)
    log_data = []

    transformations = [
        ("temperature", lambda img: change_temperature(img, random.choice([1000, 4000, 6500, 9000]))),
        ("night", lambda img: (day_to_night(img))),
        ("color_tone", lambda img: adjust_color_tone(img, (
            random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30)
        ))),
        ("tone_mapping", lambda img: (apply_tone_mapping(img)))
    ]

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
            
            transformation_name, transformation_func = random.choice(transformations)
            img_transformed, params = transformation_func(img)
            cv2.imwrite(output_path, img_transformed)

            log_data.append({
                "filename": filename,
                "transformation": transformation_name,
                "parameters": params
            })

            pbar.update(1)  # Update progress bar

    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=4)

    print(f"\nProcessed images saved to {output_folder}")
    print(f"Transformation log saved to {log_file}")

def main():
    parser = argparse.ArgumentParser(description="Randomly disturb images in a folder.")
    parser.add_argument("input_folder", type=str, help="Path to the input folder containing images.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder to save disturbed images.")
    parser.add_argument("log_file", type=str, help="Path to save the JSON log file of applied transformations.")

    args = parser.parse_args()
    process_images(args.input_folder, args.output_folder, args.log_file)

if __name__ == "__main__":
    main()
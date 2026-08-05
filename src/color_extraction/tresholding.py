import cv2
import numpy as np
import os

input_folder = r"D:/SKRIPSI/dataset_train/0 pelat karat"
output_hsv_folder  = r"D:/SKRIPSI/hasil latih/variasi/hsv/train/karat_hsv"
output_mask_folder = r"D:/SKRIPSI/hasil latih/variasi/hsv/train/tresholding/karat_mask"

os.makedirs(output_hsv_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

h_min, h_max = 0, 177
s_min, s_max = 13, 255
v_min, v_max = 14, 110

def apply_hsv_range(img, h_min, h_max, s_min, s_max, v_min, v_max):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    return hsv, mask

def process_images(input_folder,
                   output_hsv_folder, output_mask_folder,
                   h_min, h_max, s_min, s_max, v_min, v_max):

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(input_folder, filename)

            img = cv2.imread(img_path)
            if img is None:
                print(f"Gambar tidak ditemukan: {filename}")
                continue

            hsv_img, mask = apply_hsv_range(
                img, h_min, h_max, s_min, s_max, v_min, v_max
            )

            name, ext = os.path.splitext(filename)

            # Simpan HSV di folder khusus
            hsv_out = os.path.join(output_hsv_folder, f"{name}_hsv{ext}")
            cv2.imwrite(hsv_out, hsv_img)

            # Simpan mask di folder lain
            mask_out = os.path.join(output_mask_folder, f"{name}_mask{ext}")
            cv2.imwrite(mask_out, mask)

            print(f"Disimpan HSV:  {hsv_out}")
            print(f"Disimpan mask: {mask_out}")

process_images(input_folder,
               output_hsv_folder, output_mask_folder,
               h_min, h_max, s_min, s_max, v_min, v_max)

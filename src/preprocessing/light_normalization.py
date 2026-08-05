import cv2
import numpy as np
import os

input_folder = r"D:/SKRIPSI/dataset_test/pelat baik"
output_folder = r"D:/SKRIPSI/dataset_test/3 pelat baik/"

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

gamma = 2.0
gamma_exp = 2.75 / gamma   # sesuai program kamu

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg')):
        
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Gagal membaca: {filename}")
            continue

        # Gamma correction
        gamma_correction = np.array(
            255 * (img / 255) ** gamma_exp,
            dtype='uint8'
        )

        # Simpan ke folder output
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, gamma_correction)

        print(f"Selesai: {filename}")

print("=== Semua gambar selesai diproses ===")

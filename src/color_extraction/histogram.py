from fileinput import filename
import os
import cv2
import numpy as np
import pandas as pd
import logging
import re
from typing import Dict, Tuple

def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', s)]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

BINS_H, BINS_S, BINS_V = 36, 32, 32

CLASS_MAPPING = {
    'karat': 0,
    'pengelupasan_cat': 1,
    'sambungan_las': 2,
    'baik': 3
}

def validate_image(img_path: str) -> bool:
    if not os.path.exists(img_path):
        logger.error(f"File tidak ditemukan: {img_path}")
        return False
    
    if not img_path.lower().endswith(('.jpg')):
        logger.error(f"Format file tidak didukung: {img_path}")
        return False
    
    return True

def load_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)

    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def extract_hsv_features(img_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
      
    mask = load_mask(mask_path)
    if mask.shape[:2] != img.shape[:2]:
        raise ValueError(f"Ukuran mask dan gambar harus sama: {img_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    idx = mask == 255
    if not np.any(idx):
        raise ValueError(f"Mask kosong di: {mask_path}")

    h = hsv[..., 0][idx]
    s = hsv[..., 1][idx]
    v = hsv[..., 2][idx]

    mean_H = h.mean() / 179.0
    mean_S = s.mean() / 255.0
    mean_V = v.mean() / 255.0

    std_H = h.std() / 179.0
    std_S = s.std() / 255.0
    std_V = v.std() / 255.0

    mean_std = np.array([
        mean_H, mean_S, mean_V,
        std_H, std_S, std_V
    ], dtype=np.float32)

    hh, _ = np.histogram(h, bins=BINS_H, range=(0, 180))
    hs, _ = np.histogram(s, bins=BINS_S, range=(0, 256))
    hv, _ = np.histogram(v, bins=BINS_V, range=(0, 256))

    hh = hh / (hh.sum() + 1e-8)
    hs = hs / (hs.sum() + 1e-8)
    hv = hv / (hv.sum() + 1e-8)

    hist_vector = np.concatenate([hh, hs, hv]).astype(np.float32)

    return mean_std, hist_vector

def process_images_hsv_with_existing_masks(
    class_folders_rgb: Dict[str, str],
    class_folders_mask: Dict[str, str],
    output_csv_path: str,
    use_mask_suffix: bool = False,   # True kalau nama mask pakai _mask
    mask_suffix: str = "_mask"
) -> Dict:

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    stats = {"success": 0, "failed": 0, "per_class": {}}
    all_features = []

    csv_columns = [
    'image_file', 'class',
    'mean_H', 'mean_S', 'mean_V',
    'std_H', 'std_S', 'std_V']
    csv_columns += [f"H_{i+1}" for i in range(BINS_H)]
    csv_columns += [f"S_{i+1}" for i in range(BINS_S)]
    csv_columns += [f"V_{i+1}" for i in range(BINS_V)]

    for class_name, rgb_folder in class_folders_rgb.items():
        mask_folder = class_folders_mask.get(class_name, None)

        if mask_folder is None:
            logger.error(f"Folder mask untuk class '{class_name}' belum didefinisikan.")
            stats["per_class"][class_name] = {"success": 0, "failed": 0}
            continue

        if not os.path.exists(rgb_folder):
            logger.error(f"Folder RGB class '{class_name}' tidak ditemukan: {rgb_folder}")
            stats["per_class"][class_name] = {"success": 0, "failed": 0}
            continue

        if not os.path.exists(mask_folder):
            logger.error(f"Folder MASK class '{class_name}' tidak ditemukan: {mask_folder}")
            stats["per_class"][class_name] = {"success": 0, "failed": 0}
            continue

        image_files = sorted(
        [ f for f in os.listdir(rgb_folder)
        if f.lower().endswith(('.jpg'))],key=natural_sort_key)

        if not image_files:
            logger.error(f"Tidak ada file citra di folder RGB class '{class_name}'")
            stats["per_class"][class_name] = {"success": 0, "failed": 0}
            continue

        total_images = len(image_files)
        class_success = 0
        class_failed = 0

        logger.info(f"\nMemproses class '{class_name}': {total_images} citra")

        for filename in image_files:
            try:
                rgb_path = os.path.join(rgb_folder, filename)
                if not validate_image(rgb_path):
                    raise ValueError("Validasi file gagal")

                if use_mask_suffix:
                    name, ext = os.path.splitext(filename)
                    mask_filename = f"{name}{mask_suffix}{ext}"

                else:
                    # nama mask sama persis dengan nama RGB
                    mask_filename = filename

                mask_path = os.path.join(mask_folder, mask_filename)

                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask tidak ditemukan: {mask_path}")

                mean, hist_vector = extract_hsv_features(rgb_path, mask_path)

                class_label = CLASS_MAPPING[class_name]
                row = [filename, class_label] + mean.tolist() + hist_vector.tolist()

                all_features.append(row)

                stats["success"] += 1
                class_success += 1

            except Exception as e:
                stats["failed"] += 1
                class_failed += 1
                logger.error(f"GAGAL ({class_name}): {filename} - {str(e)}")
                continue

        stats["per_class"][class_name] = {"success": class_success, "failed": class_failed}
        logger.info(f"  ✓ Berhasil: {class_success}, ✗ Gagal: {class_failed}")

    df = pd.DataFrame(all_features, columns=csv_columns)
    logger.info(f"Jumlah fitur tersimpan (baris CSV): {len(all_features)}")
    if len(all_features) == 0:
        logger.warning("Tidak ada fitur yang berhasil diproses. CSV akan kosong / tidak berguna.")

    try:
        df.to_csv(output_csv_path, index=False)
        logger.info(f"CSV berhasil disimpan: {output_csv_path}")
    except Exception as e:
        logger.error(f"Gagal menyimpan CSV: {str(e)}")
        raise

    logger.info("\n" + "="*70)
    logger.info("RINGKASAN PEMROSESAN HSV + MASK EROSI MULTI-CLASS")
    logger.info("="*70)
    logger.info(f"Total citra diproses: {stats['success'] + stats['failed']}")
    logger.info(f"Berhasil: {stats['success']}")
    logger.info(f"Gagal: {stats['failed']}")

    for class_name, class_stats in stats["per_class"].items():
        logger.info(f"  {class_name}: {class_stats['success']} ✓, {class_stats['failed']} ✗")

    total = stats['success'] + stats['failed']
    success_rate = (stats['success'] / total * 100) if total > 0 else 0
    logger.info(f"Success Rate: {success_rate:.2f}%")
    logger.info(f"CSV Output: {output_csv_path}")
    logger.info("="*70)

    return stats

if __name__ == "__main__":
    CLASS_FOLDERS_RGB = {
        'karat': r'D:/SKRIPSI/dataset_vald/0 pelat karat',
        'pengelupasan_cat': r'D:/SKRIPSI/dataset_vald/1 pelat pengelupasan cat',
        'sambungan_las': r'D:/SKRIPSI/dataset_vald/2 pelat sambungan las',
        'baik': r'D:/SKRIPSI/dataset_vald/3 pelat baik'}

    CLASS_FOLDERS_MASK = {
        'karat': r'D:\SKRIPSI\hasil validasi\closing\0 pelat karat',
        'pengelupasan_cat': r'D:\SKRIPSI\hasil validasi\closing\1 pelat pengelupasan cat',
        'sambungan_las': r'D:\SKRIPSI\hasil validasi\closing\2 pelat sambungan las',
        'baik': r'D:\SKRIPSI\hasil validasi\closing\3 pelat baik'}

    OUTPUT_CSV_PATH = r'D:\SKRIPSI\hasil validasi\histogram\HSV_fitur_validasi.csv'

    try:
        result = process_images_hsv_with_existing_masks(
            CLASS_FOLDERS_RGB,
            CLASS_FOLDERS_MASK,
            OUTPUT_CSV_PATH,
            use_mask_suffix=True,     
            mask_suffix="_mask"
        )
    except Exception as e:
        logger.critical(f"Program error: {str(e)}")

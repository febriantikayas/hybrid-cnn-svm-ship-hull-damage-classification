import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
import os
import re
import logging
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]  
GLCM_LEVELS = 256 
GLCM_FEATURES = ['contrast', 'correlation', 'energy', 'homogeneity']
TOTAL_FEATURES = len(GLCM_FEATURES) * len(GLCM_ANGLES) * len(GLCM_DISTANCES)  

def validate_image(img_path: str) -> bool:
    if not os.path.exists(img_path):
        logger.error(f"File tidak ditemukan: {img_path}")
        return False
    
    if not img_path.lower().endswith(('.jpg')):
        logger.error(f"Format file tidak didukung: {img_path}")
        return False
    
    return True

def extract_glcm_features(img: np.ndarray) -> Tuple[np.ndarray, Dict]:
    min_size = max(GLCM_DISTANCES) + 1
    if img.shape[0] < min_size or img.shape[1] < min_size:
        raise ValueError(f"Gambar terlalu kecil untuk GLCM (ukuran: {img.shape}, minimum: {min_size}x{min_size})")
    
    glcm = graycomatrix(
        img,
        distances=GLCM_DISTANCES,
        angles=GLCM_ANGLES,
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True
    )
    
    features_dict = {}
    feature_list = []
    
    for feature_name in GLCM_FEATURES:
        feature_matrix = graycoprops(glcm, feature_name)
        for d_idx, d in enumerate(GLCM_DISTANCES):
            for a_idx, angle in enumerate(GLCM_ANGLES):
                value = feature_matrix[d_idx, a_idx]
                feature_list.append(value)
    
    feature_vector = np.array(feature_list)
    
    return feature_vector, features_dict

def numeric_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else -1

def process_images_multi_class(
    class_folders: Dict[str, str],
    output_base_folder: str,
    output_csv_path: str,
) -> Dict:
    
    os.makedirs(output_base_folder, exist_ok=True)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    stats = {"success": 0, "failed": 0, "per_class": {}}
    all_features = []

    label_mapping = {
    'karat': 0,
    'pengelupasan_cat': 1,
    'sambungan_las': 2,
    'baik': 3}
    
    csv_columns = ['image_file', 'class']
    for feature_name in GLCM_FEATURES:
        for d in GLCM_DISTANCES:
            for angle in GLCM_ANGLES:
                angle_deg = int(np.degrees(angle))
                csv_columns.append(f'{feature_name}_d{d}_{angle_deg}deg')
    
    for class_name, input_folder in class_folders.items():
        
        if not os.path.exists(input_folder):
            error_msg = f"Class folder '{class_name}' tidak ditemukan: {input_folder}"
            logger.error(error_msg)
            stats["per_class"][class_name] = {"success": 0, "failed": 0}
            continue
        
        image_files = sorted(
        [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')],
        key=numeric_sort_key)
        
        if not image_files:
            logger.error(f"Tidak ada file citra di folder class '{class_name}'")
            stats["per_class"][class_name] = {"success": 0, "failed": 0}
            continue
        
        total_images = len(image_files)
        class_success = 0
        class_failed = 0
        
        logger.info(f"\nMemproses class '{class_name}': {total_images} citra")
        
        for filename in image_files:
            try:
                input_path = os.path.join(input_folder, filename)
                if not validate_image(input_path):
                    raise ValueError(f"Validasi file gagal")
        
                img = cv2.imread(input_path)
                if img is None:
                    raise ValueError("Gagal membaca file")

                img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
                feature_vector, _ = extract_glcm_features(img_grayscale)
        
                row = [filename, label_mapping[class_name]] + feature_vector.tolist()
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
    logger.info("RINGKASAN PEMROSESAN MULTI-CLASS")
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
    
    CLASS_FOLDERS = {
        'karat': r'D:\SKRIPSI\dataset_vald\\0 pelat karat',
        'pengelupasan_cat': r'D:\SKRIPSI\dataset_vald\1 pelat pengelupasan cat',
        'sambungan_las': r'D:\SKRIPSI\dataset_vald\2 pelat sambungan las',
        'baik': r'D:\SKRIPSI\dataset_vald\3 pelat baik'
    }
    
    OUTPUT_BASE_FOLDER = r'D:\SKRIPSI\hasil validasi\glcm\grayscale_vald'
    OUTPUT_CSV_PATH = r'D:\SKRIPSI\hasil validasi\glcm\GLCM_fitur_valdidasi.csv'
    
    try:
        result = process_images_multi_class(
            CLASS_FOLDERS,
            OUTPUT_BASE_FOLDER,
            OUTPUT_CSV_PATH,
        )
        
    except Exception as e:
        logger.critical(f"Program error: {str(e)}")

import os
import cv2

def resize_images(input_folder, output_folder):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Folder '{output_folder}' telah dibuat.")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            input_image_path = os.path.join(input_folder, filename)
            output_image_path = os.path.join(output_folder, filename)

            try:
                img = cv2.imread(input_image_path)
                img_resized = cv2.resize(img, (224, 224)) 
                cv2.imwrite(output_image_path, img_resized)
                print(f"Berhasil mengubah ukuran '{filename}'")
            except Exception as e:
                print(f"Gagal memproses '{filename}': {e}")

input_folder = 'D:/SKRIPSI/dataset/data validasi'
output_folder = 'D:/SKRIPSI/data_vald'

resize_images(input_folder, output_folder)

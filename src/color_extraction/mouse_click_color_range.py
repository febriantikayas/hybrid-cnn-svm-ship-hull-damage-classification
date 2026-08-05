import cv2
import pandas as pd
import os

img_path = r"D:/SKRIPSI/hasil latih/preprocessing/preprocessing_normalisasi/plate sambungan las/las_200.jpg"
csv_path = r"D:/SKRIPSI/hasil latih/preprocessing/hsv/las/hasil_hsv_las.csv"

img = cv2.imread(img_path)
if img is None:
    print("Gambar tidak ditemukan atau path salah")
    exit()

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=["H", "S", "V"])


def mouse_click(event, x, y):
    global df

    if event == cv2.EVENT_LBUTTONDOWN:
        h, s, v = hsv[y, x]

        print(f"H={h}, S={s}, V={v}")

        new_row = {"H": int(h), "S": int(s), "V": int(v)}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df.to_csv(csv_path, index=False)
        print("Data tersimpan di CSV")

cv2.imshow("Gambar", img)
cv2.setMouseCallback("Gambar", mouse_click)

cv2.waitKey(0)
cv2.destroyAllWindows()

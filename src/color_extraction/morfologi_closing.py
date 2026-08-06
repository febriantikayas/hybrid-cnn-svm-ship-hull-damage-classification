import numpy as np
import cv2
import os
from glob import glob

def erosi(F, H):
    th, lh = H.shape
    tf, lf = F.shape
    hotx = lh // 2
    hoty = th // 2

    Xh, Yh = [], []
    for y in range(th):
        for x in range(lh):
            if H[y, x] == 1:
                Xh.append(x - hotx)
                Yh.append(y - hoty)

    G = np.zeros((tf, lf), dtype=np.uint8)

    for y in range(tf):
        for x in range(lf):
            cocok = True
            for i in range(len(Xh)):
                xx = x + Xh[i]
                yy = y + Yh[i]
                if xx < 0 or yy < 0 or xx >= lf or yy >= tf:
                    cocok = False
                    break
                if F[yy, xx] != 1:
                    cocok = False
                    break
            if cocok:
                G[y, x] = 1
    return G


def dilasi(F, H):
    th, lh = H.shape
    tf, lf = F.shape
    hotx = lh // 2
    hoty = th // 2

    Xh, Yh = [], []
    for y in range(th):
        for x in range(lh):
            if H[y, x] == 1:
                Xh.append(x - hotx)
                Yh.append(y - hoty)

    G = np.zeros((tf, lf), dtype=np.uint8)

    for y in range(tf):
        for x in range(lf):
            if F[y, x] == 1:
                for i in range(len(Xh)):
                    xx = x + Xh[i]
                    yy = y + Yh[i]
                    if 0 <= xx < lf and 0 <= yy < tf:
                        G[yy, xx] = 1
    return G


def closing(F, H):
    Fd = dilasi(F, H)
    Fc = erosi(Fd, H)
    return Fc


if __name__ == "__main__":
    input_dir = 
    output_dir =  

    os.makedirs(output_dir, exist_ok=True)

    H = np.ones((3, 3), dtype=np.uint8)
    pad = H.shape[0] // 2

    image_paths = glob(os.path.join(input_dir, "*.jpg"))

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Gagal baca: {img_path}")
            continue
            
        BW = (img > 0).astype(np.uint8)

        BWp = np.pad(BW, pad_width=pad, mode='constant', constant_values=0)

        Gp = closing(BWp, H)

        G = Gp[pad:-pad, pad:-pad]

        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, G * 255)

        print(f"Disimpan: {output_path}")

    print("Proses selesai.")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

glcm_train = pd.read_csv()
hsv_train  = pd.read_csv()

train_df = glcm_train.merge(hsv_train, on=['image_file','class'])

X_train_trad = train_df.drop(['image_file','class'], axis=1).values
y_train = train_df['class'].values

glcm_val = pd.read_csv()
hsv_val  = pd.read_csv()

val_df = glcm_val.merge(hsv_val, on=['image_file','class'])

X_val_trad = val_df.drop(['image_file','class'], axis=1).values
y_val = val_df['class'].values

cnn_data = np.load()

X_train_cnn = cnn_data['X_train']
y_train_cnn = cnn_data['y_train']
train_files_cnn = cnn_data['train_files']

df_cnn_train = pd.DataFrame(X_train_cnn)
df_cnn_train['image_file'] = train_files_cnn
df_cnn_train['class'] = y_train_cnn

df_cnn_train['image_file'] = df_cnn_train['image_file'].apply(os.path.basename)
train_df = train_df.merge(df_cnn_train, on=['image_file','class'])

X_val_cnn = cnn_data['X_val']
y_val_cnn = cnn_data['y_val']
val_files_cnn = cnn_data['val_files']

df_cnn_val = pd.DataFrame(X_val_cnn)
df_cnn_val['image_file'] = val_files_cnn
df_cnn_val['class'] = y_val_cnn

df_cnn_val['image_file'] = df_cnn_val['image_file'].apply(os.path.basename)
val_df = val_df.merge(df_cnn_val, on=['image_file','class'])

print("Train GLCM awal:", len(glcm_train))
print("Train setelah semua merge:", len(train_df))

print("Val GLCM awal:", len(glcm_val))
print("Val setelah semua merge:", len(val_df))

X_train_combined = train_df.drop(['image_file','class'], axis=1).values
y_train = train_df['class'].values

X_val_combined = val_df.drop(['image_file','class'], axis=1).values
y_val = val_df['class'].values

feature_columns = train_df.drop(['image_file','class'], axis=1).columns

idx = 0 
sample_features = X_train_combined[idx]
sample_label = y_train[idx]
sample_file = train_df.iloc[idx]['image_file']

print(f"\nContoh satu sampel: {sample_file} (Label: {sample_label})")

formatted_features = ", ".join([f"{val:.6f}" for val in sample features])
print("\nNilai fitur urut:")
print(formatted_features)

print("\nShape Train Gabungan:", X_train_combined.shape)
print("Shape Val Gabungan:", X_val_combined.shape)

print("\nJumlah fitur sebelum merge:")
print("GLCM:", len(glcm_train.columns) - 2)  # minus image_file & class
print("HSV :", len(hsv_train.columns) - 2)
print("CNN :", X_train_cnn.shape[1])

print("\nJumlah fitur setelah merge:")
print("Gabungan:", X_train_combined.shape[1])

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_combined)
X_val_scaled   = scaler.transform(X_val_combined)

print("\nCek mean train (≈0):", np.mean(X_train_scaled))
print("Cek std train (≈1):", np.std(X_train_scaled))

feature_columns = train_df.drop(['image_file','class'], axis=1).columns

df_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
df_train_scaled['class'] = y_train

df_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_columns)
df_val_scaled['class'] = y_val

output_train_path = r"D:fitur_GHC_train_normalisasi.csv"
output_val_path   = r"D:fitur_GHC_val_normalisasi.csv"

df_train_scaled.to_csv(output_train_path, index=False)
df_val_scaled.to_csv(output_val_path, index=False)

print("\nJumlah kolom hasil normalisasi:", len(df_train_scaled.columns))
print("\nFile normalisasi berhasil disimpan")

svm = SVC(kernel='linear', C=)
svm.fit(X_train_scaled, y_train)

y_pred = svm.predict(X_val_scaled)

accuracy = accuracy_score(y_val, y_pred)
print("\nAccuracy: {:.6f}%".format(accuracy * 100))

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix GLCM+HSV+CNN ")
plt.tight_layout()
plt.show()

print("Jumlah data train akhir:", len(train_df))
print("Jumlah data val akhir:", len(val_df))

print("\nClassification Report:\n")
print(classification_report(y_val, y_pred, digits=4))

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = r"D:\SKRIPSI\hasil latih\model_cnn\best_model_baru.h5"
train_dir  = r"D:\SKRIPSI\dataset_train"
val_dir    = r"D:\SKRIPSI\dataset_vald"

model = tf.keras.models.load_model(model_path)

model(tf.zeros((1, 224, 224, 3)))

feature_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.get_layer("deep_feature").output
)

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False   # PENTING! supaya urutan tetap
)

print(train_gen.class_indices)

val_gen = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse',
    shuffle=False
)

print(train_gen.class_indices)

train_gen.reset()
val_gen.reset()

X_train = feature_model.predict(train_gen)
X_val   = feature_model.predict(val_gen)

y_train = train_gen.classes
y_val   = val_gen.classes

train_files = train_gen.filenames
val_files   = val_gen.filenames

# Ambil hanya nama file tanpa folder class
train_files = [os.path.basename(f) for f in train_files]
val_files   = [os.path.basename(f) for f in val_files]

np.savez(
    r"D:\SKRIPSI\hasil validasi\model_cnn\fitur_train_val.npz",
    X_train=X_train,
    y_train=y_train,
    train_files=train_files,
    X_val=X_val,
    y_val=y_val,
    val_files=val_files
)

print("Ekstraksi fitur selesai.")
print("Shape X_train:", X_train.shape)
print("Shape X_val  :", X_val.shape)
print("Jumlah train_files:", len(train_files))
print("Jumlah val_files:", len(val_files))

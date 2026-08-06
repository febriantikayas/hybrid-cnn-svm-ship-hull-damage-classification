import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt

OUT_DIR   = 
TRAIN_DIR = 
VAL_DIR   = 

os.makedirs(OUT_DIR, exist_ok=True)

datagen = ImageDataGenerator(rescale=1./255)

def make_gen(folder, shuffle):
    return datagen.flow_from_directory(
        folder,
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",
        shuffle=shuffle)

train_gen = make_gen(TRAIN_DIR, shuffle=True)
val_gen   = make_gen(VAL_DIR,   shuffle=False)

num_classes = train_gen.num_classes

print("Train class_indices:", train_gen.class_indices)
print("Val class_indices:", val_gen.class_indices)
print("Num classes:", num_classes)

model = Sequential([
    Conv2D(64, (3, 3), activation="relu", padding="same", 
           input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation="relu", padding="same"), 
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation="relu", padding="same"), 
    MaxPooling2D((2, 2)),

    GlobalAveragePooling2D(),

    Dense(512, activation="relu"), 
    Dropout(0.3),

    Dense(256, activation="relu"), 
    Dropout(0.3),

    Dense(512, activation="linear", name="deep_feature"),
    Dense(num_classes, activation="softmax", name="classifier"), 
    ], name="CNN_FeatureExtractor")

model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.1,
    restore_best_weights=True, verbose=2)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(OUT_DIR, ""), monitor='val_loss',
    save_best_only=True, verbose=2)

model.compile(
    optimizer=Adam(learning_rate=0.0003),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"])

history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=[early_stop, checkpoint])

acc      = history.history['accuracy']
val_acc  = history.history['val_accuracy']
loss     = history.history['loss']
val_loss = history.history['val_loss']

epoch_range = range(1, len(acc) + 1)

plt.figure()
plt.plot(epoch_range, acc)
plt.plot(epoch_range, val_acc)
plt.title("Grafik Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "accuracy_baru.png"))
plt.show()

plt.figure()
plt.plot(epoch_range, loss)
plt.plot(epoch_range, val_loss)
plt.title("Grafik Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "loss_baru.png"))
plt.show()

print("Total epoch yang direncanakan: 100")
print("Total epoch yang dijalankan :", len(acc))

final_model_path = os.path.join(OUT_DIR, "cnn_with_classifier_final_baru.h5")
model.save(final_model_path)

print("Best model saved at:", os.path.join(OUT_DIR, "best_model.h5"))
print("Final model saved at:", final_model_path)

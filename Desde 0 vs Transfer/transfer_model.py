import os
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import json

IMG_SIZE = 224
TRAIN_DIR = "train"
TEST_DIR = "test"
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

def load_data(data_dir):
    x = []
    y = []

    class_names = ["cats", "dogs"]

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            if not file_name.lower().endswith(VALID_EXTENSIONS):
                continue

            img_path = os.path.join(class_path, file_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            x.append(img)
            y.append(label)

    x = np.array(x, dtype="float32")
    x = preprocess_input(x)
    y = np.array(y, dtype="float32")
    return x, y

print("Cargando datos de entrenamiento...")
x_train, y_train = load_data(TRAIN_DIR)

print("Cargando datos de prueba...")
x_test, y_test = load_data(TEST_DIR)

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=10,
    batch_size=32
)

loss, acc = model.evaluate(x_test, y_test, verbose=0)

results = {
    "model": "transfer_learning",
    "test_loss": float(loss),
    "test_accuracy": float(acc),
    "final_train_accuracy": float(history.history["accuracy"][-1]),
    "final_val_accuracy": float(history.history["val_accuracy"][-1])
}

with open("transfer_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\nResultados guardados en transfer_results.json")
print(results)
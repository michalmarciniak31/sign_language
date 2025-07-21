import json
import cv2
import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam


with open(r"C:\Users\pichc\Downloads\FreiHAND_pub_v2\training_xyz.json") as f:
    xyz = json.load(f)

with open(r"C:\Users\pichc\Downloads\FreiHAND_pub_v2\training_K.json") as f:
    K = json.load(f)


def project_xyz_to_uv(xyz_points, K):
    uv_points = []
    for point in xyz_points:
        X, Y, Z = point
        if Z == 0:
            uv_points.append([0,0])
            continue
        x = X / Z
        y = Y / Z
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        u = fx * x + cx
        v = fy * y + cy
        uv_points.append(int(u))
        uv_points.append(int(v))
    return uv_points


landmarks = []
for i in range(len(K)):
    landmarks.append(project_xyz_to_uv(xyz[i], K[i]))


img_height, img_width = 224, 224
learning_rate = 1e-4
dropout_rate = 0.3
n_outputs = 42  # 21 punktów * 2 (x, y)

# Wejście: obraz RGB
image_input = Input(shape=(img_height, img_width, 3), name='image_input')
cnn_base = MobileNetV2(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg')
cnn_features = cnn_base(image_input)

# Dodanie Dense + Dropout
x = Dense(512, activation='relu')(cnn_features)
x = Dropout(dropout_rate)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(dropout_rate)(x)
x = Dense(128, activation='relu')(x)

# Wyjście: landmarki
landmark_output = Dense(n_outputs, activation='linear', name='landmark_output')(x)

# Budowa modelu
model = Model(inputs=image_input, outputs=landmark_output)

# Kompilacja modelu
optimizer = Adam(learning_rate=learning_rate)
loss = Huber(delta=10.0)  # bardziej odporny na outliery

model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

model.summary()

# EarlyStopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Trening modelu

def load_dataset(image_paths, landmark_data, img_size=(224,224)):
    images = []
    for path in image_paths:
        img = tf.keras.utils.load_img(path, target_size=img_size)
        img = tf.keras.utils.img_to_array(img) / 255.0
        images.append(img)
        print(path)
    images = np.array(images)
    landmarks = np.array(landmark_data, dtype=np.float32)/224.0

    return images, landmarks

final_images = []
final_landmarks = []

img_pth = []
for i in os.listdir(r'C:\Users\pichc\Downloads\FreiHAND_pub_v2\training\rgb'):
    img_pth.append(os.path.join(r'C:\Users\pichc\Downloads\FreiHAND_pub_v2\training\rgb', i))

final_images, final_landmarks = load_dataset(img_pth[:5000], landmarks[:5000])

history = model.fit(
    final_images,
    final_landmarks,
    validation_split=0.2,
    epochs=64,
    batch_size=64,
    callbacks=[early_stop]
)

img_pth_rgb = []
for i in os.listdir(r'C:\Users\pichc\Downloads\FreiHAND_pub_v2\training\rgb'):
    img_pth_rgb.append(os.path.join(r'C:\Users\pichc\Downloads\FreiHAND_pub_v2\training\rgb', i))
img_pth_mask = []
for i in os.listdir(r'C:\Users\pichc\Downloads\FreiHAND_pub_v2\training\mask'):
    img_pth_mask.append(os.path.join(r'C:\Users\pichc\Downloads\FreiHAND_pub_v2\training\mask', i))


def load_dataset_mask(image_paths, mask_paths, landmark_data, img_size=(224, 224)):
    images = []

    for img_path, mask_path in zip(image_paths, mask_paths):
        print(img_path)
        img = tf.keras.utils.load_img(img_path, target_size=img_size)
        img = tf.keras.utils.img_to_array(img) / 255.0

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = mask / 255.0  # zamiana na wartości 0-1
        mask = np.expand_dims(mask, axis=-1)  # shape (H, W, 1)

        # Rozszerz maskę do 3 kanałów
        mask_3c = np.repeat(mask, 3, axis=-1)  # shape (H, W, 3)

        # Zastosuj maskę do obrazu
        masked_img = img * mask_3c

        images.append(masked_img)

    images = np.array(images)
    landmarks = np.array(landmark_data, dtype=np.float32) / 224.0

    return images, landmarks

final_images_mask, final_landmarks_mask = load_dataset_mask(img_pth_rgb[:10000], img_pth_mask[:10000], landmarks[:10000])

history = model.fit(
    final_images_mask,
    final_landmarks_mask,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
)

# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Cannot access webcam.")
#     exit()
#
while True:
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Failed to grab frame.")
#         break
#
#     # Display the frame
    frame = cv2.imread(r"C:\Users\pichc\Downloads\FreiHAND_pub_v2\training\rgb\00000001.jpg")
#     #resized = cv2.resize(frame, (224, 224))
    pred_image = frame.astype('float32')
    pred_image = tf.keras.utils.img_to_array(pred_image) / 255.0
    input_frame = np.expand_dims(pred_image, axis=0)  # shape: (1,224,224,3)

    output = model.predict(input_frame) * 224
    output2 = []
    for i in range(0, len(output[0]), 2):
        output2.append([int(output[0][i]), int(output[0][i+1])])

    for (x, y) in output2:
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    cv2.imshow("frame", frame)

#    #Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the capture and close window
# cap.release()
# cv2.destroyAllWindows()

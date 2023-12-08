import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

#Can change to a value <= 200. If you processed a dataset larger than WLASL200, first change line 29 to the correct size.
dataset_size = 200

#Comment one of these lines out
#keypoint_system = 'mediapipe'
keypoint_system = 'openpose'

hand_keypoint_count = 21 * 2 * 2

if keypoint_system == 'mediapipe':
    pose_keypoint_count = 33 * 2
    face_keypoint_count = 468 * 2
else:
    pose_keypoint_count = 25 * 2
    face_keypoint_count = 70 * 2

X = pd.read_pickle('WLASL200_' + keypoint_system + '.pkl')
Y = X['label']
X = X.drop(columns=['label'])
example_count = len(X)

max_frame_count = 0

for idx in range(example_count):
    if len(X['hands'][idx]) > max_frame_count:
        max_frame_count = len(X['hands'][idx])

for idx in range(example_count):
    while len(X['hands'][idx]) < max_frame_count:
        X['hands'][idx] = np.concatenate((X['hands'][idx], X['hands'][idx]))
        X['face'][idx] = np.concatenate((X['face'][idx], X['face'][idx]))
        X['pose'][idx] = np.concatenate((X['pose'][idx], X['pose'][idx]))
    if len(X['hands'][idx]) > max_frame_count:
        X['hands'][idx] = X['hands'][idx][:max_frame_count]
        X['face'][idx] = X['face'][idx][:max_frame_count]
        X['pose'][idx] = X['pose'][idx][:max_frame_count]

hands = [np.delete(frame, 2, 2) for frame in X['hands']]
X = X.drop(columns=['hands'])
pose = [np.delete(frame, 2, 2) for frame in X['pose']]
X = X.drop(columns=['pose'])
face = [np.delete(frame, 2, 2) for frame in X['face']]
X = None

for idx in range(example_count):
    hands[idx] = hands[idx].reshape((max_frame_count, hand_keypoint_count))
    face[idx] = face[idx].reshape((max_frame_count, face_keypoint_count))
    pose[idx] = pose[idx].reshape((max_frame_count, pose_keypoint_count))

y_series = pd.Series(Y)

# Calculate the class frequencies
class_frequencies = y_series.value_counts()

# Identify the dataset_size most frequent glosses
selected_count = sum(class_frequencies[:dataset_size])

# Filter X and Y based on selected classes
hands = hands[:selected_count]
pose = pose[:selected_count]
face = face[:selected_count]
Y = Y[:selected_count]

X = np.dstack((np.stack(hands), np.stack(pose), np.stack(face)))

hands = None
pose = None
face = None

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42, stratify=Y)

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

for k_option in [1, 3, 5, 10]:
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(8, kernel_size=4, activation='relu', input_shape=(max_frame_count, hand_keypoint_count + pose_keypoint_count + face_keypoint_count)),
        tf.keras.layers.MaxPooling1D(pool_size=3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics= [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=k_option)]
             )

    model.fit(X_train, y_train_encoded, epochs=150, batch_size=32, validation_data=(X_test, y_test_encoded))

    test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
    print(f'Test accuracy: {test_acc} for k={k_option}')
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
tqdm.pandas()
import pickle
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Flatten, Embedding, Dropout, BatchNormalization


PATH = ''
with open(PATH+'/X_train_flat.pkl','rb') as f:
    X_train_flat = pickle.load(f)

with open(PATH+'/X_test_flat.pkl','rb') as f:
    X_test_flat = pickle.load(f)

with open(PATH+'/X_train_cood.pkl','rb') as f:
    X_train_cood = pickle.load(f)

with open(PATH+'/X_test_cood.pkl','rb') as f:
    X_test_cood = pickle.load(f)

with open(PATH+'/y_train.pkl','rb') as f:
    y_train = pickle.load(f)

with open(PATH+'/y_test.pkl','rb') as f:
    y_test = pickle.load(f)

X_train = X_train_flat
X_test = X_test_flat

label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

len(label_encoder.classes_)

X = np.concatenate((X_train, X_test), axis=0)
Y = np.concatenate((y_train_encoded, y_test_encoded), axis=0)
y_series = pd.Series(Y)
class_frequencies = y_series.value_counts()
selected_classes = class_frequencies[class_frequencies > 12].index
filtered_x = X[y_series.isin(selected_classes)]
filtered_y = Y[y_series.isin(selected_classes)]
filtered_y = np.array(filtered_y)

X_train, X_test, y_train, y_test = train_test_split(filtered_x, filtered_y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(y_train)
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
len(label_encoder.classes_)

# Model Building

# 1-D CNN

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, kernel_size=4, activation='relu', input_shape=(15, 150)),
    tf.keras.layers.MaxPooling1D(pool_size=3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics= [tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10)]
         )

model.fit(X_train, y_train_encoded, epochs=150, batch_size=32, validation_data=(X_test, y_test_encoded))

test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc}')

# BiLSTM

model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(15, 150)),
    Dropout(0.1),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.1),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32, return_sequences=True)),
    Bidirectional(LSTM(16, return_sequences=True)),
    Flatten(),
    Dense(1000, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)]
              )

model.fit(X_train, y_train_encoded, epochs=150, batch_size=32, validation_data=(X_test, y_test_encoded))

test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
print(f'Test accuracy: {test_acc}')




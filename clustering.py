# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm
tqdm.pandas()
import pickle
from sklearn.decomposition import PCA
from ipywidgets import interact, widgets, Layout, VBox
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Unzip the dk zip files post KPE

# Following is an example Google Colab code:

# !7z e /content/drive/MyDrive/dk-001.7z
# with open('/content/dk.pkl','rb') as f:
#    dk = pickle.load(f)
PATH = ''
with open(PATH,'rb') as f:
    dk = pickle.load(f)

# Calculating Num Clusters

hik = []
for i in range(len(dk)):
    hik.append(dk.hands[i].shape[0])

pik = []
for i in range(len(dk)):
    pik.append(dk.pose[i].shape[0])

fik = []
for i in range(len(dk)):
    fik.append(dk.face[i].shape[0])

max_clusters = min(min(hik), min(pik), min(fik))

max_clusters = 15

def calculate_wcss(data, max_clusters):
    wcss = []
    for i in tqdm(range(1, max_clusters + 1)):
        reshaped_data = data[:,:,:2].reshape(data.shape[0], -1)
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(reshaped_data)
        wcss.append(kmeans.inertia_)
    return wcss


# Calculate WCSS for each column ('pose', 'face', 'hands'), replace 0 with slider_input to observe for a sample
wcss_pose = calculate_wcss(dk['pose'][0], max_clusters)
wcss_face = calculate_wcss(dk['face'][0], max_clusters)
wcss_hands = calculate_wcss(dk['hands'][0], max_clusters)

# Plot the elbow curves for each column
plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.plot(range(1, max_clusters + 1), wcss_pose, marker='o')
plt.title('Pose Clustering - Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.subplot(132)
plt.plot(range(1, max_clusters + 1), wcss_face, marker='o')
plt.title('Face Clustering - Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.subplot(133)
plt.plot(range(1, max_clusters + 1), wcss_hands, marker='o')
plt.title('Hands Clustering - Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')

plt.tight_layout()
plt.show()

# Clustering 
# Replace KMeans function with GMM// Aggglomerative Clustering // Bisecting KMeans // Spectral

num_clusters = max_clusters  # Change this to the desired number of clusters

def cluster_column(data, num_clusters):

    reshaped_data = data[:,:,:2].reshape(data.shape[0], -1)

    #----------------------
    #### Cluster Begin
    #----------------------
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(reshaped_data)
    # Get Centroids
    centroids = kmeans.cluster_centers_
    #----------------------
    #### Cluster End
    #----------------------
    
    centroids_reshaped = centroids.reshape(num_clusters, data.shape[1], 2)
    # Sort centroids
    original_order = np.arange(len(data))
    valid_indices = np.arange(min(len(original_order), num_clusters))
    # Arrage with Cluster labels
    combined_data = [(centroids_reshaped[i], cluster_labels[i], original_order[i]) for i in valid_indices]
    sorted_data = sorted(combined_data, key=lambda x: x[2])
    # Extract sorted centroids
    sorted_centroids = np.array([item[0] for item in sorted_data])
    return sorted_centroids

# Apply clustering for 'pose', 'face', and 'hands' columns
dk['pose_clustered'] = [cluster_column(dk['pose'][i], num_clusters) for i in tqdm(range(len(dk)))]
dk['face_clustered'] = [cluster_column(dk['face'][i], num_clusters) for i in tqdm(range(len(dk)))]
dk['hands_clustered'] = [cluster_column(dk['hands'][i], num_clusters) for i in tqdm(range(len(dk)))]

dhand = dk[['hands_clustered', 'label']]
dpose = dk[['pose_clustered', 'label']]
dface = dk[['face_clustered', 'label']]

pose_numpy_array = np.stack(dpose['pose_clustered'].to_numpy())
hands_numpy_array = np.stack(dhand['hands_clustered'].to_numpy())
face_numpy_array = np.stack(dface['face_clustered'].to_numpy())

# Clear memory
dk = ''

# Data Augmentation
data_array = np.concatenate((pose_numpy_array, hands_numpy_array, face_numpy_array), axis=2)
y = dhand['label']
# Train test split
X_train, X_test, y_train, y_test = train_test_split(data_array, y, test_size=0.2, random_state=42)
# Get data back
pose_restored = X_train[:, :, :33, :]
hands_restored = X_train[:, :, 33:75, :]
face_restored = X_train[:, :, 75:543, :]

pose_test = X_test[:, :, :33, :]
hands_test = X_test[:, :, 33:75, :]
face_test = X_test[:, :, 75:543, :]

# Datagen
datagen = ImageDataGenerator(
    width_shift_range=0.05,
    height_shift_range=0.05,
    rotation_range=1,
    shear_range=0.1,
    preprocessing_function=lambda x: x + 0.1 * np.random.normal(0, np.std(x), x.shape),
)

# Data Augmentation Function
def Augment(var, num_clusters, y_train, num_aug):
    X = var
    X_reshaped = X.reshape(-1, var[0][0].shape[0], 2)
    X_reshaped = np.expand_dims(X_reshaped, axis=-1)
    y_repeated = np.repeat(y_train, num_clusters)
    datagen.fit(X_reshaped)
    # Generate augmented data
    augmented_data = []
    augmented_labels = []
    for X_batch, y_batch in datagen.flow(X_reshaped, y_repeated, batch_size=num_clusters*num_aug, shuffle=False, seed=42):
        augmented_data.append(X_batch)
        augmented_labels.append(y_batch)
    # Concatenate the original and augmented data
    X_augmented = np.vstack([X_reshaped, augmented_data[0]])
    y_augmented = np.concatenate([y_repeated, augmented_labels[0]])
    k_reaug = X_augmented.reshape((len(X)+10000),num_clusters,var[0][0].shape[0],2)
    y_reaug = y_augmented.reshape((len(X)+10000),num_clusters,)
    y_reaug = y_reaug[:,0]
    return k_reaug, y_reaug

num_aug = 10000
pose_reaug, y_reaug = Augment(pose_restored, num_clusters, y_train, num_aug)
hands_reaug, _ = Augment(hands_restored, num_clusters, y_train, num_aug)
face_reaug, _ = Augment(face_restored, num_clusters, y_train, num_aug)

data_augmented = np.concatenate((pose_reaug, face_reaug, hands_reaug), axis=2)
labels_augmented = y_reaug
data_test = np.concatenate((pose_test, face_test, hands_test), axis=2)
labels_test = y_test

# Pickle dump
'''
with open('/content/data_augmented.pkl','wb') as f:
    pickle.dump(data_augmented, f)

with open('/content/labels_augmented.pkl','wb') as f:
    pickle.dump(labels_augmented, f)

with open('/content/data_test.pkl','wb') as f:
    pickle.dump(data_test, f)

with open('/content/labels_test.pkl','wb') as f:
    pickle.dump(labels_test, f)
'''

# Principal Component Analysis
top_n = 30
X = face_reaug.reshape((-1, 468 * 2))
n_components_range = np.arange(1, min(X.shape[1], top_n))
explained_var_ratio = []
for n_components in tqdm(n_components_range):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    explained_var_ratio.append(np.sum(pca.explained_variance_ratio_))

# Plot the explained variance
plt.plot(n_components_range, explained_var_ratio, marker='o')
plt.title('Explained Variance vs. Num Components')
plt.xlabel('Num Components')
plt.ylabel('Explained Variance')
plt.show()

n_components = 10

# Fit PCA
face_reshaped_data = face_reaug.reshape((-1, 468 * 2))
face_reshaped_data_test = face_test.reshape((-1, 468 * 2))
pca_face = PCA(n_components=n_components)
pca_face.fit(face_reshaped_data)

# Transform PCA
face_reaug_pca = pca_face.transform(face_reshaped_data)
face_test_pca = pca_face.transform(face_reshaped_data_test)
face_reaug_pca = face_reaug_pca.reshape((face_reaug_pca.shape[0], num_clusters, n_components))
face_train = face_reaug_pca
face_test_pca = face_test_pca.reshape((face_test_pca[0], num_clusters, n_components))

hands_train = hands_reaug.reshape((face_reaug_pca.shape[0], num_clusters, 42* 2))
hands_test = hands_test.reshape((face_test_pca[0], num_clusters, 42* 2))
pose_train = pose_reaug.reshape((face_reaug_pca.shape[0], num_clusters, 33* 2))
pose_test = pose_test.reshape((face_test_pca[0], num_clusters, 33* 2))

y_train = labels_augmented
y_test = labels_test

X_train_flat = np.concatenate((face_train, hands_train, pose_train), axis=2)
X_test_flat = np.concatenate((face_test, hands_test, pose_test), axis=2)

X_train_cood = np.concatenate((hands_reaug, pose_reaug), axis=2)
X_test_cood = np.concatenate((hands_test, pose_test), axis=2)








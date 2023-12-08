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

# Unzip the dk zip files post KPE

# Following is an example Google Colab code:

# !7z e /content/drive/MyDrive/dk-001.7z
# with open('/content/dk.pkl','rb') as f:
#    dk = pickle.load(f)
PATH = ''
with open(PATH,'rb') as f:
    dk = pickle.load(f)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

dd = dk['pose'][69][0]

x = dd[:, 0]
y = dd[:, 1]
z = dd[:, 2]

# Visualize a gesture frame

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
scatter = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5))
fig.add_trace(scatter)
fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'))
fig.update_layout(scene_camera=dict(eye=dict(x=0, y=0, z=-1)))

def update(frame):
    dd = dk['pose'][69][frame]
    x = dd[:, 0]
    y = dd[:, 1]
    z = dd[:, 2]
    return [go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5))]

num_frames = len(dk['pose'][69])
animation_frames = [go.Frame(data=update(frame), name=f'frame_{frame}') for frame in range(num_frames)]
animation_layout = go.Layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
                                            method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])],
                            sliders=[{'active': 0,
                                      'steps': [{'args': [[f'frame_{frame}']],
                                                 'label': str(frame),
                                                 'method': 'animate'} for frame in range(num_frames)],
                                      'yanchor': 'top',
                                      'xanchor': 'left',
                                      'currentvalue': {'font': {'size': 16}, 'prefix': 'Frame: ', 'visible': True, 'xanchor': 'right'},
                                      'transition': {'duration': 300, 'easing': 'cubic-in-out'}}])
animation_figure = go.Figure(data=[scatter], frames=animation_frames, layout=animation_layout)
animation_figure.show()

# Visualize a gesture x,y,frame

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
for frame in range(len(dd)):
    scatter = go.Scatter3d(
        x=dd[frame][:, 0],
        y=dd[frame][:, 1],
        z=[frame] * len(dd[frame]),
        mode='markers',
        marker=dict(size=5)
    )
    fig.add_trace(scatter)
fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Frame'))
fig.show()

# Visualize post Clustering

dd = dk['pose'][69]
reshaped_data = np.empty((0, 3))

for frame in range(len(dd)):
    xy_coordinates = dd[frame, :, :2]
    frame_column = np.full((len(xy_coordinates), 1), frame)
    combined_data = np.hstack((xy_coordinates, frame_column))
    reshaped_data = np.vstack((reshaped_data, combined_data))

# Use K-means clustering
num_clusters = 7 
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(reshaped_data)

fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
scatter = go.Scatter3d(
    x=reshaped_data[:, 0],
    y=reshaped_data[:, 1],
    z=reshaped_data[:, 2],
    mode='markers',
    marker=dict(size=3, color=labels)
)
fig.add_trace(scatter)
fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Frame'))
fig.show()


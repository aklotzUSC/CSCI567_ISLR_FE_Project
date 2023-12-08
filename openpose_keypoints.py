# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import numpy as np
import json
import pandas as pd

#CHANGE THIS TO PATH OF OPENPOSE DOWNLOAD DIR
dir = 'C:\\Users\\andre\\Documents\\School\\USC\\CSCI 567\\Project\\'

#CHANGE THIS TO DIRECTORY OF VIDEO FILES
vid_dir = 'videos/WLASL2000'

MAX_GLOSS_COUNT = 200

# Windows Import
# Change these variables to point to the correct folder (Release/x64 etc.)
sys.path.append(
    dir + 'openpose\\build\\python\\openpose\\Release');
os.add_dll_directory(
    dir +'openpose\\build\\x64\\Release')
os.add_dll_directory(dir + 'openpose\\build\\bin')
import pyopenpose as op

# Flags
parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default=(dir + "openpose\\examples\\media\\COCO_val2014_000000000241.jpg"),
                    help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = dir + "openpose\\models"
params["face"] = True
params["hand"] = True

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-', '')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-', '')
        if key not in params: params[key] = next_item

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

def extract_landmarks_from_video(video_path, start_frame=1, end_frame=-1):
    cap = cv2.VideoCapture(video_path)
    if end_frame < 0:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 1:
        start_frame = 1

    body_landmarks = []
    l_hand_handmarks = []
    r_hand_handmarks = []
    face_landmarks = []


    while cap.isOpened() and start_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        datum = op.Datum()
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        if datum.poseKeypoints is not None:
            body_landmarks.append(datum.poseKeypoints[0])
            l_hand_handmarks.append(datum.handKeypoints[0][0])
            r_hand_handmarks.append(datum.handKeypoints[1][0])
            face_landmarks.append(datum.faceKeypoints[0])
        start_frame += 1
    cap.release()

    #70 face keypoints, 25 body, 21 per hand
    return np.hstack((np.array(body_landmarks), np.array(l_hand_handmarks), np.array(r_hand_handmarks), np.array(face_landmarks)))

def extract_save(input_path, start, end, vid_idx):
    # Check if the file exists at the path
    npy_path = f'numpy_openpose/{vid_idx}.npy'
    # Extract landmarks from video
    temp = extract_landmarks_from_video(input_path, start, end)
    # Process video and save as NumPy file
    np.save(npy_path, temp)

if __name__ == '__main__':
    if not os.path.isdir('numpy_openpose'):
        os.mkdir('numpy_openpose')

    with open('WLASL_v0.3.json', 'r') as json_file:
        data = json.load(json_file)

    data = sorted(data, key=lambda word: len(word['instances']), reverse=True)

    gloss_list = [] # done
    video_name_list = [] # done
    start_frame = [] # done
    end_frame = [] # done

    for word in data:
        for i in range(len(word['instances'])):
            vid_name = os.path.join(vid_dir, f'{word["instances"][i]["video_id"]}.mp4')
            start = word['instances'][i]['frame_start']
            end = word['instances'][i]['frame_end']
            label = word['gloss']

            video_name_list.append(vid_name)
            gloss_list.append(label)
            start_frame.append(start)
            end_frame.append(end)

    last_gloss = ''
    gloss_count = 0

    processed_count = len(video_name_list)

    for idx in range(len(video_name_list)):
        if gloss_list[idx] != last_gloss:
            last_gloss = gloss_list[idx]
            gloss_count += 1
            if gloss_count > MAX_GLOSS_COUNT:
                processed_count = idx
                break
        extract_save(video_name_list[idx], start_frame[idx], end_frame[idx], idx)
        print(f"Finished video {idx}")

    words = []
    for i in range(processed_count):
        h = f'numpy_openpose/{i}.npy'
        hh = np.load(h)
        words.append(hh)

    hands = []
    pose = []
    face = []

    for i in range(len(words)):
        hands.append([word[25:67] for word in words[i]])
        pose.append([word[0:25] for word in words[i]])
        face.append([word[67:137] for word in words[i]])

    for i in range(len(hands)):
        hands[i] = np.array(hands[i])

    for i in range(len(face)):
        face[i] = np.array(face[i])

    for i in range(len(pose)):
        pose[i] = np.array(pose[i])

        #25 body, 21 left hand, 21 right hand, 70 face; Total: 137
    dk = pd.DataFrame({'face': face, 'hands': hands, 'pose': pose, 'label': gloss_list[:processed_count]})

    dk.to_pickle("WLASL" + str(gloss_count) + "_openpose.pkl")
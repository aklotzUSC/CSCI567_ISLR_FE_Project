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
import mediapipe as mp

#CHANGE THIS TO DIRECTORY OF VIDEO FILES
vid_dir = 'videos/WLASL2000'

MAX_GLOSS_COUNT = 200

def detect_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
            mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
            mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        pose_results = pose.process(frame_rgb)
        hand_results = hands.process(frame_rgb)
        face_results = face_mesh.process(frame_rgb)

    return pose_results, hand_results, face_results


def extract_landmarks_from_video(video_path, start_frame=1, end_frame=-1):
    cap = cv2.VideoCapture(video_path)
    if end_frame < 0:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if start_frame < 1:
        start_frame = 1

    landmarks_data = []

    with mp.solutions.pose.Pose() as pose, mp.solutions.hands.Hands() as hands, mp.solutions.face_mesh.FaceMesh() as face_mesh:
        while cap.isOpened() and start_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            results = detect_landmarks(frame)
            landmarks_data.append({
                "frame_number": start_frame,
                "body_pose_landmarks": results[0].pose_landmarks,
                "hand_landmarks": results[1].multi_hand_landmarks,
                "face_mesh_landmarks": results[2].multi_face_landmarks
            })
            start_frame += 1

    cap.release()
    return landmarks_data
def extract_data(frame):
    pose_landmarks = []
    face_landmarks = []
    hands_landmarks = []
    if frame['body_pose_landmarks'] != None:
        for landmark in frame['body_pose_landmarks'].landmark:
            pose_landmarks.append([landmark.x, landmark.y, landmark.z])
    else:
        pose_landmarks = np.zeros([1, 33, 3])
    if frame['hand_landmarks'] != None:
        for landmark in frame['hand_landmarks'][0].landmark:
            hands_landmarks.append([landmark.x, landmark.y, landmark.z])
        if len(frame['hand_landmarks']) > 1:
            for landmark in frame['hand_landmarks'][1].landmark:
                hands_landmarks.append([landmark.x, landmark.y, landmark.z])
        else:
            hands_landmarks = np.row_stack((np.array(hands_landmarks), np.zeros([21, 3])))
        hands_landmarks = np.array([hands_landmarks])
    else:
        hands_landmarks = np.zeros([1,42,3])
    if frame['face_mesh_landmarks'] != None:
        for landmark in frame['face_mesh_landmarks'][0].landmark:
            face_landmarks.append([landmark.x, landmark.y, landmark.z])
    else:
        face_landmarks = np.zeros([1, 468, 3])
    if isinstance(pose_landmarks, np.ndarray):
        pose = pose_landmarks
    else:
        pose = np.array([pose_landmarks])

    if isinstance(face_landmarks, np.ndarray):
        face = face_landmarks
    else:
        face = np.array([face_landmarks])
    if isinstance(hands_landmarks, np.ndarray):
        hands = hands_landmarks
    else:
        hands = np.array([hands_landmarks])
    stacked_landmarks = np.hstack((pose, hands, face))
    return (stacked_landmarks)

def process_video(landmarks,frame_count):
    stacked_arrays = [extract_data(landmarks[i]) for i in range(frame_count)]
    temp = np.vstack(stacked_arrays)
    return temp

def extract_save(input_path, start, end, vid_idx):
    # Check if the file exists at the path
    npy_path = f'numpy_mediapipe/{vid_idx}.npy'
    # Extract landmarks from video
    temp = extract_landmarks_from_video(input_path, start, end)
    # Process video and save as NumPy file
    npyArray = process_video(temp, len(temp))
    np.save(npy_path, npyArray)

if __name__ == '__main__':
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
        h = f'Numpy_new/{i}.npy'
        hh = np.load(h)
        words.append(hh)

    hands = []
    pose = []
    face = []

    for i in range(len(words)):
        hands.append([word[33:75] for word in words[i]])
        pose.append([word[0:33] for word in words[i]])
        face.append([word[75:543] for word in words[i]])

    for i in range(len(hands)):
        hands[i] = np.array(hands[i])

    for i in range(len(face)):
        face[i] = np.array(face[i])

    for i in range(len(pose)):
        pose[i] = np.array(pose[i])

        #25 body, 21 left hand, 21 right hand, 70 face; Total: 137
    dk = pd.DataFrame({'face': face, 'hands': hands, 'pose': pose, 'label': gloss_list[:processed_count]})

    dk.to_pickle("WLASL" + str(gloss_count) + "_mediapipe.pkl")
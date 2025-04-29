import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Setup MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Extract pose features from a frame
def extract_pose_features(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        return np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
    else:
        return np.zeros(33 * 2)

# Load all videos from folder and extract pose features
def load_videos_from_folder(folder_path, label):
    video_sequences = []
    labels = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.mp4', '.avi', '.mov')):
            continue

        cap = cv2.VideoCapture(os.path.join(folder_path, filename))
        video_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            features = extract_pose_features(frame)
            video_frames.append(features)

        cap.release()

        if video_frames:
            video_array = np.array(video_frames)
            video_sequences.append(video_array)
            labels.append(label)

    return video_sequences, labels

# Build the LSTM model
def create_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=input_shape),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train the model
def train_model(video_data, labels):
    max_len = max(len(seq) for seq in video_data)
    padded_data = []

    for seq in video_data:
        padded_seq = np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0)), 'constant')
        padded_data.append(padded_seq)

    padded_data = np.array(padded_data, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    print(f"Training data shape: {padded_data.shape}, dtype: {padded_data.dtype}")
    print(f"Labels shape: {labels.shape}, dtype: {labels.dtype}")

    if len(padded_data) < 2:
        print("⚠️ Only one sample available. Skipping train/test split.")
        model = create_model(padded_data.shape[1:])
        model.fit(padded_data, labels, epochs=10, batch_size=1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, random_state=42)
        model = create_model(X_train.shape[1:])
        model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

    model.save('parkinsons_pose_model.h5')
    print("✅ Model saved as 'parkinsons_pose_model.h5'")

# Main runner
if __name__ == "__main__":
    healthy_videos, healthy_labels = load_videos_from_folder('train_healthy', 0)
    unhealthy_videos, unhealthy_labels = load_videos_from_folder('train_unhealthy', 1)

    all_videos = healthy_videos + unhealthy_videos
    all_labels = healthy_labels + unhealthy_labels

    if not all_videos:
        print("❌ No videos found in training folders.")
    else:
        train_model(all_videos, all_labels)

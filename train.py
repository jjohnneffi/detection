import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Set up MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Extract Pose Features from a Frame
def extract_pose_features(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        body_parts = []
        for landmark in landmarks:
            body_parts.append([landmark.x, landmark.y])
        return np.array(body_parts).flatten()
    return np.zeros(33 * 2)

# Create LSTM Model
def create_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=False, input_shape=input_shape),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train Model
def train_model(video_data, labels):
    max_len = max(len(seq) for seq in video_data)
    padded_data = []

    for seq in video_data:
        seq = seq.reshape(-1, 66)  # 33 landmarks * 2 coords
        padded_seq = np.pad(seq, ((0, max_len - seq.shape[0]), (0, 0)), 'constant')
        padded_data.append(padded_seq)

    padded_data = np.array(padded_data)

    if len(padded_data) < 2:
        print("⚠️ Only one sample available. Skipping train/test split.")
        model = create_model(padded_data.shape[1:])
        model.fit(padded_data, np.array(labels), epochs=10, batch_size=1)
    else:
        X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, test_size=0.2, random_state=42)
        model = create_model(X_train.shape[1:])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    model.save('parkinsons_pose_model.h5')
    print("✅ Model saved as 'parkinsons_pose_model.h5'")

# Main
if __name__ == "__main__":
    video_file = 'train.mp4'
    video_frames = []

    cap = cv2.VideoCapture(video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features = extract_pose_features(frame)
        video_frames.append(features)
    cap.release()

    # For one sample, wrap it into a list
    all_videos = [np.concatenate(video_frames)]
    all_labels = [1]  # Replace with your actual label list if using more videos

    train_model(all_videos, all_labels)

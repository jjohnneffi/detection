import cv2
import numpy as np
import librosa
import tensorflow as tf
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LSTM, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Set up MediaPipe for pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Preprocess Audio (Convert .mp4 video to Mel Spectrogram)
def preprocess_audio(video_path):
    y, sr = librosa.load(video_path, sr=None, mono=True, duration=30.0)  # Adjust the duration as needed
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram

# Extract Pose Features from Video Frame (Pose Estimation)
def extract_pose_features(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        body_parts = []
        for landmark in landmarks:
            body_parts.append([landmark.x, landmark.y])
        return np.array(body_parts).flatten()
    return np.zeros(33*3)  # If no pose is detected, return zero array

# Create CNN-LSTM Model for Video Analysis (Combining Audio and Pose Features)
def create_model(input_shape_audio, input_shape_video):
    audio_input = tf.keras.Input(shape=input_shape_audio)
    video_input = tf.keras.Input(shape=input_shape_video)
    
    # Audio model (CNN for Mel Spectrograms)
    x_audio = Conv2D(32, (3, 3), activation='relu')(audio_input)
    x_audio = Dropout(0.2)(x_audio)
    x_audio = Conv2D(64, (3, 3), activation='relu')(x_audio)
    x_audio = Dropout(0.2)(x_audio)
    x_audio = Flatten()(x_audio)
    
    # Video model (LSTM for Pose Features)
    x_video = LSTM(128, return_sequences=False)(video_input)
    x_video = Dense(64, activation='relu')(x_video)
    
    # Combine both branches
    combined = tf.keras.layers.concatenate([x_audio, x_video])
    output = Dense(1, activation='sigmoid')(combined)
    
    model = tf.keras.Model(inputs=[audio_input, video_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training the model with Audio and Video Data (Mel Spectrogram and Pose Data)
def train_model(audio_data, video_data, labels):
    audio_data = pad_sequences(audio_data, padding='post')
    video_data = np.array(video_data)
    
    X_train, X_test, y_train, y_test = train_test_split([audio_data, video_data], labels, test_size=0.2, random_state=42)
    
    input_shape_audio = (X_train[0].shape[1], X_train[0].shape[2], 1)  # Mel-spectrogram size
    input_shape_video = (X_train[1].shape[1],)  # Pose features size
    model = create_model(input_shape_audio, input_shape_video)
    model.fit([X_train[0], X_train[1]], y_train, epochs=10, batch_size=32, validation_data=([X_test[0], X_test[1]], y_test))
    
    # Save the trained model
    model.save('parkinsons_model.h5')
    print("Model trained and saved as 'parkinsons_model.h5'")

# Main script for training
if __name__ == "__main__":
    training_video_file = 'train.mp4'  # Your training video file
    
    # Step 1: Process the training video for both audio and video features
    audio_features = preprocess_audio(training_video_file)
    video_frames = []
    cap = cv2.VideoCapture(training_video_file)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features = extract_pose_features(frame)
        video_frames.append(features)
    cap.release()
    
    # Step 2: Labels (1 for PD-positive, 0 for PD-negative)
    labels = [1]  # You would replace this with your actual labels
    
    # Step 3: Train the model
    train_model([audio_features], video_frames, labels)

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load model
model = tf.keras.models.load_model('parkinsons_pose_model.h5')

# Set up MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Extract pose features from a frame
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

# Run prediction on a video
def predict(video_path):
    video_frames = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features = extract_pose_features(frame)
        video_frames.append(features)
    cap.release()

    if not video_frames:
        print("⚠️ No pose data extracted from video.")
        return

    video_array = np.array(video_frames).reshape(-1, 66)

    # Dynamically determine expected input length from model
    model_input_len = model.input_shape[1]

    if video_array.shape[0] < model_input_len:
        padded = np.pad(video_array, ((0, model_input_len - video_array.shape[0]), (0, 0)), 'constant')
    else:
        padded = video_array[:model_input_len]

    padded = np.expand_dims(padded, axis=0)  # Add batch dimension

    # Prediction
    prediction = model.predict(padded)[0][0]
    label = "Parkinson's Positive" if prediction >= 0.5 else "Parkinson's Negative"
    print(f"✅ Prediction: {label}")
    print(f"📊 Confidence Score: {prediction:.4f}")

    # Advanced Seaborn Plot
    data = {
        "Class": ["Negative", "Positive"],
        "Confidence": [1 - prediction, prediction]
    }
    df = pd.DataFrame(data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 3))
    palette = sns.color_palette("coolwarm", as_cmap=True)
    bar = sns.barplot(data=df, x="Confidence", y="Class", orient='h', palette="coolwarm")

    # Annotate bars with percentage
    for i, (conf, cls) in enumerate(zip(df["Confidence"], df["Class"])):
        bar.text(conf + 0.01, i, f"{conf:.2%}", color='black', va='center')

    plt.xlim(0, 1)
    plt.title("Parkinson's Prediction Confidence")
    plt.xlabel("Confidence Score")
    plt.ylabel("Prediction Class")
    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    test_video_path = 'test.mp4'  # Replace with your test video
    predict(test_video_path)

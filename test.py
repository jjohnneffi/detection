import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

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
        return np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
    return np.zeros(33 * 2)

# Process single video
def process_video(video_path, input_len):
    frames = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        features = extract_pose_features(frame)
        frames.append(features)

    cap.release()
    if not frames:
        return None

    video_array = np.array(frames).reshape(-1, 66)
    if video_array.shape[0] < input_len:
        padded = np.pad(video_array, ((0, input_len - video_array.shape[0]), (0, 0)), 'constant')
    else:
        padded = video_array[:input_len]

    return padded

# Predict and evaluate
def predict_and_evaluate(video_path, true_label=1):
    input_len = model.input_shape[1]
    features = process_video(video_path, input_len)

    if features is None:
        print("⚠️ No pose data extracted from video.")
        return

    prediction_prob = model.predict(np.expand_dims(features, axis=0))[0][0]
    predicted_label = int(prediction_prob >= 0.5)

    # Show classification result
    label_name = "Parkinson's Positive" if predicted_label == 1 else "Parkinson's Negative"
    print(f"✅ Predicted: {label_name}")
    print(f"📊 Confidence Score: {prediction_prob:.4f}")

    # Print metrics
    y_true = [true_label]
    y_pred = [predicted_label]
    y_prob = [prediction_prob]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)

    print("\n📋 Metrics:")
    print(f"Accuracy:  {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall:    {rec:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"AUC Score: {auc:.2f}")

    # Bar chart of confidence
    data = {"Class": ["Negative", "Positive"], "Confidence": [1 - prediction_prob, prediction_prob]}
    df = pd.DataFrame(data)

    plt.figure(figsize=(6, 3))
    sns.barplot(data=df, x="Confidence", y="Class", orient='h', palette="coolwarm")
    for i, (conf, cls) in enumerate(zip(df["Confidence"], df["Class"])):
        plt.text(conf + 0.01, i, f"{conf:.2%}", color='black', va='center')
    plt.xlim(0, 1)
    plt.title("Prediction Confidence")
    plt.xlabel("Score")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig("prediction_bar.png")
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("confusion_matrix_single.png")
    plt.show()

# Main
if __name__ == "__main__":
    test_video_path = "test.mp4"
    true_label = 1  # Replace with 0 if you know test.mp4 is a healthy subject
    predict_and_evaluate(test_video_path, true_label)

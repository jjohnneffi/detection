import cv2
import numpy as np
import librosa
import tensorflow as tf
import mediapipe as mp

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

# Load the trained model
model = tf.keras.models.load_model('parkinsons_model.h5')

# Real-time Detection on Test Video
def real_time_detection(model, video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame for pose estimation
        frame_resized = cv2.resize(frame, (224, 224))
        features = extract_pose_features(frame_resized)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        
        # Predict using the trained model
        audio_features = preprocess_audio(video_path)  # You can extract the same audio features from the test video
        audio_features = np.expand_dims(audio_features, axis=0)  # Add batch dimension
        audio_features = np.expand_dims(audio_features, axis=-1)  # Add channel dimension for CNN
        
        prediction = model.predict([audio_features, features])
        label = "Parkinson's Disease" if prediction > 0.5 else "Healthy"
        
        # Display the result on the video frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Real-Time Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

# Main script for testing
if __name__ == "__main__":
    test_video_file = 'test.mp4'  # Your test video file
    real_time_detection(model, test_video_file)

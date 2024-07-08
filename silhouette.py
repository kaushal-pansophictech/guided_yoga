
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import tensorflow as tf
import threading
import queue

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
model_mediapipe = None
labels = None

# Load models and labels
try:
    model_mediapipe = tf.keras.models.load_model("pose_model.h5", compile=False)
    labels = np.load("labels.npy")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define yoga poses dictionary
yoga_poses = {
    "ArdhaChandrasana": {
        "difficulty": "Intermediate",
        "description": "Half Moon Pose. A standing balance posture that strengthens the legs, ankles, and core while improving coordination and concentration.",
        "benefits": "Improves balance and coordination, strengthens legs and ankles, stretches the chest, shoulders, and spine.",
        "image": "ArdhaChandrasana.jpeg"
    },
    "BaddhaKonasana": {
        "difficulty": "Beginner",
        "description": "Bound Angle Pose. A seated forward bend that stretches the inner thighs, groin, and knees while providing a gentle twist to the spine.",
        "benefits": "Stimulates abdominal organs, ovaries and prostate gland, bladder, and kidneys. Stretches the inner thighs, groins, and knees.",
        "image": "BaddhaKonasana.jpg"
    },
    "Downward-Facing Dog": {
        "difficulty": "Beginner",
        "description": "An inversion that builds strength in the arms, shoulders, and core while stretching the hamstrings and calves.",
        "benefits": "Calms the brain and helps relieve stress and mild depression, energizes the body, stretches the shoulders, hamstrings, calves, arches, and hands.",
        "image": "Downward_dog.jpeg"
    },
    "Natarajasana": {
        "difficulty": "Advanced",
        "description": "Dancer Pose. A graceful and challenging standing posture that requires balance, flexibility, and concentration.",
        "benefits": "Stretches the shoulders, chest, thighs, groin, and abdomen. Strengthens the legs and ankles, improves balance.",
        "image": "Natarajasana.jpg"
    },
    "Triangle": {
        "difficulty": "Beginner",
        "description": "Triangle Pose. A standing pose that stretches the legs, hips, and spine while improving balance and focus.",
        "benefits": "Stretches and strengthens the thighs, knees, and ankles, relieves backache, especially through second trimester of pregnancy.",
        "image": "Triangle.jpg"
    },
    "UtkataKonasana": {
        "difficulty": "Intermediate",
        "description": "Goddess Pose. A powerful standing pose that strengthens the legs, ankles, and core while improving balance and stability.",
        "benefits": "Strengthens the legs, glutes, hips, and core muscles, improves balance, opens the hips.",
        "image": "UtkataKonasana.jpg"
    },
    "Veerabhadrasana": {
        "difficulty": "Intermediate",
        "description": "Warrior Pose. A series of standing postures that build strength, stamina, and focus while promoting a sense of empowerment and determination.",
        "benefits": "Strengthens and stretches the legs and ankles, increases stamina, relieves backaches.",
        "image": "Veerabhadrasana.jpg"
    },
    "Vrukshasana": {
        "difficulty": "Beginner",
        "description": "Tree Pose. A standing balance posture that strengthens the legs, ankles, and core while improving focus and concentration.",
        "benefits": "Strengthens thighs, calves, ankles, and spine, stretches the groins and inner thighs, chest, and shoulders.",
        "image": "Vrukshasana.png"
    }
}

# Function to detect pose from image
def detect_pose(image, pose, draw_landmarks=True):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks and draw_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image, results.pose_landmarks, results.segmentation_mask

# Function to extract keypoints from landmarks
def extract_keypoints(landmarks):
    keypoints = []
    for landmark in landmarks.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    keypoints = keypoints[:33]  # Adjust according to your model's requirements
    return np.array(keypoints)

# Function to identify pose from keypoints
def identify_pose(keypoints):
    keypoints = keypoints.reshape(1, 33, 4)  # Adjust according to your model's requirements
    predictions = model_mediapipe.predict(keypoints)
    pose_index = np.argmax(predictions)
    pose_name = labels[pose_index]
    return pose_name

# Function to provide pose feedback
def get_pose_feedback(landmarks, pose_name, feedback_queue):
    # Placeholder logic for feedback based on detected pose
    if pose_name == "ArdhaChandrasana":
        feedback = "Ensure your balance is centered and maintain a straight line from your raised leg to your extended arm."
    elif pose_name == "BaddhaKonasana":
        feedback = "Focus on relaxing your inner thighs and letting your knees gently open outward."
    else:
        feedback = "Adjust your stance to ensure proper alignment and balance."

    feedback_queue.put(feedback)

# Main function to run the application
def main():
    st.title("Yoga Pose Detection and Feedback")
    pose = mp_pose.Pose(min_detection_confidence=0.5, enable_segmentation=True)

    st.sidebar.subheader("Pose Selection")
    pose_name = st.sidebar.selectbox("Choose a Yoga Pose", list(yoga_poses.keys()))
    st.sidebar.write(f"**Difficulty**: {yoga_poses[pose_name]['difficulty']}")
    st.sidebar.write(f"**Description**: {yoga_poses[pose_name]['description']}")

    try:
        pose_image = Image.open(yoga_poses[pose_name]['image'])
        st.sidebar.image(pose_image, caption=pose_name, use_column_width=True)
    except Exception as e:
        st.error("Error loading pose image: " + str(e))

    show_keypoints = st.sidebar.checkbox("Show Keypoints", value=True)

    start_yoga = st.button("Start Yoga")

    if start_yoga:
        cap = cv2.VideoCapture(2)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frame rate to 30 FPS

        stframe = st.empty()
        feedback_display = st.empty()
        pose_display = st.empty()
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Error accessing webcam.")
                break

            # Process the frame with MediaPipe Pose
            frame, landmarks, segmentation_mask = detect_pose(frame, pose, draw_landmarks=show_keypoints)

            if landmarks:
                keypoints = extract_keypoints(landmarks)
                detected_pose = identify_pose(keypoints)

                # Get feedback for detected pose
                feedback_queue = queue.Queue()
                feedback_thread = threading.Thread(target=get_pose_feedback, args=(landmarks, detected_pose, feedback_queue))
                feedback_thread.start()
                feedback_thread.join()

                if not feedback_queue.empty():
                    feedback = feedback_queue.get()
                    feedback_display.write(f"Pose Feedback: {feedback}")

                # Display detected pose
                pose_display.write(f"Detected Pose: {detected_pose}")

            if segmentation_mask is not None:
                # Apply the segmentation mask to the frame
                condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
                light_blue = np.zeros(frame.shape, dtype=np.uint8)
                light_blue[:] = (255, 182, 193)  # Light blue color (BGR format)
                
                # Alpha blending
                alpha = 0.5  # Transparency factor
                frame = np.where(condition, cv2.addWeighted(light_blue, alpha, frame, 1 - alpha, 0), frame)

            # Display the frame with pose detection
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", caption="Pose Detection Feed")

        cap.release()

if __name__ == "__main__":
    main()

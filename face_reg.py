import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from PIL import Image
import tempfile
from tensorflow.keras.models import load_model
import os
import random
import openai  
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize variables
model_mediapipe = None
labels = None

# Try loading the model
while model_mediapipe is None:
    try:
        model_mediapipe = load_model("pose_model.h5", compile=False)
        labels = np.load("labels.npy")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        time.sleep(1)

# Initialize MediaPipe Pose, Face Mesh, and Hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

st.title("Yoga Anytime")

# Load an image for the welcome page
try:
    welcome_image = Image.open('yogawellcome.jpg')
    st.image(welcome_image, use_column_width=True)
except Exception as e:
    st.error("Error loading welcome image: " + str(e))

st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Welcome", "Pose Detection", "Yoga Identifier", "Capture and Share", "User Profile"], key="app_mode_select")

yoga_poses = {
    "ArdhaChandrasana": {
        "difficulty": "Intermediate",
        "description": "Half Moon Pose. A standing balance posture that strengthens the legs, ankles, and core while improving coordination and concentration.",
        "benefits": "Improves balance and coordination, strengthens legs and ankles, stretches the chest, shoulders, and spine.",
        "image": "ArdhaChandrasana.jpeg",
        "feedback": [
            "Make sure your hips are stacked and your gaze is forward.",
            "Engage your core to keep balance.",
            "Stretch your upper arm upwards and your lower arm towards the floor."
        ]
    },
    "BaddhaKonasana": {
        "difficulty": "Beginner",
        "description": "Bound Angle Pose. A seated forward bend that stretches the inner thighs, groin, and knees while providing a gentle twist to the spine.",
        "benefits": "Stimulates abdominal organs, ovaries and prostate gland, bladder, and kidneys. Stretches the inner thighs, groins, and knees.",
        "image": "BaddhaKonasana.jpg",
        "feedback": [
            "Sit tall with your spine straight.",
            "Gently press your knees towards the floor.",
            "Keep your shoulders relaxed."
        ]
    },
    "Downward-Facing Dog": {
        "difficulty": "Beginner",
        "description": "An inversion that builds strength in the arms, shoulders, and core while stretching the hamstrings and calves.",
        "benefits": "Calms the brain and helps relieve stress and mild depression, energizes the body, stretches the shoulders, hamstrings, calves, arches, and hands.",
        "image": "Downward_dog.jpeg",
        "feedback": [
            "Push your heels towards the floor.",
            "Lift your hips up and back.",
            "Keep your ears aligned with your upper arms and your fingers spread wide."
        ]
    },
    "Natarajasana": {
        "difficulty": "Advanced",
        "description": "Dancer Pose. A graceful and challenging standing posture that requires balance, flexibility, and concentration.",
        "benefits": "Stretches the shoulders, chest, thighs, groin, and abdomen. Strengthens the legs and ankles, improves balance.",
        "image": "Natarajasana.jpg",
        "feedback": [
            "Stand tall on one leg, and lift the other leg behind you.",
            "Hold your ankle with the opposite hand and reach forward with your free hand.",
            "Focus on a point to maintain balance."
        ]
    },
    "Triangle": {
        "difficulty": "Beginner",
        "description": "Triangle Pose. A standing pose that stretches the legs, hips, and spine while improving balance and focus.",
        "benefits": "Stretches and strengthens the thighs, knees, and ankles, relieves backache, especially through second trimester of pregnancy.",
        "image": "Triangle.jpg",
        "feedback": [
            "Keep your legs straight and engage your thighs.",
            "Extend your top arm towards the ceiling and look up at your hand.",
            "Open your chest and lengthen your spine."
        ]
    },
    "UtkataKonasana": {
        "difficulty": "Intermediate",
        "description": "Goddess Pose. A powerful standing pose that strengthens the legs, ankles, and core while improving balance and stability.",
        "benefits": "Strengthens the legs, glutes, hips, and core muscles, improves balance, opens the hips.",
        "image": "UtkataKonasana.jpg",
        "feedback": [
            "Bend your knees and keep them in line with your toes.",
            "Engage your core and keep your chest lifted.",
            "Reach your arms out to the sides and bend your elbows."
        ]
    },
    "Veerabhadrasana": {
        "difficulty": "Intermediate",
        "description": "Warrior Pose. A series of standing postures that build strength, stamina, and focus while promoting a sense of empowerment and determination.",
        "benefits": "Strengthens and stretches the legs and ankles, increases stamina, relieves backaches.",
        "image": "Veerabhadrasana.jpg",
        "feedback": [
            "Bend your front knee over your ankle and extend your arms parallel to the floor.",
            "Keep your back leg straight and strong.",
            "Look over your front hand and keep your shoulders relaxed."
        ]
    },
    "Vrukshasana": {
        "difficulty": "Beginner",
        "description": "Tree Pose. A standing balance posture that strengthens the legs, ankles, and core while improving focus and concentration.",
        "benefits": "Strengthens thighs, calves, ankles, and spine, stretches the groins and inner thighs, chest, and shoulders.",
        "image": "Vrukshasana.png",
        "feedback": [
            "Place your foot on your inner thigh or calf, avoiding the knee.",
            "Press your hands together in prayer position.",
            "Find a focal point to help with balance."
        ]
    }
}

openai.api_key = os.environ.get("OPENAI_API_KEY")

def detect_pose(image, pose, face_mesh, hands, draw_landmarks=True):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(image_rgb)
    face_results = face_mesh.process(image_rgb)
    hands_results = hands.process(image_rgb)
    
    if draw_landmarks:
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    return image, pose_results.pose_landmarks, face_results.multi_face_landmarks, hands_results.multi_hand_landmarks

def extract_keypoints(landmarks):
    keypoints = []
    for landmark in landmarks.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
    keypoints = keypoints[:33]
    return np.array(keypoints)

def start_timer(duration):
    start_time = time.time()
    end_time = start_time + duration
    return end_time

def update_timer(end_time):
    remaining_time = end_time - time.time()
    if remaining_time > 0:
        return remaining_time
    else:
        return 0

def identify_pose(keypoints):
    keypoints = keypoints.reshape(1, 33, 4)
    predictions = model_mediapipe.predict(keypoints)
    pose_index = np.argmax(predictions)
    pose_name = labels[pose_index]
    return pose_name

def get_pose_feedback(selected_pose_name):
    feedback = random.choice(yoga_poses[selected_pose_name]["feedback"])
    return feedback

def save_video(frames, filename, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def capture_and_share():
    st.header("Capture and Share Your Yoga Pose")
    
    if st.button("Start Capture"):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 15)  # Set the frame rate to 15 FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set the resolution to 320x240
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        frames = []
        start_time = time.time()
        recording_time = 30  # seconds

        stframe = st.empty()

        # Initialize pose, face mesh, and hands detection
        pose = mp_pose.Pose(min_detection_confidence=0.5)
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
        hands = mp_hands.Hands(min_detection_confidence=0.5)

        while time.time() - start_time < recording_time:
            ret, frame = cap.read()
            if not ret:
                st.error("Error accessing webcam.")
                break

            # Process the frame with MediaPipe solutions
            frame, pose_landmarks, face_landmarks, hand_landmarks = detect_pose(frame, pose, face_mesh, hands, draw_landmarks=True)
            frames.append(frame)

            # Display the frame with detections
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", caption="Recording Yoga Pose")

        cap.release()
        cv2.destroyAllWindows()

        # Save the video
        video_filename = "yoga_pose_video.avi"
        save_video(frames, video_filename)

        st.success("Yoga pose video captured successfully!")

        # Provide a link to download the video
        st.markdown(f"[Download Video](./{video_filename})", unsafe_allow_html=True)

        # Optionally, you can use a service to upload the video and get a shareable link.
        # Here, we're providing a simple download link.

        st.markdown("### Share your video on social media")
        st.markdown(f"[Share on Facebook](https://www.facebook.com/sharer/sharer.php?u=http://localhost:8501/{video_filename})")
        st.markdown(f"[Share on X](https://twitter.com/intent/tweet?url=http://localhost:8501/{video_filename})")
        st.markdown(f"[Share on WhatsApp](https://api.whatsapp.com/send?text=http://localhost:8501/{video_filename})")

def get_dynamic_feedback(selected_pose_name):
    """Use the LLM to get dynamic feedback for the selected yoga pose."""
    prompt = f"Provide one line of feedback to correct the yoga pose: {selected_pose_name}"
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a yoga pose assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    feedback = response['choices'][0]['message']['content'].strip()
    return feedback

# def get_dynamic_feedback(landmarks, selected_pose_name):
#     keypoints_data = str(landmarks)

#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": f"You are a yoga pose assistant. Help users correct their {selected_pose_name} pose based on the provided keypoints data. Provide one line of feedback to correct the {selected_pose_name}, do not mentioned that based on your provided data like stuff in the response"},
#             {"role": "user", "content": f"Here are the keypoints data from the user's selected pose: {keypoints_data}. Please provide one feedback on how to improve or adjust the {selected_pose_name} pose."}
#         ]
#     )
#     feedback = response['choices'][0]['message']['content'].strip()
#     return feedback

def main():
    pose_timer = 20  # seconds
    end_time = None

    if app_mode == "Welcome":
        st.write("Welcome to the Yoga Pose App! Select 'Pose Detection' or 'Yoga Identifier' from the sidebar to get started.")
    
    elif app_mode == "Pose Detection":
        st.header("Yoga Pose Detection with Timer")
        
        st.sidebar.subheader("Pose Selection")
        selected_pose_name = st.sidebar.selectbox("Choose a Yoga Pose", list(yoga_poses.keys()), key="pose_select")
        st.sidebar.write(f"**Difficulty**: {yoga_poses[selected_pose_name]['difficulty']}")
        st.sidebar.write(f"**Description**: {yoga_poses[selected_pose_name]['description']}")
        
        try:
            pose_image = Image.open(yoga_poses[selected_pose_name]['image'])
            st.image(pose_image, caption=selected_pose_name, use_column_width=True)
        except Exception as e:
            st.error("Error loading pose image: " + str(e))

        show_keypoints = st.sidebar.checkbox("Show Keypoints", value=True, key="show_keypoints")

        start_detection = st.button("Start Pose Detection", key="start_detection")

        if start_detection:
            pose = mp_pose.Pose(min_detection_confidence=0.5)
            face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
            hands = mp_hands.Hands(min_detection_confidence=0.5)

            cap = cv2.VideoCapture(2)
            cap.set(cv2.CAP_PROP_FPS, 15)  # Set the frame rate to 15 FPS
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set the resolution to 320x240
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

            stframe = st.empty()
            feedback_placeholder = st.empty()
            pose_placeholder = st.empty()
            timer_placeholder = st.empty()
            prev_time = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.sidebar.error("Error accessing webcam.")
                    break

                # Process the frame with MediaPipe solutions
                frame, pose_landmarks, face_landmarks, hand_landmarks = detect_pose(frame, pose, face_mesh, hands, draw_landmarks=show_keypoints)
                if pose_landmarks:
                    keypoints = extract_keypoints(pose_landmarks)
                    detected_pose = identify_pose(keypoints)

                    # Only provide feedback for the selected pose
                    feedback = get_dynamic_feedback(selected_pose_name)

                    # Update the placeholders with detected pose and feedback
                    pose_placeholder.write(f"**Detected Pose**: {detected_pose}")
                    feedback_placeholder.write(f"**Feedback**: {feedback}")

                    # Check if detected pose matches the selected pose
                    correct_pose = detected_pose == selected_pose_name
                    if correct_pose:
                        if end_time is None:
                            end_time = start_timer(pose_timer)
                        remaining_time = update_timer(end_time)
                        if remaining_time == 0:
                            timer_placeholder.success("Congratulations, you have held the pose for the required time!")
                            next_pose_name = random.choice([pose for pose in yoga_poses.keys() if pose != selected_pose_name])
                            if st.button("Go to Next Pose", key="next_pose"):
                                st.session_state["next_pose"] = next_pose_name
                                st.experimental_rerun()
                        timer_placeholder.write(f"**Hold the pose for**: {remaining_time:.2f} seconds.")
                    else:
                        timer_placeholder.error("Please adjust your posture to match the selected pose.")
                        end_time = None  # Reset timer if pose is incorrect

                    # Add delay for feedback
                    time.sleep(2)  # 2-second delay

                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(frame, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the frame with pose detection
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame, channels="RGB", caption="Pose Detection Feed")

            cap.release()
            cv2.destroyAllWindows()

    elif app_mode == "Yoga Identifier":
        st.header("Yoga Pose Identifier")
        st.sidebar.subheader("Upload an Image/Video/GIF")
        uploaded_file = st.sidebar.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "mp4", "gif"], key="file_upload")

        pose = mp_pose.Pose(min_detection_confidence=0.5)
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5)
        hands = mp_hands.Hands(min_detection_confidence=0.5)
        
        if uploaded_file is not None:
            if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                st.image(image, channels="BGR")
                image, pose_landmarks, face_landmarks, hand_landmarks = detect_pose(image, pose, face_mesh, hands, draw_landmarks=True)
                if pose_landmarks:
                    keypoints = extract_keypoints(pose_landmarks)
                    detected_pose = identify_pose(keypoints)
                    feedback = get_dynamic_feedback(detected_pose)
                    st.write(f"**Detected Pose**: {detected_pose}")
                    st.write(f"**Description**: {yoga_poses[detected_pose]['description']}")
                    st.write(f"**Benefits**: {yoga_poses[detected_pose]['benefits']}")
                    st.write(f"**Feedback**: {feedback}")
                    st.image(image, channels="BGR")

            elif uploaded_file.type == "video/mp4":
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())

                vidcap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                while vidcap.isOpened():
                    ret, frame = vidcap.read()
                    if not ret:
                        break
                    frame, pose_landmarks, face_landmarks, hand_landmarks = detect_pose(frame, pose, face_mesh, hands, draw_landmarks=True)
                    if pose_landmarks:
                        keypoints = extract_keypoints(pose_landmarks)
                        detected_pose = identify_pose(keypoints)
                        feedback = get_dynamic_feedback(detected_pose)
                        st.write(f"**Detected Pose**: {detected_pose}")
                        st.write(f"**Description**: {yoga_poses[detected_pose]['description']}")
                        st.write(f"**Benefits**: {yoga_poses[detected_pose]['benefits']}")
                        st.write(f"**Feedback**: {feedback}")
                        stframe.image(frame, channels="BGR")
                vidcap.release()

            elif uploaded_file.type == "image/gif":
                gif = Image.open(uploaded_file)
                st.image(gif)
                gif.seek(0)
                try:
                    while True:
                        frame = gif.convert('RGB')
                        frame = np.array(frame)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        frame, pose_landmarks, face_landmarks, hand_landmarks = detect_pose(frame, pose, face_mesh, hands, draw_landmarks=True)
                        if pose_landmarks:
                            keypoints = extract_keypoints(pose_landmarks)
                            detected_pose = identify_pose(keypoints)
                            feedback = get_dynamic_feedback(detected_pose)
                            st.write(f"**Detected Pose**: {detected_pose}")
                            st.write(f"**Description**: {yoga_poses[detected_pose]['description']}")
                            st.write(f"**Benefits**: {yoga_poses[detected_pose]['benefits']}")
                            st.write(f"**Feedback**: {feedback}")
                            st.image(frame, channels="BGR")
                        gif.seek(gif.tell() + 1)
                except EOFError:
                    pass

    elif app_mode == "Capture and Share":
        capture_and_share()

    elif app_mode == "User Profile":
        st.write("User Profile Management - Coming soon!")

if __name__ == "__main__":
    main()

import tkinter as tk
from tkinter import filedialog, ttk
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import tensorflow as tf
import time
import tempfile
import random
import openai
import threading

# Set up OpenAI API key
import os
from dotenv import load_dotenv
# Set up OpenAI API key
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load models and labels
model_mediapipe = tf.keras.models.load_model("pose_model.h5", compile=False)
labels = np.load("labels.npy")

# Define yoga poses dictionary
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
    feedback = "Adjust your stance to ensure proper alignment and balance."
    if pose_name in yoga_poses:
        feedback = random.choice(yoga_poses[pose_name]['feedback'])
    feedback_queue.put(feedback)

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

# Main Tkinter Application
class YogaApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Yoga Anytime App")
        self.geometry("800x600")
        
        self.pose = mp_pose.Pose(min_detection_confidence=0.5)
        
        self.create_widgets()
        self.cap = None
        self.end_time = None
        self.pose_timer = 20
        self.current_pose_name = None
        self.next_pose = None
        
    def create_widgets(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        
        self.welcome_frame = ttk.Frame(self.notebook)
        self.pose_detection_frame = ttk.Frame(self.notebook)
        self.yoga_identifier_frame = ttk.Frame(self.notebook)
        self.capture_share_frame = ttk.Frame(self.notebook)
        self.user_profile_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.welcome_frame, text="Welcome")
        self.notebook.add(self.pose_detection_frame, text="Pose Detection")
        self.notebook.add(self.yoga_identifier_frame, text="Yoga Identifier")
        self.notebook.add(self.capture_share_frame, text="Capture and Share")
        self.notebook.add(self.user_profile_frame, text="User Profile")
        
        self.create_welcome_page()
        self.create_pose_detection_page()
        self.create_yoga_identifier_page()
        self.create_capture_share_page()
        self.create_user_profile_page()
        
    def create_welcome_page(self):
        welcome_label = ttk.Label(self.welcome_frame, text="Welcome to Yoga Anytime App!", font=("Arial", 20))
        welcome_label.pack(pady=20)
        try:
            welcome_image = Image.open('yogawellcome.jpg')
            welcome_image = ImageTk.PhotoImage(welcome_image)
            image_label = ttk.Label(self.welcome_frame, image=welcome_image)
            image_label.image = welcome_image
            image_label.pack()
        except Exception as e:
            error_label = ttk.Label(self.welcome_frame, text=f"Error loading welcome image: {e}", font=("Arial", 12))
            error_label.pack()

    def create_pose_detection_page(self):
        pose_label = ttk.Label(self.pose_detection_frame, text="Yoga Pose Detection with Timer", font=("Arial", 20))
        pose_label.pack(pady=10)

        self.pose_selection = ttk.Combobox(self.pose_detection_frame, values=list(yoga_poses.keys()), state='readonly')
        self.pose_selection.current(0)
        self.pose_selection.pack(pady=10)
        
        self.show_keypoints = tk.BooleanVar(value=True)
        show_keypoints_check = ttk.Checkbutton(self.pose_detection_frame, text="Show Keypoints", variable=self.show_keypoints)
        show_keypoints_check.pack(pady=10)
        
        self.pose_feedback_label = ttk.Label(self.pose_detection_frame, text="", font=("Arial", 12))
        self.pose_feedback_label.pack(pady=10)
        
        self.timer_label = ttk.Label(self.pose_detection_frame, text="", font=("Arial", 16))
        self.timer_label.pack(pady=10)
        
        start_button = ttk.Button(self.pose_detection_frame, text="Start Pose Detection", command=self.start_pose_detection)
        start_button.pack(pady=10)
        
        self.pose_canvas = tk.Canvas(self.pose_detection_frame, width=640, height=480)
        self.pose_canvas.pack(pady=10)
        
    def create_yoga_identifier_page(self):
        identifier_label = ttk.Label(self.yoga_identifier_frame, text="Yoga Pose Identifier", font=("Arial", 20))
        identifier_label.pack(pady=10)
        
        self.upload_button = ttk.Button(self.yoga_identifier_frame, text="Upload Image/Video/GIF", command=self.upload_file)
        self.upload_button.pack(pady=10)
        
        self.identifier_canvas = tk.Canvas(self.yoga_identifier_frame, width=640, height=480)
        self.identifier_canvas.pack(pady=10)
        
        self.identifier_result_label = ttk.Label(self.yoga_identifier_frame, text="", font=("Arial", 12))
        self.identifier_result_label.pack(pady=10)
        
    def create_capture_share_page(self):
        capture_label = ttk.Label(self.capture_share_frame, text="Capture and Share Your Yoga Pose", font=("Arial", 20))
        capture_label.pack(pady=10)
        
        self.start_capture_button = ttk.Button(self.capture_share_frame, text="Start Capture", command=self.start_capture)
        self.start_capture_button.pack(pady=10)
        
        self.capture_canvas = tk.Canvas(self.capture_share_frame, width=640, height=480)
        self.capture_canvas.pack(pady=10)
        
        self.share_link_label = ttk.Label(self.capture_share_frame, text="", font=("Arial", 12))
        self.share_link_label.pack(pady=10)
        
    def create_user_profile_page(self):
        user_profile_label = ttk.Label(self.user_profile_frame, text="User Profile Management - Coming soon!", font=("Arial", 20))
        user_profile_label.pack(pady=20)

    def start_pose_detection(self):
        self.current_pose_name = self.pose_selection.get()
        self.pose_feedback_label.config(text=f"Selected Pose: {self.current_pose_name}")
        self.end_time = None
        self.timer_label.config(text="")
        
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  # Change camera index to 0
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set lower resolution
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        def pose_detection_loop():
            frame_counter = 0  # Counter to process every nth frame
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_counter += 1
                if frame_counter % 5 != 0:  # Process every 5th frame
                    continue
                
                frame, landmarks, _ = detect_pose(frame, self.pose, draw_landmarks=self.show_keypoints.get())
                
                if landmarks:
                    keypoints = extract_keypoints(landmarks)
                    detected_pose = identify_pose(keypoints)
                    feedback = get_dynamic_feedback(detected_pose)
                    
                    self.pose_feedback_label.config(text=f"Detected Pose: {detected_pose}\nFeedback: {feedback}")
                    
                    correct_pose = detected_pose == self.current_pose_name
                    if correct_pose:
                        if self.end_time is None:
                            self.end_time = time.time() + self.pose_timer
                        remaining_time = self.end_time - time.time()
                        if remaining_time <= 0:
                            self.timer_label.config(text="Congratulations, you have held the pose for the required time!")
                            self.end_time = None
                            self.next_pose = random.choice([pose for pose in yoga_poses.keys() if pose != self.current_pose_name])
                            next_pose_button = ttk.Button(self.pose_detection_frame, text=f"Next Pose: {self.next_pose}", command=self.start_next_pose)
                            next_pose_button.pack(pady=10)
                        else:
                            self.timer_label.config(text=f"Hold the pose for: {remaining_time:.2f} seconds.")
                    else:
                        self.timer_label.config(text="Please adjust your posture to match the selected pose.")
                        self.end_time = None
                
                # Display the frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_image = Image.fromarray(frame)
                frame_image = ImageTk.PhotoImage(frame_image)
                self.pose_canvas.create_image(0, 0, anchor=tk.NW, image=frame_image)
                self.pose_canvas.image = frame_image
                
                self.update_idletasks()
        
        threading.Thread(target=pose_detection_loop).start()
    
    def start_next_pose(self):
        self.pose_selection.set(self.next_pose)
        self.start_pose_detection()

    def upload_file(self):
        filetypes = (
            ("Image files", "*.jpg *.jpeg *.png"),
            ("Video files", "*.mp4"),
            ("GIF files", "*.gif")
        )
        file_path = filedialog.askopenfilename(title="Choose a file", filetypes=filetypes)
        if file_path:
            self.process_uploaded_file(file_path)

    def process_uploaded_file(self, file_path):
        self.identifier_result_label.config(text="")
        
        if file_path.lower().endswith((".jpg", ".jpeg", ".png")):
            image = cv2.imread(file_path)
            self.display_image(image, self.identifier_canvas)
            
            image, landmarks, _ = detect_pose(image, self.pose)
            if landmarks:
                keypoints = extract_keypoints(landmarks)
                detected_pose = identify_pose(keypoints)
                feedback = get_dynamic_feedback(detected_pose)
                
                self.identifier_result_label.config(text=f"Detected Pose: {detected_pose}\nFeedback: {feedback}")

        elif file_path.lower().endswith(".mp4"):
            cap = cv2.VideoCapture(file_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame, landmarks, _ = detect_pose(frame, self.pose)
                if landmarks:
                    keypoints = extract_keypoints(landmarks)
                    detected_pose = identify_pose(keypoints)
                    feedback = get_dynamic_feedback(detected_pose)
                    
                    self.identifier_result_label.config(text=f"Detected Pose: {detected_pose}\nFeedback: {feedback}")
                
                self.display_image(frame, self.identifier_canvas)
                
                self.update_idletasks()
            cap.release()

        elif file_path.lower().endswith(".gif"):
            gif = Image.open(file_path)
            self.display_gif(gif, self.identifier_canvas)
            
    def display_image(self, image, canvas):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=image)
        canvas.image = image
        
    def display_gif(self, gif, canvas):
        def gif_loop():
            while True:
                try:
                    frame = gif.copy()
                    frame = frame.convert("RGB")
                    frame = np.array(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame, landmarks, _ = detect_pose(frame, self.pose)
                    self.display_image(frame, canvas)
                    gif.seek(gif.tell() + 1)
                except EOFError:
                    gif.seek(0)
                    continue
        
        threading.Thread(target=gif_loop).start()

    def start_capture(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)  # Change camera index to 0
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set lower resolution
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        
        def capture_loop():
            frames = []
            start_time = time.time()
            recording_time = 30  # seconds
            
            while time.time() - start_time < recording_time:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame, landmarks, _ = detect_pose(frame, self.pose)
                frames.append(frame)
                
                self.display_image(frame, self.capture_canvas)
                
                self.update_idletasks()
            
            # Save the video
            video_filename = "yoga_pose_video.avi"
            self.save_video(frames, video_filename)
            self.share_link_label.config(text=f"Video saved: {video_filename}")
        
        threading.Thread(target=capture_loop).start()

    def save_video(self, frames, filename, fps=30):
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()

if __name__ == "__main__":
    app = YogaApp()
    app.mainloop()

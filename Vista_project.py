import streamlit as st
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
from gtts import gTTS
import os
import glob
import tempfile
import zipfile
import cv2
import uuid
from collections import defaultdict

# Load models
st.write("[INFO] Loading models...")
yolo_model = YOLO("yolov8n.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Utility Functions
def analyze_scene(image):
    np_img = np.array(image.convert("RGB"))
    results = yolo_model(np_img)[0]
    detections = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        label = yolo_model.names[int(cls)]
        detections.append(label)
    return detections

def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def speak(text, filename="reply.mp3"):
    tts = gTTS(text)
    temp_path = os.path.join(tempfile.gettempdir(), filename)
    tts.save(temp_path)
    st.audio(temp_path, format="audio/mp3")

def summarize_objects(object_history):
    summary = defaultdict(int)
    for obj in object_history:
        summary[obj] += 1
    return summary

# Streamlit UI
st.title("🧠 BlindSight : AI-Powered Vision Assistant for the Visually Impaired")
mode = st.radio("Select Analysis Mode:", ["Single Image", "Folder of Images (.zip)", "Webcam", "Video File"], horizontal=True)

# Single Image Mode
if mode == "Single Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        caption = generate_caption(image)
        objects = analyze_scene(image)
        st.write(f"📝 **Caption:** {caption}")
        st.write("📦 **Detected Objects:**")
        for label in objects:
            st.write(f"- {label}")
        question = st.text_input("❓ Ask a question (e.g., describe the scene):", key="single_image_input")
        if question:
            st.write(f"🤖 **Answer:** {caption}")
            speak(caption)

# Folder of Images Mode
elif mode == "Folder of Images (.zip)":
    uploaded_zip = st.file_uploader("Upload a ZIP file containing images", type=["zip"])
    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.read())
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(tmp_dir)
            image_files = [f for f in glob.glob(os.path.join(tmp_dir, "*")) if f.lower().endswith(("jpg", "jpeg", "png", "bmp"))]
            for img_path in image_files:
                img = Image.open(img_path).convert("RGB")
                st.image(img, caption=f"Processed Image: {os.path.basename(img_path)}", use_container_width=True)
                caption = generate_caption(img)
                objects = analyze_scene(img)
                st.write(f"📝 **Caption:** {caption}")
                st.write("📦 **Detected Objects:**")
                for label in objects:
                    st.write(f"- {label}")
                question = st.text_input(f"❓ Ask a question for {os.path.basename(img_path)}:", key=f"zip_input_{os.path.basename(img_path)}")
                if question:
                    st.write(f"🤖 **Answer:** {caption}")
                    speak(caption)

# Webcam Mode
elif mode == "Webcam":
    st.write("🔎 Starting Webcam...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            caption = generate_caption(pil_image)
            objects = analyze_scene(pil_image)
            st.image(frame, channels="BGR", use_container_width=True)
            st.write(f"📝 **Caption:** {caption}")
            st.write("📦 **Detected Objects:**")
            for label in objects:
                st.write(f"- {label}")
            speak(caption)
        cap.release()
    else:
        st.error("Unable to access the webcam.")

# Video Mode
elif mode == "Video File":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4"])
    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
            tmp_vid.write(uploaded_video.read())
            cap = cv2.VideoCapture(tmp_vid.name)
            object_history = []
            object_crops = {}
            caption_frame = None
            frame_id = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_id += 1
                if frame_id == 10:
                    caption_frame = frame.copy()
                results = yolo_model(frame)[0]
                names = results.names
                detections = results.boxes.cls.cpu().numpy().astype(int)
                boxes = results.boxes.xyxy.cpu().numpy()
                labels = [names[d] for d in detections]
                object_history.extend(labels)
                for (label, box) in zip(labels, boxes):
                    if label not in object_crops:
                        x1, y1, x2, y2 = map(int, box)
                        object_crop = frame[y1:y2, x1:x2]
                        object_crops[label] = object_crop
            cap.release()
            if caption_frame is not None:
                pil_image = Image.fromarray(cv2.cvtColor(caption_frame, cv2.COLOR_BGR2RGB))
                caption = generate_caption(pil_image)
                st.write(f"📝 **Caption:** {caption}")
                speak("Scene description: " + caption)
            obj_summary = summarize_objects(object_history)
            st.write("📦 **Detected Object Summary:**")
            for obj, count in obj_summary.items():
                st.write(f"{obj} - {count}")
            speak("Detected objects are: " + ", ".join([f"{obj} {count}" for obj, count in obj_summary.items()]))
            for label, image in object_crops.items():
                st.image(image, caption=f"{label}", channels="BGR", use_container_width=True)
            st.success("✅ Video analysis complete.")
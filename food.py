from ultralytics import YOLO
import streamlit as st

# Load model nhận diện món ăn (Detect)
@st.cache_resource
def load_food_model():
    # Đảm bảo file yolov8n.pt nằm trong folder model
    return YOLO("./model/yolov8n.pt")

# Hàm dự đoán món ăn
def predict_food(model, image, conf):
    # Trả về kết quả detect
    return model.predict(image, conf=conf)
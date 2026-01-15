from ultralytics import YOLO
import streamlit as st
import os

# Bảng hệ số calo theo size (Update theo file size.pt của bạn)
SIZE_MULTIPLIERS = {
    "small": 0.7,   "nho": 0.7,   "be": 0.7,
    "medium": 1.0,  "vua": 1.0,   "tb": 1.0,
    "large": 1.5,   "to": 1.5,    "lon": 1.5,   "big": 1.5
}

# Load model phân loại size (Classify)
@st.cache_resource
def load_size_model():
    path = "./model/size.pt"
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            print(f"⚠️ Lỗi load size model: {e}")
            return None
    return None

# Hàm dự đoán size từ ảnh đã cắt (crop)
def predict_size(model, crop_image):
    if model is None:
        return 1.0, "Mặc định (No Model)"
    
    # Dự đoán (verbose=False để tắt log thừa)
    results = model.predict(crop_image, verbose=False)
    
    # Lấy kết quả Top 1
    if results and results[0].probs:
        top1_index = results[0].probs.top1
        label = results[0].names[top1_index].lower()
        
        # Tra bảng hệ số
        multiplier = SIZE_MULTIPLIERS.get(label, 1.0)
        display_label = f"{label.upper()} (x{multiplier})"
        
        return multiplier, display_label
    
    return 1.0, "Không xác định"
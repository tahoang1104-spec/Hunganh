import av
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import numpy as np
import os
from class_names import class_names

# --- CẤU HÌNH HỆ SỐ SIZE ---
# Sửa tên class ở đây cho khớp với model size của bạn
SIZE_MULTIPLIERS = {
    "small": 0.7,
    "nho": 0.7,
    "medium": 1.0,
    "vua": 1.0,
    "large": 1.5,
    "to": 1.5,
    "big": 1.5
}

# --- 1. LOAD MODEL & CSS ---
@st.cache_resource
def load_models():
    # Load model Food
    model_food = YOLO("./model/yolov8n.pt")
    
    # Load model Size (có xử lý lỗi nếu file hỏng/thiếu)
    model_size = None
    if os.path.exists("./model/size.pt"):
        try:
            model_size = YOLO("./model/size.pt")
        except Exception as e:
            print(f"Lỗi load model size: {e}")
    
    return model_food, model_size

def styling_css():
    if os.path.exists('./assets/css/general-style.css'):
        with open('./assets/css/general-style.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- HÀM HỖ TRỢ TÍNH TOÁN ---
def get_box_center(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (x1 + x2) / 2, (y1 + y2) / 2

def is_center_inside(center, box_wrapper):
    cx, cy = center
    x1, y1, x2, y2 = box_wrapper.xyxy[0].tolist()
    return x1 < cx < x2 and y1 < cy < y2

# --- 2. HÀM HIỂN THỊ KẾT QUẢ (Đã sửa lỗi NoneType) ---
def display_results(food_results, size_results, container_placeholder):
    container = container_placeholder.container()
    
    with container:
        st.divider()
        st.subheader("🥗 Kết quả phân tích chi tiết")
        
        total_calories = 0
        found_any = False
        
        # Duyệt qua từng món ăn
        for r in food_results:
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                if class_id >= len(class_names): continue
                
                info = class_names[class_id]
                name = info["name"]
                base_nutri = info["nutrition"]
                
                # --- LOGIC TÌM SIZE (ĐÃ VÁ LỖI) ---
                multiplier = 1.0
                size_label = "Vừa (Mặc định)"
                
                if size_results:
                    food_center = get_box_center(box)
                    for s_r in size_results:
                        # >>> DÒNG SỬA LỖI QUAN TRỌNG <<<
                        # Nếu model size không trả về boxes (None), thì bỏ qua
                        if s_r.boxes is None: 
                            continue 
                            
                        for s_box in s_r.boxes:
                            if is_center_inside(food_center, s_box):
                                s_name = size_results[0].names[int(s_box.cls[0].item())].lower()
                                if s_name in SIZE_MULTIPLIERS:
                                    multiplier = SIZE_MULTIPLIERS[s_name]
                                    size_label = f"{s_name.upper()} (x{multiplier})"
                                else:
                                    size_label = f"{s_name}"
                # -----------------------------------

                cal = int(base_nutri.get('Calories', 0) * multiplier)
                fat = round(base_nutri.get('Fat', 0) * multiplier, 1)
                sugar = round(base_nutri.get('Sugar', 0) * multiplier, 1)
                
                found_any = True
                total_calories += cal
                
                with st.expander(f"🔹 {name} - Size: {size_label}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("🔥 Calo", f"{cal}")
                    c2.metric("🥩 Chất béo", f"{fat}g")
                    c3.metric("🍬 Đường", f"{sugar}g")

        if found_any:
            st.success(f"📊 **TỔNG KẾT:** Tổng cộng khoảng **{total_calories} kcal**.")
        else:
            st.warning("⚠️ Không tìm thấy món ăn.")

# --- 3. CHỨC NĂNG: ẢNH ---
def detect_image(conf, uploaded_file, models):
    model_food, model_size = models
    
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
    
    if st.button("🔍 Phân tích ngay"):
        with st.spinner("Đang chạy 2 Model AI..."):
            # 1. Chạy Model Food
            res_food = model_food.predict(image, conf=conf)
            
            # 2. Chạy Model Size (Nếu có)
            res_size = None
            plot_img = res_food[0].plot()
            
            if model_size:
                # Giảm độ tin cậy size xuống thấp chút để dễ bắt
                res_size = model_size.predict(image, conf=0.15) 
                
                # Vẽ khung size (nếu có) đè lên ảnh để debug
                if res_size and res_size[0].boxes is not None:
                     plot_img = res_size[0].plot(img=plot_img)

            res_image = Image.fromarray(plot_img[..., ::-1])
            
            with col2:
                st.image(res_image, caption="Kết quả nhận diện", use_container_width=True)
            
            # Gọi hàm hiển thị
            display_results(res_food, res_size, st.empty())

# --- CÁC HÀM KHÁC (VIDEO, WEBCAM) GIỮ NGUYÊN ---
def detect_video(conf, uploaded_file, models):
    st.warning("Chức năng Size chưa hỗ trợ Video.")

def detect_webcam(conf, models):
    st.warning("Chức năng Size chưa hỗ trợ Webcam.")

def detect_camera(conf, models, url): pass
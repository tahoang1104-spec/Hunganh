import streamlit as st
from PIL import Image
import numpy as np
import os
import pandas as pd
from class_names import class_names

# Import module vệ tinh
from food import predict_food
from size import predict_size

# --- 1. HÀM CSS (ĐÃ SỬA LỖI CHARMAP) ---
def styling_css():
    css_path = './assets/css/general-style.css'
    if os.path.exists(css_path):
        # --- SỬA Ở ĐÂY: Thêm encoding='utf-8' ---
        with open(css_path, encoding='utf-8') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# --- 2. HÀM HIỂN THỊ KẾT QUẢ & BẢNG ---
def display_analysis(food_results, size_model, original_image, container):
    with container:
        st.divider()
        st.subheader("🥗 Kết quả chi tiết")
        
        total_calories = 0
        found_any = False
        table_data = [] 

        # --- BƯỚC 1: QUÉT TẤT CẢ CÁC BOX ĐỂ TÌM BÚN CHẢ ---
        all_detected_boxes = []
        has_bun_cha = False
        
        for r in food_results:
            for box in r.boxes:
                class_id = int(box.cls[0].item())
                if class_id < len(class_names):
                    name_raw = class_names[class_id]["name"].lower()
                    all_detected_boxes.append((box, class_id))
                    # Tìm từ khóa "bun cha" hoặc "bún chả" trong tên class
                    if "bun cha" in name_raw or "bún chả" in name_raw:
                        has_bun_cha = True

        # --- BƯỚC 2: LỌC VÀ HIỂN THỊ ---
        for box, class_id in all_detected_boxes:
            info = class_names[class_id]
            name = info["name"]
            name_lower = name.lower()

            # NẾU ĐÃ CÓ BÚN CHẢ, THÌ BỎ QUA CÁC BOX CHỈ LÀ "BÚN" (TRÁNH TRÙNG LẶP)
            # Điều kiện: Nếu có Bun cha VÀ tên món này chứa chữ "bun" nhưng KHÔNG chứa chữ "cha"
            if has_bun_cha:
                is_only_bun = ("bun" in name_lower or "bún" in name_lower) and \
                             ("cha" not in name_lower and "chả" not in name_lower)
                if is_only_bun:
                    continue 

            base_nutri = info["nutrition"]
            
            # Cắt ảnh & Tính size
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop_img = original_image.crop((int(x1), int(y1), int(x2), int(y2)))
            multiplier, size_label = predict_size(size_model, crop_img)

            # Tính dinh dưỡng
            cal = int(base_nutri.get('Calories', 0) * multiplier)
            fat = round(base_nutri.get('Fat', 0) * multiplier, 1)
            sugar = round(base_nutri.get('Sugar', 0) * multiplier, 1)
            
            total_calories += cal
            found_any = True
            
            table_data.append({
                "Tên món": name,
                "Kích cỡ": size_label,
                "Calo (kcal)": cal,
                "Chất béo (g)": fat,
                "Đường (g)": sugar
            })
            
            with st.expander(f"🔹 {name} - {size_label}", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("🔥 Calo", f"{cal}")
                c2.metric("🥩 Béo", f"{fat}g")
                c3.metric("🍬 Đường", f"{sugar}g")

        if found_any:
            st.markdown("### 📋 Bảng Tổng Hợp Dinh Dưỡng")
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
            st.success(f"📊 **TỔNG CỘNG BỮA ĂN:** ~ **{total_calories} kcal**")
        else:
            st.warning("⚠️ Không tìm thấy món ăn nào.")



# --- 3. HÀM XỬ LÝ CHÍNH (CÓ SESSION STATE) ---
def process_image(conf, uploaded_file, model_food, model_size):
    image = Image.open(uploaded_file)
    
    # Reset nếu upload ảnh mới
    if 'last_uploaded' not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        st.session_state.has_processed = False
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.food_results = None
        st.session_state.current_image = None

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Ảnh gốc", use_container_width=True)
        
    # Nút bấm chạy AI
    if st.button("🔍 Phân tích ngay"):
        with st.spinner("Đang chạy AI (Detect + Classify)..."):
            res_food = predict_food(model_food, image, conf)
            
            # Lưu kết quả vào bộ nhớ
            st.session_state.food_results = res_food
            st.session_state.current_image = image
            st.session_state.has_processed = True
            
    # Hiển thị (Luôn chạy nếu đã có kết quả trong bộ nhớ)
    if st.session_state.get('has_processed'):
        res_food = st.session_state.food_results
        org_image = st.session_state.current_image
        
        plot_img = res_food[0].plot()
        res_image = Image.fromarray(plot_img[..., ::-1])
        
        with col2:
            st.image(res_image, caption="AI Nhận diện", use_container_width=True)
        
        display_analysis(res_food, model_size, org_image, st.container())

# Placeholder
def process_video(): st.info("🚧 Chức năng Video đang cập nhật...")
def process_webcam(): st.info("🚧 Chức năng Webcam đang cập nhật...")
def process_camera(): st.write("Đang kết nối camera...")

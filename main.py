import streamlit as st
# Lưu ý: Import hàm load_models (số nhiều) từ utils mới
from utils import detect_image, detect_video, detect_webcam, detect_camera, load_models, styling_css

# 1. Cấu hình trang
st.set_page_config(
    page_title="FoodDetector Pro",
    page_icon="🍲",
    layout="wide"
)

# 2. Load CSS & Models (Food + Size)
try:
    styling_css()
    # Hàm này bây giờ trả về 2 model (model_food, model_size)
    models = load_models()
except Exception as e:
    st.error(f"⚠️ Lỗi khởi động hệ thống: {e}")
    st.stop()

# 3. Tạo Menu điều hướng bên trái
with st.sidebar:
    st.title("🍲 FoodDetector")
    selected_page = st.radio("Đi tới:", ["Trang chủ", "Giới thiệu", "Mã nguồn"])
    st.markdown("---")
    
    st.header("⚙️ Cài đặt")
    confidence = st.slider("Độ tin cậy (Confidence)", 10, 100, 40) / 100
    
    st.info("💡 Mẹo: Model Size hoạt động tốt nhất với ảnh chụp rõ nét.")

# 4. Giao diện trang TRANG CHỦ
if selected_page == "Trang chủ":
    # --- HIỂN THỊ BANNER (Đã sửa lỗi deprecated) ---
    try:
        st.image("welcome.png", use_container_width=True) 
    except:
        # Nếu không tìm thấy ảnh thì hiện chữ
        st.warning("⚠️ Chưa có file welcome.png trong thư mục.")
        st.title("🕵️ Nhận diện & Tính Calo Món Ăn")
    # -----------------------------------------------

    st.markdown("### Chọn phương thức đầu vào:")

    # 4 Tab chức năng
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Ảnh (Food + Size)", "🎥 Video", "📷 Webcam", "📹 IP Camera"])

    with tab1: # Tab Ảnh - Hỗ trợ tính Size
        st.subheader("📸 Tải ảnh món ăn")
        uploaded_file = st.file_uploader("Chọn ảnh (jpg, png)...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file:
            # Truyền cả bộ 'models' vào hàm xử lý
            detect_image(confidence, uploaded_file, models)

    with tab2: # Tab Video
        st.subheader("🎥 Tải video món ăn")
        uploaded_video = st.file_uploader("Chọn video (mp4, avi)...", type=['mp4', 'avi'])
        if uploaded_video:
            detect_video(confidence, uploaded_video, models)

    with tab3: # Tab Webcam
        st.subheader("📷 Camera trực tiếp")
        st.info("Bấm START để bật camera")
        detect_webcam(confidence, models)

    with tab4: # Tab IP Camera
        st.subheader("📹 Kết nối Camera IP")
        rtsp_url = st.text_input("Nhập địa chỉ RTSP:", placeholder="rtsp://admin:pass@192.168.1.x:554/...")
        if st.button("Kết nối Camera"):
            if rtsp_url:
                detect_camera(confidence, models, rtsp_url)
            else:
                st.warning("Vui lòng nhập địa chỉ RTSP")

# 5. Giao diện trang GIỚI THIỆU
elif selected_page == "Giới thiệu":
    st.header("ℹ️ Về dự án")
    st.markdown("""
    **FoodDetector Pro** là phiên bản nâng cấp với khả năng nhận diện kép:
    
    1.  **Nhận diện món ăn:** Sử dụng YOLOv8n (67 món Việt Nam).
    2.  **Nhận diện kích cỡ:** Sử dụng Model phụ trợ để xác định (Nhỏ, Vừa, Lớn).
    
    **Cách hoạt động:**
    - Nếu phát hiện size **Lớn** (Large), lượng Calo sẽ nhân hệ số **x1.5**.
    - Nếu phát hiện size **Nhỏ** (Small), lượng Calo sẽ nhân hệ số **x0.7**.
    """)

elif selected_page == "Mã nguồn":
    st.header("📂 Mã nguồn")
    st.write("Dự án được xây dựng trên nền tảng Streamlit và YOLOv8.")
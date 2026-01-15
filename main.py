import streamlit as st
from streamlit_option_menu import option_menu # <-- Thư viện menu đẹp
from food import load_food_model
from size import load_size_model
from utils import styling_css, process_image, process_video, process_webcam, process_camera

# 1. Cấu hình trang
st.set_page_config(
    page_title="FoodDetector",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load CSS và Model
try:
    styling_css()
    model_food = load_food_model()
    model_size = load_size_model()
except Exception as e:
    st.error(f"Lỗi khởi động: {e}")
    st.stop()

# 3. Giao diện Sidebar (GIỐNG HỆT BẢN GỐC)
with st.sidebar:
    # Logo hoặc Tiêu đề to
    st.markdown("<h1 style='text-align: center; color: #FEC51C;'>🍲 FoodDetector</h1>", unsafe_allow_html=True)
    
    # Menu chọn trang với Icon đẹp
    selected = option_menu(
        menu_title=None,  # Không cần tiêu đề phụ
        options=["Home", "About", "Github"], 
        icons=["house", "info-circle", "github"], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#FEC51C", "color": "black"}, # Màu vàng khi chọn
        }
    )
    
    st.markdown("---")
    st.subheader("⚙️ Settings")
    confidence = st.slider("Độ tin cậy (Confidence)", 10, 100, 40) / 100

# 4. Điều hướng các trang
if selected == "Home":
    # Phần Banner hoặc Tiêu đề trang chủ
    try:
        st.image("welcome.png", use_container_width=True)
    except:
        st.title("Phân tích Dinh Dưỡng & Size 📏")

    # 4 Tab chức năng
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image", "🎥 Video", "📷 Webcam", "📹 IP Camera"])

    with tab1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            process_image(confidence, uploaded_file, model_food, model_size)

    with tab2:
        st.subheader("Upload Video")
        uploaded_video = st.file_uploader("Choose a video...", type=['mp4', 'avi'])
        if uploaded_video:
            process_video()

    with tab3:
        st.subheader("Webcam Live")
        process_webcam()
        
    with tab4:
        st.subheader("RTSP Camera")
        rtsp_url = st.text_input("RTSP URL:")
        if st.button("Connect"):
            process_camera()

elif selected == "About":
    st.title("ℹ️ About FoodDetector")
    st.info("""
    **FoodDetector** là ứng dụng AI hỗ trợ nhận diện món ăn Việt Nam và tính toán lượng calo.
    
    - **Models:** YOLOv8 (Detection + Classification)
    - **Data:** 67 Vietnamese Foods
    - **Features:** Calorie estimation based on Food Type & Size.
    """)

elif selected == "Github":
    st.title("📂 Source Code")
    st.markdown("""
    ### 🔗 GitHub Repository
    Truy cập mã nguồn gốc tại: [github.com/nvhnam/fooddetector](https://github.com/nvhnam/fooddetector)
    """)
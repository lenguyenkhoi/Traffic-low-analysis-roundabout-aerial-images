import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Dự đoán", 
    page_icon="🔍", 
    layout="wide"
)

st.title("🔍Phát hiện vật thể trong vòng xuyến")
st.markdown("---")

st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True) #ẩn các pagelink
st.sidebar.header("Menu")
st.sidebar.page_link("app.py", label="🏠 Trang Chủ")
st.sidebar.page_link("pages/eda.py", label="📊 Khám phá dữ liệu (EDA)")
st.sidebar.page_link("pages/model.py", label="🤖 Mô hình")
st.sidebar.page_link("pages/model_evaluation.py", label="⚙️ Quá trình huấn luyện và tối ưu hóa")
st.sidebar.page_link("pages/real_perform.py",label= "📈 Đánh giá hiệu năng thực tế")
st.sidebar.page_link("pages/ea.py",label= "📋Tổng kết và Phân tích lỗi")
st.sidebar.page_link("pages/demo_ea.py", label="🧪Demo Error Analysis")
st.sidebar.page_link("pages/predict.py", label="🔍 Dự đoán")

st.sidebar.header("Cài đặt Mô hình")

#  Upload ảnh
st.sidebar.markdown("---")
uploaded_file = st.file_uploader("Vui lòng tải ảnh lên", type=['jpg', 'jpeg', 'png'])

# Thanh trượt Confidence độ tin cậy
confidence = st.slider("Ngưỡng tin cậy (Confidence Score)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
st.caption("Tăng ngưỡng này nếu mô hình nhận diện nhầm quá nhiều. Giảm nếu mô hình bỏ sót xe.")
# đường dẫn tới mô hình 
MODEL_DICT = {
    "YOLOv11 Nano (n)": r"YOLOv11_training/training_backup_YOLOv11n/weights/best.pt",
    "YOLOv11 Small (s)": r"YOLOv11_training/training_backup_YOLOv11s/weights/best.pt",
    "YOLOv11 Medium (m)": r"YOLOv11_training/training_backup_YOLOv11m/weights/best.pt",
    "YOLOv11 Small (Optuna)": r"YOLOv11_training/model_tuning/weights/best.pt"
}

# selected_model_name = st.selectbox("Lựa chọn phiên bản YOLOv11:", list(MODEL_DICT.keys()))


@st.cache_resource
def load_model(model_path):
    """Hàm load mô hình YOLO, sử dụng cache để không phải load lại mỗi khi đổi ảnh"""
    return YOLO(model_path)


if uploaded_file is not None:
    # Đọc ảnh gốc bằng PIL
    image = Image.open(uploaded_file)
    img_array = np.array(image) # Chuyển sang mảng Numpy cho YOLO

    # Chia cột hiển thị (Trái: Gốc, Phải: Kết quả)

    st.subheader("📷 Ảnh Đầu Vào")
    st.image(image, use_container_width=True)

    # Nút bấm để thực hiện dự đoán
    st.subheader("🚀 Phân tích & Dự đoán của các mô hình")
    if "res1" not in st.session_state:
        st.session_state.res1 = None
    if "res2" not in st.session_state:
        st.session_state.res2 = None
    if "res3" not in st.session_state:
        st.session_state.res3 = None
        
    col1,col2,col3 = st.columns(3)    
    with col1:
        try:
            selected_model_1 = st.selectbox("Lựa chọn phiên bản YOLOv11:", list(MODEL_DICT.keys()), key = "Model 1")

            if st.button("Predict model 1", key="btn1"):
                model = load_model(MODEL_DICT[selected_model_1])

                with st.spinner(f"Đang chạy {selected_model_1}..."):
                    # YOLOv11 nhận Numpy array làm đầu vào
                    results = model.predict(source=img_array, conf=confidence)
                    
                    # Trích xuất ảnh đã vẽ BBox
                    res_img = results[0].plot()
                    # Do YOLO trả về hệ màu BGR, cần chuyển sang RGB để Streamlit hiện đúng màu
                    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    
                    # Đếm số lượng phương tiện 
                    num_vehicles = len(results[0].boxes)
                    boxes = results[0].boxes

                    if boxes is not None and len(boxes) > 0:
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                        names = results[0].names  # map id -> tên class

                        counts = {}
                        for cls_id in class_ids:
                            class_name = names[cls_id]
                            counts[class_name] = counts.get(class_name, 0) + 1

                        total = len(class_ids)
                    else:
                        counts = {}
                        total = 0
                    
                    st.session_state.res1 = (res_img_rgb, num_vehicles,counts)
                    
            if st.session_state.res1 is not None:
                img, total, counts = st.session_state.res1
                st.subheader(f"Kết Quả Dự Đoán ")
                st.image(img, use_container_width=True)
                st.success(f"**Tổng số phương tiện phát hiện được:** {total} chiếc.")
                for cls_name, cnt in counts.items():
                    st.success(f"**{cls_name}:** {cnt} chiếc")

                
        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra khi tải mô hình: {e}")
            st.warning("Vui lòng kiểm tra lại xem đường dẫn file .pt trong thư mục 'models' đã chính xác chưa.")


    with col2:
        try:
            selected_model_2 = st.selectbox("Lựa chọn phiên bản YOLOv11:", list(MODEL_DICT.keys()), key = "Model 2")
            if st.button("Predict model 2", key="btn2"):
                model = load_model(MODEL_DICT[selected_model_2])
                with st.spinner(f"Đang chạy {selected_model_2}..."):
                    
                    # YOLOv11 nhận Numpy array làm đầu vào
                    results = model.predict(source=img_array, conf=confidence)
                    
                    # Trích xuất ảnh đã vẽ BBox
                    res_img = results[0].plot()
                    # Do YOLO trả về hệ màu BGR, cần chuyển sang RGB để Streamlit hiện đúng màu
                    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    
                    # Đếm số lượng phương tiện 
                    num_vehicles = len(results[0].boxes)
                    boxes = results[0].boxes

                    if boxes is not None and len(boxes) > 0:
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                        names = results[0].names  # map id -> tên class

                        counts = {}
                        for cls_id in class_ids:
                            class_name = names[cls_id]
                            counts[class_name] = counts.get(class_name, 0) + 1

                        total = len(class_ids)
                    else:
                        counts = {}
                        total = 0
                    # Lưu session_state
                    st.session_state.res2 = (res_img_rgb, num_vehicles,counts)
                    
            if st.session_state.res2 is not None:
                img, total, counts = st.session_state.res2
                st.subheader(f"Kết Quả Dự Đoán ")
                st.image(img, use_container_width=True)
                st.success(f"**Tổng số phương tiện phát hiện được:** {total} chiếc")
                for cls_name, cnt in counts.items():
                    st.success(f"**{cls_name}:** {cnt} chiếc")

        except Exception as e:
            st.error(f"❌ Có lỗi xảy ra khi tải mô hình: {e}")
            st.warning("Vui lòng kiểm tra lại xem đường dẫn file .pt trong thư mục 'models' đã chính xác chưa.")


    with col3:
        try:
            selected_model_3 = st.selectbox("Lựa chọn phiên bản YOLOv11:", list(MODEL_DICT.keys()), key = "Model 3")
            if st.button("Predict model 3", key="btn3"):
                model = load_model(MODEL_DICT[selected_model_3])
                with st.spinner(f"Đang chạy {selected_model_3}..."):
                   # YOLOv11 nhận Numpy array làm đầu vào
                    results = model.predict(source=img_array, conf=confidence)
                    
                    # Trích xuất ảnh đã vẽ BBox
                    res_img = results[0].plot()
                    # Do YOLO trả về hệ màu BGR, cần chuyển sang RGB để Streamlit hiện đúng màu
                    res_img_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    
                    # Đếm số lượng phương tiện 
                    num_vehicles = len(results[0].boxes)
                    
                    boxes = results[0].boxes

                    if boxes is not None and len(boxes) > 0:
                        class_ids = boxes.cls.cpu().numpy().astype(int)
                        names = results[0].names  # map id -> tên class

                        counts = {}
                        for cls_id in class_ids:
                            class_name = names[cls_id]
                            counts[class_name] = counts.get(class_name, 0) + 1

                        total = len(class_ids)
                    else:
                        counts = {}
                        total = 0
                    # Lưu session_state
                    st.session_state.res3 = (res_img_rgb, num_vehicles,counts)
    
            if st.session_state.res3 is not None:
                img, total, counts = st.session_state.res3
                st.subheader(f"Kết Quả Dự Đoán ")
                st.image(img, use_container_width=True)
                st.success(f"**Tổng số phương tiện phát hiện được:** {total} chiếc")
                for cls_name, cnt in counts.items():
                    st.success(f"**{cls_name}:** {cnt} chiếc")

        except Exception as e:
                st.error(f"❌ Có lỗi xảy ra khi tải mô hình: {e}")
                st.warning("Vui lòng kiểm tra lại xem đường dẫn file .pt trong thư mục 'models' đã chính xác chưa.")
else:
    # Màn hình chờ khi chưa có ảnh
    st.info("Vui lòng tải một bức ảnh để mô hình dự đoán")
    
    

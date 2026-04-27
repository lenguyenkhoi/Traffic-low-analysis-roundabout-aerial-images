import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="PHÂN TÍCH LƯU LƯỢNG GIAO THÔNG TẠI VÒNG XUYẾN QUA ẢNH HÀNG KHÔNG", 
    page_icon="🚗", 
    layout="wide"
)

st.title("PHÂN TÍCH LƯU LƯỢNG GIAO THÔNG TẠI VÒNG XUYẾN QUA ẢNH HÀNG KHÔNG")
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


st.header("1. Giới thiệu bối cảnh")
st.write("""
Trong bối cảnh đô thị hóa ngày càng gia tăng, việc giám sát và phân tích mật độ giao thông đóng vai trò quan trọng trong quản lý hạ tầng và điều phối giao thông thông minh. Dự án này tập trung vào việc xây dựng một hệ thống phát hiện và phân loại phương tiện giao thông từ ảnh chụp trên cao (top-down view) sử dụng mô hình học sâu.

Mục tiêu chính của dự án là:
- Phát hiện các phương tiện trong ảnh (object detection)
- Phân loại phương tiện theo các nhóm như: car, cycle, bus, truck, van
- Đánh giá hiệu năng mô hình thông qua các chỉ số như Precision, Recall, mAP50 và mAP50-95
- Phân tích chuyên sâu các lỗi (Error Analysis) nhằm hiểu rõ điểm mạnh và hạn chế của mô hình

Trong dự án, các phiên bản khác nhau của mô hình YOLOv11 (n, s, m) được huấn luyện và so sánh nhằm tìm ra cấu hình tối ưu. Đặc biệt, phương pháp tối ưu siêu tham số bằng Optuna được áp dụng để cải thiện hiệu năng mô hình.

Bên cạnh đó, dự án cũng thực hiện:
- Tiền xử lý dữ liệu và phân chia tập train/validation/test
- Áp dụng các kỹ thuật tăng cường dữ liệu (data augmentation)
- Trực quan hóa kết quả dự đoán và phân tích lỗi (False Positive, False Negative, Localization Error, Duplicate Detection)

Kết quả của dự án không chỉ dừng lại ở việc đạt được các chỉ số đánh giá cao, mà còn cung cấp cái nhìn toàn diện về hành vi của mô hình trong các tình huống thực tế, đặc biệt là trong các cảnh có mật độ giao thông cao.

Dự án hướng tới việc xây dựng nền tảng cho các ứng dụng thực tế như:
- Giám sát giao thông tự động
- Phân tích mật độ phương tiện
- Hỗ trợ hệ thống giao thông thông minh (ITS)
""")
st.markdown("----")
st.header("2. Bài toán (Problem Statement)")
st.write("""
Bài toán đặt ra là xây dựng một mô hình có khả năng:
- Nhận diện chính xác các phương tiện giao thông trong ảnh chụp từ trên cao
- Phân biệt các loại phương tiện có hình dạng tương tự (đặc biệt là car và cycle)
- Hoạt động hiệu quả trong các điều kiện phức tạp như:
    + Mật độ giao thông cao
    + Các phương tiện chồng lấn hoặc che khuất nhau
    + Ảnh có nhiễu như bóng đổ, vạch kẻ đường

Đây là một bài toán Object Detection với độ khó cao do:
- Góc nhìn top-down làm mất đặc trưng chiều cao
- Kích thước object nhỏ và không đồng đều
- Dễ xảy ra lỗi duplicate, miss detection và localization error
""")
st.markdown("----")
st.header("3. Mô tả dữ liệu (Dataset Description)")
st.write("""
Dataset sử dụng trong dự án bao gồm các ảnh giao thông tại khu vực vòng xuyến, được gán nhãn theo định dạng YOLO.

Thông tin dataset:
- Ảnh đầu vào: ảnh chụp từ trên cao (aerial view)
- Định dạng nhãn: YOLO format (class_id, x_center, y_center, width, height)
- Các lớp (classes):
    + car
    + cycle
    + bus
    + truck
    + van

Quy trình xử lý dữ liệu:
- Lọc các ảnh có annotation hợp lệ (loại bỏ file rỗng hoặc lỗi)
- Chia tập dữ liệu thành train/validation/test
- Áp dụng các kỹ thuật tăng cường dữ liệu (augmentation) trong quá trình huấn luyện:
    + Rotation (xoay ảnh)
    + Scaling (phóng to/thu nhỏ)
    + Mosaic augmentation

Dataset được thiết kế nhằm phản ánh các tình huống giao thông thực tế với mật độ phương tiện khác nhau.
""")
st.subheader("Minh họa dataset")
col1, col2,col3 = st.columns(3)
with col1:
    st.subheader("Ảnh") 
    st.image("demo/00001_frame000005_original.jpg", caption="Minh họa ảnh")
with col2: 
    st.subheader("File tọa độ tương ứng")
    with open("demo/00001_frame000005_original.txt", "r", encoding="utf-8") as file:
        content = file.read()
        st.text_area("Nội dung file:", content, height=300)
with col3:
    st.subheader("File nhãn")
    with open("demo/classes.txt", "r", encoding="utf-8") as file:
        content = file.read()
        st.text_area("Nội dung file:", content, height=300)
    
st.markdown("----")
st.header("4. Phương pháp (Methodology)")
st.write("""
Dự án sử dụng các phiên bản khác nhau của mô hình YOLOv11:
- YOLOv11n (nhanh, nhẹ)
- YOLOv11s (cân bằng)
- YOLOv11m (chính xác cao hơn)

Ngoài ra, mô hình YOLOv11s được tối ưu bằng Optuna để tìm bộ siêu tham số tốt nhất.

Các bước thực hiện:
1. Huấn luyện mô hình với các tham số chung
2. So sánh hiệu năng giữa các model
3. Tối ưu hyperparameters bằng Optuna
4. Đánh giá bằng các metrics: Precision, Recall, mAP
5. Phân tích Confusion Matrix
6. Phân tích Precision-Recall Curve
6. Phân tích IoU
7. Phân tích lỗi chi tiết (Error Analysis)
""")
st.markdown("----")
st.header("5. Kết quả đạt được (Results Overview)")
st.write("""
Mô hình sau khi tối ưu đạt được hiệu năng cao:
- Precision cao -> giảm thiểu dự đoán sai
- Recall cao -> hạn chế bỏ sót đối tượng
- mAP50 và mAP50-95 tốt -> đảm bảo độ chính xác tổng thể

Đặc biệt:
- Phân loại gần như hoàn hảo giữa các lớp
- Hoạt động tốt trong môi trường mật độ giao thông cao
""")
st.markdown("---")
st.header("6. Kết luận (Conclusion)")
st.write("""
Dự án đã xây dựng thành công một hệ thống phát hiện và phân tích phương tiện giao thông từ ảnh trên cao với độ chính xác cao.

Các điểm nổi bật:
- Mô hình có khả năng phát hiện gần như toàn bộ phương tiện (Recall cao)
- Phân loại chính xác giữa các lớp (Classification tốt)
- Hoạt động ổn định trong điều kiện thực tế

Tuy nhiên vẫn tồn tại một số hạn chế:
- Một số lỗi duplicate detection (bắt trùng)
- Sai lệch bounding box (localization error)
- Nhạy với nhiễu như bóng đổ hoặc vật thể tương tự

Hướng phát triển:
- Tối ưu thêm NMS để giảm duplicate
- Tăng cường dữ liệu cho object nhỏ (cycle)
- Cải thiện localization bằng tuning loss function

Dự án là nền tảng quan trọng cho các ứng dụng:
- Giám sát giao thông thông minh (ITS)
- Phân tích mật độ phương tiện
- Hỗ trợ quy hoạch đô thị
""")


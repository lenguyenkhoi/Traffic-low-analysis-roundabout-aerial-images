import streamlit as st
import os
import cv2
import numpy as np
st.set_page_config("EDA", layout= "wide")


st.title("📊 Khám phá dữ liệu (EDA)")
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

# st.header("Dataset")
# st.subheader("Tổng quan dataset roundabount")
# st.write("""
# Dataset gồm ảnh 976 file ảnh được chụp từ trên cao, chứa nhiều phương tiện với phân bố không đồng đều. 
# Mỗi file ảnh tương ứng với file nhãn của ảnh và có một file class.txt. Ví Dụ:
# - File ảnh có ID 00001_frame000005_original.jpg sẽ có file nhãn tương ứng 00001_frame000005_original.txt
# - Lưu ý: File nhãn bao gồm ID của vật thể và tọa độ tương ứng. 
# """)
# col1, col2 = st.columns(2)
# with col1:
#     st.subheader("Ảnh") 
#     st.image("data/yolov9data/00001_frame000005_original.jpg ", caption="Minh họa ảnh")
# with col2: 
#     st.subheader("File nhãn tương ứng")
#     with open("data/yolov9data/00001_frame000005_original.txt", "r", encoding="utf-8") as file:
#         content = file.read()
#     st.text_area("Nội dung file:", content, height=300)
# st.markdown("-----")

st.header("EDA")
st.write("""Mục đích của EDA (Exploratory Data Analysis) là phân tích và hiểu rõ đặc điểm của dữ liệu, 
         bao gồm phân bố các lớp, kích thước và vị trí đối tượng, từ đó phát hiện các vấn đề như mất cân bằng dữ liệu, 
         nhiễu hoặc sai lệch, giúp đưa ra các quyết định phù hợp trong quá trình huấn luyện mô hình.""")
st.image("result/images/EDA_dataset.png", caption= "EDA Dataset")
st.subheader("Nhận xét: ")
col1,col2,col3 = st.columns(3)

with col1: 
    st.write("""
    **1. Phân bố lớp (Class Distribution):**  
    Dữ liệu bị mất cân bằng nghiêm trọng khi lớp *car* chiếm số lượng áp đảo so với các lớp còn lại. 
    Trong khi đó, các lớp như *bus*, *truck* và *van* có số lượng rất ít. 
    Điều này có thể khiến mô hình thiên về dự đoán lớp phổ biến và giảm khả năng nhận diện chính xác các lớp hiếm.
    """)
with col2:
    st.write(""" 
    **2. Phân bố không gian (Spatial Heatmap):**  
    Các phương tiện không phân bố đều trên ảnh mà tập trung chủ yếu tại một số khu vực nhất định (vùng mật độ cao). 
    Điều này cho thấy tồn tại hiện tượng *spatial bias*, nghĩa là mô hình có thể học được xu hướng vị trí thay vì đặc trưng thực sự của đối tượng. 
    Khi gặp các trường hợp phương tiện xuất hiện ở vị trí khác, mô hình có thể hoạt động kém hiệu quả hơn.
    """)
with col3:
     st.write(""" 
    **3. Phân bố kích thước bounding box (BBox Area Distribution):**  
    Phần lớn bounding box có diện tích rất nhỏ, cho thấy đa số phương tiện trong ảnh có kích thước nhỏ. 
    Đây là một thách thức lớn đối với mô hình Object Detection vì các đối tượng nhỏ thường khó phát hiện hơn và dễ bị bỏ sót.
    """)
     
st.subheader("Kết luận: ")
st.write("""
    Dataset tồn tại đồng thời nhiều vấn đề như mất cân bằng lớp, lệch phân bố không gian và đối tượng kích thước nhỏ. 
    Do đó, cần áp dụng các kỹ thuật như data augmentation, điều chỉnh tham số mô hình hoặc lựa chọn kiến trúc phù hợp để cải thiện hiệu năng nhận diện.
    
    **Chiến lược Tăng cường dữ liệu (Augmentation Strategy)**: 
    + Lật ngang (fliplr) và dọc (flipud): chuyển các điểm nóng từ bên trái sang bên phải, 
    và từ dưới lên trên, giúp cân bằng lại không gian phân bổ trên toàn bộ bức ảnh.
    + Xoay ngẫu nhiên (degrees): Xử lý tính đa hướng của phương tiện tại vòng xuyến, 
    vì xe cộ liên tục bẻ lái theo đường cong.
    + Thu phóng (scale) và ghép ảnh (mosaic): để mô hình học được các đặc trưng của phương tiện ở nhiều kích thước khác nhau (đặc biệt là xe đạp rất nhỏ trong ảnh).
    """)



st.markdown("------")
st.header("Bounding box")
st.subheader("Giới thiệu về bounding box")

st.write("""
         Trong các bài toán Object Detection bằng mô hình YOLO, Bounding Box là công cụ cốt lõi dùng để khoanh vùng và định vị tọa độ không gian của các mục tiêu bên trong dữ liệu hình ảnh.
         
         Về mặt cấu trúc, mỗi bounding box được dự đoán thông qua các tham số về tọa độ tâm $(x, y)$, chiều rộng $w$, chiều cao $h$, cùng với điểm tin cậy (confidence score) phản ánh xác suất thực sự chứa đối tượng. 
         
         Trong quá trình huấn luyện, việc tối ưu hóa độ chính xác của các hộp giới hạn này so với nhãn thực tế thường được đánh giá trực tiếp qua hàm mất mát dựa trên chỉ số IoU (Intersection over Union) đóng vai trò then chốt quyết định hiệu năng và độ nhạy bén của toàn bộ hệ thống nhận diện.
         """)
with st.expander("Ví dụ: "):
    st.image("result/images/boundingbox.png",caption="Bounding Box Roundabount")
    

st.markdown("-----")
st.header("Demo bounding box")
st.write("Dưới đây là phần demo bounding box dựa trên file ảnh và file nhãn tương ứng mà bạn tải lên")
col1,col2,col3= st.columns(3)
with col1: 
    uploaded_image = st.file_uploader("Vui lòng tải ảnh lên", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        st.image(uploaded_image)
with col2:
    uploaded_labels = st.file_uploader("Vui lòng tải file nhãn tương ứng", type=['txt'])
    if uploaded_labels is not None:
        content = uploaded_labels.read().decode("utf-8")
        st.text(content)
with col3:
    uploaded_classes = st.file_uploader("Vui lòng tải file clasess", type=['txt'])
    if uploaded_classes is not None:
        content = uploaded_classes.read().decode("utf-8")
        st.text(content)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

def draw_yolo_bbox_streamlit(uploaded_image, uploaded_labels, uploaded_classes=None):
    if uploaded_image is None or uploaded_labels is None:
        return

    # reset pointer
    uploaded_image.seek(0)
    uploaded_labels.seek(0)
    uploaded_classes.seek(0)

    # đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]

    # đọc class names
    class_names = {}
    if uploaded_classes:
        class_lines = uploaded_classes.read().decode("utf-8").splitlines()
        class_names = {i: name.strip() for i, name in enumerate(class_lines)}

    # đọc labels
    label_lines = uploaded_labels.read().decode("utf-8").splitlines()

    for line in label_lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        cls_id, x, y, bw, bh = map(float, parts)
        cls_id = int(cls_id)
        color = COLORS[cls_id % len(COLORS)]
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        label = class_names.get(cls_id, str(cls_id))

        # bbox màu đỏ cho dễ nhìn
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        cv2.putText( img, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),2)
    # st.success("Đã tạo thành công bounding Box")
    st.image(img, use_container_width=True, caption="Bounding Box")
    
if uploaded_image and uploaded_labels and uploaded_classes is not None:
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Ảnh gốc")
        st.image(uploaded_image, caption = "Ảnh gốc")
    with col2:
        st.subheader("Bounding Box")
        draw_yolo_bbox_streamlit(uploaded_image,uploaded_labels,uploaded_classes)

# st.markdown("------")

import streamlit as st
import plotly.graph_objects as go
import os
import cv2
import numpy as np
from helpers import compute_iou_simple
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Tổng kết và Phân tích Hạn chế (Error Analysis)", 
    page_icon="📋", 
    layout="wide"
)


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

st.title("Tổng kết và Phân tích Hạn chế (Error Analysis)")
# EA
st.header("Error Analysis (Phân tích lỗi)")
st.image("result/images/demo_ea.png", caption="Minh họa các lỗi")
st.write("""
Các lỗi trong mô hình Object Detection bao gồm:

- False Positive (FP): Mô hình phát hiện các đối tượng không tồn tại, thường do nhầm lẫn với background.
- False Negative (FN): Mô hình bỏ sót các đối tượng có thật, đặc biệt là các đối tượng nhỏ hoặc bị che khuất.
- Localization Error: Mô hình phát hiện đúng đối tượng nhưng vị trí bounding box không chính xác.
- Classification Error: Mô hình gán sai nhãn cho đối tượng, thường xảy ra giữa các lớp có đặc điểm tương đồng.
- Duplicate Detection: Mô hình tạo nhiều bounding box cho cùng một đối tượng do xử lý NMS chưa hiệu quả.
""")

st.subheader("Phân tích lỗi theo mô hình tối ưu")

st.write("""
Để đánh giá sâu hơn hiệu năng của mô hình YOLOv11, không chỉ dừng lại ở các chỉ số như Precision, Recall hay mAP, tiến hành phân tích chi tiết các loại lỗi phổ biến trong bài toán Object Detection.

Các lỗi được phân loại như sau:

- **False Positive (FP)**: Mô hình phát hiện các đối tượng không tồn tại trong thực tế. Nguyên nhân thường do sự nhầm lẫn giữa background và object (ví dụ: bóng đổ hoặc vật thể có hình dạng tương tự xe).

- **False Negative (FN)**: Mô hình bỏ sót các đối tượng có thật trong ảnh. Lỗi này thường xảy ra với các đối tượng nhỏ, bị che khuất hoặc nằm ở rìa ảnh.

- **Localization Error**: Mô hình phát hiện đúng đối tượng nhưng vị trí bounding box không chính xác (IoU thấp). Điều này ảnh hưởng trực tiếp đến chỉ số mAP, đặc biệt là mAP50-95.

- **Classification Error**: Mô hình phát hiện đúng vị trí nhưng gán sai nhãn lớp, thường xảy ra giữa các lớp có đặc điểm tương đồng như car, van và truck.

- **Duplicate Detection**: Mô hình tạo nhiều bounding box cho cùng một đối tượng, do cơ chế Non-Max Suppression (NMS) chưa tối ưu.

Thông qua việc trực quan hóa các loại lỗi này trên từng ảnh (với màu sắc phân biệt), hiểu rõ hơn điểm mạnh và hạn chế của mô hình, từ đó đề xuất các hướng cải thiện như:
- Tăng dữ liệu cho các lớp yếu
- Điều chỉnh confidence threshold
- Tối ưu hyperparameters
- Cải thiện chất lượng annotation

Việc phân tích lỗi giúp đảm bảo rằng mô hình không chỉ đạt kết quả tốt về mặt số liệu, mà còn hoạt động hiệu quả trong các tình huống thực tế.
""")
st.subheader("Dưới đây là trực quan hóa các lỗi của mô hình YOLOv11s-optuna")
st.write("""
### 🎨 Quy ước màu sắc

- 🟢 True Positive (TP): Phát hiện đúng
- 🔴 False Positive (FP): Phát hiện sai
- 🔵 False Negative (FN): Bỏ sót
- 🟡 Localization Error: Sai vị trí bbox
- 🟣 Classification Error: Sai nhãn
- 🟠 Duplicate Detection: Trùng lặp bbox
""")

st.subheader("Ảnh 1") 
st.image(r"result_vis/compare_00001_frame000442_original.jpg", caption="Minh họa ảnh 1")
st.subheader("Ảnh 2") 
st.image(r"result_vis/compare_00001_frame000497_original.jpg", caption="Minh họa ảnh 2")
st.subheader("Ảnh 3") 
st.image(r"result_vis/compare_00001_frame000558_original.jpg", caption="Minh họa ảnh 3")
st.subheader("Ảnh 4") 
st.image(r"result_vis/compare_00001_frame000832_original.jpg", caption="Minh họa ảnh 4")
st.subheader("Ảnh 5") 
st.image(r"result_vis/compare_00001_frame001016_original.jpg", caption="Minh họa ảnh 5 ")
st.subheader("File lỗi tương ứng") 
col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    st.subheader("Ảnh 1")
    file_path = os.path.join("result_vis", "compare_00001_frame000442_original.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        st.text(content)
    else:
        st.error("❌ Không tìm thấy file")
with col2:
    st.subheader("Ảnh 2")
    file_path = os.path.join("result_vis", "compare_00001_frame000497_original.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        st.text(content)
    else:
        st.error("❌ Không tìm thấy file")
with col3:
    st.subheader("Ảnh 3")
    file_path = os.path.join("result_vis", "compare_00001_frame000558_original.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        st.text(content)
    else:
        st.error("❌ Không tìm thấy file")
        
with col4:
    st.subheader("Ảnh 4")
    file_path = os.path.join("result_vis", "compare_00001_frame000832_original.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        st.text(content)
    else:
        st.error("❌ Không tìm thấy file")
        
with col5:
    st.subheader("Ảnh 5")
    file_path = os.path.join("result_vis", "compare_00001_frame001016_original.txt")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
        st.text(content)
    else:
        st.error("❌ Không tìm thấy file")
st.subheader("Nhận xét và Phân tích kết quả (Diagnostic Analysis)")
st.write("""
  Bảng thống kê lỗi (Error Analysis) từ 5 bức ảnh ngẫu nhiên trong tập Test cung cấp một góc nhìn chi tiết vào cách mô hình `YOLOv11s-optuna` xử lý các tình huống thực tế. 
  Dữ liệu cho thấy mô hình hoạt động cực kỳ ổn định, nhưng vẫn tồn tại một số sai số mang tính đặc thù của mạng nơ-ron:

**Nhận diện đúng phần lớn dữ liệu (Ảnh 2, Ảnh 3, Ảnh 4):** Mô hình đạt điểm tuyệt đối. Tổng số đối tượng dự đoán (Prediction) khớp 100% với thực tế (Ground Truth). 
Số lượng True Positive (TP) đạt tối đa, không có bất kỳ lỗi nào xảy ra.

**Lỗi nhận diện khống - False Positive (Ảnh 1):** Thực tế (Ground Truth) chỉ có 20 xe (18 `car`, 2 `cycle`), nhưng mô hình dự đoán có tới 21 xe (19 `car`, 2 `cycle`). 
Sinh ra 1 lỗi False Positive (FP = 1). Điều này chứng minh lại phân tích từ Confusion Matrix trước đó: 
Mô hình thỉnh thoảng bị dự đoán nhầm bởi lớp đa số, dẫn đến việc nhìn một bóng râm, vệt sáng hoặc chướng ngại vật trên mặt đường thành lớp (`car`).

**Lỗi phân loại sai - Classification Error (Ảnh 5):** Thực tế có 23 xe (21 `car`, 2 `cycle`). Mô hình bắt đủ 23 xe (không bỏ sót), nhưng lại dự đoán thành 20 `car`, 2 `cycle`, và **1 `van`**. 
Hệ thống ghi nhận 1 lỗi Classification. Đây chính là "tác dụng phụ" của việc chạy Optuna: Khi ép mô hình phải chú ý và tăng độ nhạy với lớp ít (van), trở nên quá nhạy cảm.
Khi gặp một lớp car có hình dáng hơi vuông vức dạng hình hộp (ví dụ: xe SUV hoặc hatchback), mô hình đã vội vàng dán nhãn nó là xe Van.
  
   """)
st.subheader("Đề xuất hướng cải thiện")
st.write("""
  Để giải quyết triệt để 2 vấn đề còn rớt lại (False Positive cảnh nền và nhầm lẫn Car/Van) cho các phiên bản nâng cấp của dự án trong tương lai, 
  Các đề xuất được đề ra như sau:

**Khắc phục lỗi False Positive (Nhận bừa cảnh nền thành xe):**
- **Tăng ngưỡng tự tin (Confidence Threshold):** Trong bước dự đoán thực tế (Inference), chỉ cần nâng nhẹ tham số `conf` (ví dụ từ 0.25 lên 0.35 hoặc 0.4). 
Các Bounding Box do dự đoán mò cảnh nền thường có độ tự tin rất thấp, 
việc tăng ngưỡng này sẽ lọc sạch hoàn toàn các lỗi FP như ở Ảnh 1.
- **Thêm ảnh Background (Negative Mining):** Bổ sung các bức ảnh chụp vòng xoay trống (không có bất kỳ xe nào) vào tập Train và không gán nhãn gì trên các ảnh này. 
Việc này ép mô hình học được khái niệm "không có gì cả", giúp nó bớt thói quen đi tìm xe trong các góc khuất hoặc bóng râm.

**Khắc phục lỗi Nhầm lẫn Car và Van (Classification Error)**
- **Tinh chỉnh trọng số Loss Function (Focal Loss):** Can thiệp sâu hơn vào hàm Loss của YOLO, tăng mức phạt (penalty) khi mô hình phân loại nhầm giữa các lớp có hình dáng tương đồng (Car vs Van, Van vs Truck).
- **Tăng cường dữ liệu có chủ đích (Targeted Augmentation):** Thu thập thêm chuyên biệt hình ảnh của các dòng xe SUV, Hatchback cỡ lớn (những xe con dễ bị nhầm thành Van) ở nhiều góc độ ánh sáng khác nhau để đưa vào tập Train. 
Điều này giúp mô hình phân định ranh giới (Decision Boundary) giữa hai lớp này sắc nét hơn.
    """)
st.subheader("Kết luận: ")
st.write("""
    **Không có lỗi vẽ khung (Localization errors = 0):** Mô hình vẽ Bounding Box cực kỳ xuất sắc. Không có hiện tượng khung bị lệch, bị quá to hay quá nhỏ so với vật thể thực tế.
    **Không bỏ sót vật thể (False Negative = 0):** Khả năng quét (Recall) của hệ thống là hoàn hảo trong các mẫu thử này. Không một chiếc xe nào lọt qua được camera mà không bị nhận diện.
    **Lỗi rất nhỏ và cục bộ:** Các lỗi sinh ra (1 FP, 1 Classification) là những sai lệch ở mức độ hoàn toàn chấp nhận được đối với bài toán nhận diện ngoài trời phức tạp, không làm thay đổi hay ảnh hưởng đến sai số đếm lưu lượng giao thông tổng thể.    """)


st.markdown("----")
st.header("Tổng kết")

st.write("""
Quá trình nghiên cứu, xây dựng và tối ưu hóa hệ thống nhận diện phương tiện giao thông tại vòng xoay sử dụng kiến trúc YOLOv11 đã đạt được những thành công đúng với mục tiêu đề ra ban đầu. 
Cụ thể:

**Lựa chọn kiến trúc tối ưu:** Qua quá trình đánh giá chéo giữa các phiên bản YOLOv11 (Nano, Small, Medium) trên tập Validation, dự án đã xác định được `YOLOv11s` là phiên bản cân bằng hoàn hảo nhất giữa tốc độ xử lý và độ chính xác, phù hợp làm nền tảng phát triển.

**Đột phá nhờ Hyperparameter Tuning:** Việc ứng dụng thuật toán Optuna để tinh chỉnh siêu tham số đã giải quyết thành công bài toán mất cân bằng dữ liệu (Class Imbalance). Mô hình `YOLOv11s-optuna` đã cải thiện rõ rệt khả năng nhận diện các phương tiện thiểu số (xe Van, xe máy) mà không làm suy giảm hiệu năng của các lớp đa số (Ô tô con, xe buýt).

**Hiệu năng thực chiến xuất sắc:** Trên tập dữ liệu test, hệ thống ghi nhận chỉ số mAP@0.5 đạt **0.940**. 

**Khả năng chống chịu nhiễu không gian tốt:** Phân tích sai số IoU chứng minh hệ thống vẽ Bounding Box cực kỳ chính xác (IoU trung bình > 0.90). Đặc biệt, mô hình không hề bị suy giảm phong độ tại khu vực mép trái ($x < 0.3$) nơi có mật độ giao thông chen chúc và phức tạp nhất (đã được chỉ ra trong bước Khám phá dữ liệu EDA).

**Độ tin cậy cao:** Phân tích Error Analysis cho thấy hệ thống không mắc lỗi định vị (Localization) và không bỏ sót phương tiện (False Negative). Các sai sót rớt lại chỉ ở mức độ vi mô (nhận diện nhầm cảnh nền hoặc nhầm lẫn giữa SUV/Van) và hoàn toàn có thể kiểm soát được bằng các biện pháp hậu xử lý (Post-processing).

### Hướng phát triển tương lai
Mặc dù hệ thống đã đạt độ chính xác cao, dự án vẫn có thể được mở rộng và hoàn thiện hơn nữa trong tương lai thông qua các hướng đi sau:

**Nâng cấp mức độ nhận diện:**
   * Áp dụng kỹ thuật **Negative Mining** (thêm ảnh nền trống vào tập huấn luyện) và tăng ngưỡng tự tin (Confidence Threshold) để triệt tiêu hoàn toàn lỗi False Positive do bóng râm và chướng ngại vật.
   * Thu thập thêm dữ liệu chuyên biệt về các loại xe dễ nhầm lẫn (Hatchback/SUV cỡ lớn) kết hợp tinh chỉnh **Focal Loss** để cải thiện độ phân giải giữa lớp Car và Van.
**Phát triển thành Pipeline End-to-End toàn diện:** * Tích hợp thêm các thuật toán Object Tracking (như ByteTrack hoặc DeepSORT) để không chỉ nhận diện mà còn theo dõi quỹ đạo (trajectory) và đếm lưu lượng phương tiện ra/vào vòng xoay.
   * Xây dựng luồng dữ liệu (Data Pipeline) đẩy kết quả đếm xe thời gian thực lên Cloud (Google Cloud/BigQuery) và trực quan hóa lên Dashboard (Power BI/Streamlit) để phục vụ bài toán phân tích kinh doanh và điều phối giao thông thông minh.
         
         """)
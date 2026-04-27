import streamlit as st
import plotly.graph_objects as go
import os
import cv2
import numpy as np
from helpers import compute_iou_simple
from PIL import Image
from ultralytics import YOLO
st.set_page_config(
    page_title="Quá trình huấn luyện và tối ưu hóa", 
    page_icon="⚙️", 
    layout="wide"
)

st.title("⚙️Quá trình huấn luyện và tối ưu hóa")
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
st.header("So sánh các phiên bản YOLOv11")
st.subheader("Giới thiệu các chỉ số đánh giá")
st.write("""
         Trong bài toán Object Detection, các chỉ số như Precision, Recall và mAP được sử dụng 
để đánh giá hiệu năng của mô hình ở cả khả năng phát hiện và định vị đối tượng.

**Precision (Độ chính xác)**  
Precision cho biết trong tất cả các dự đoán mà mô hình đưa ra, có bao nhiêu dự đoán là đúng.  
Precision cao nghĩa là mô hình ít bị dự đoán sai (False Positive thấp).

**Recall (Độ bao phủ)**  
Recall đo lường khả năng mô hình tìm ra tất cả các đối tượng thực tế trong ảnh.  
Recall cao nghĩa là mô hình ít bỏ sót đối tượng (False Negative thấp).

**mAP (mean Average Precision)**  
mAP là chỉ số tổng hợp, được tính bằng cách lấy trung bình Average Precision (AP) trên tất cả các lớp.  
Chỉ số này phản ánh toàn diện hiệu năng của mô hình, bao gồm cả phát hiện đúng và định vị chính xác.

**mAP@0.5 (mAP50)**  
Đây là mAP được tính tại ngưỡng IoU = 0.5.  
Tức là một dự đoán được xem là đúng nếu IoU ≥ 0.5.  
Chỉ số này đánh giá khả năng phát hiện đối tượng ở mức cơ bản.

**mAP@0.5:0.95 (mAP50-95)**  
Đây là mAP được tính trung bình trên nhiều ngưỡng IoU từ 0.5 đến 0.95 (bước 0.05).  
Chỉ số này khắt khe hơn, phản ánh chính xác khả năng định vị bounding box của mô hình.
         """)
st.subheader("Metrics so sánh các mô hình")
with st.expander("Metrics so sánh các mô hình"):
    col1,col2 = st.columns(2)
    with col1:
        st.image(r"result/images/comparison_map.png" , caption="mAP50-95 các mô hình YOLOv11")
    with col2:
        st.image(r"result/images/p_r_comparison.png" , caption="precision-recall các mô hình YOLOv11")

    st.header("So sánh hiệu năng các mô hình YOLOv11")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""


        ### Bảng tổng hợp hiệu năng tổng thể 

        | Model      | Precision | Recall | mAP50 | mAP50-95 |
        |------------|-----------|--------|-------|----------|
        | YOLOv11n   | 0.814     | 0.801  | 0.844 | 0.603    |
        | YOLOv11s   | 0.897     | 0.857  | 0.890 | 0.691    |
        | YOLOv11m   | **0.941** | **0.887** | **0.939** | **0.740** |
        """)
    with col2:
        st.markdown("""
    ### So sánh chỉ số mAP50-95 theo từng nhãn (Class)

    | Class  | Instances | YOLOv11n | YOLOv11s | YOLOv11m |
    |--------|-----------|----------|----------|----------|
    | car    | 3034      | 0.795    | 0.825    | **0.829**|
    | cycle  | 65        | 0.396    | 0.484    | **0.589**|
    | bus    | 21        | 0.611    | 0.696    | **0.741**|
    | truck  | 23        | 0.704    | **0.761**| 0.759    |
    | van    | 8         | 0.511    | 0.686    | **0.784**|

   
    """)
    st.markdown("---")
    st.write("""
    ### Nhận xét & Phân tích 

    Dựa vào số liệu thực nghiệm trên tập Validation, rút ra các đặc điểm sau của từng phiên bản YOLOv11:

    1. **Nhóm phương tiện chiếm phần lớn (`car`):** Đặc trưng của ô tô rất dễ nhận diện và số lượng dữ liệu đã bão hòa (hơn 3000 instances). Do đó, cả 3 mô hình đều đạt mức mAP50 tiệm cận sự hoàn hảo (~0.995). Việc tăng kích thước mô hình không mang lại lợi ích đáng kể cho lớp này.
    2. **Sự vượt trội của các mô hình lớn:**
    Về mặt tổng thể, `YOLOv11m` cho ra các thông số ấn tượng nhất, đặc biệt là kéo Recall của các lớp khó lên mức cao. Bản `YOLOv11n` bộc lộ rõ điểm yếu khi đối mặt với các lớp thiểu số (Recall của `van` chỉ đạt 0.500, bỏ sót 50% phương tiện).

    #### Phát hiện điểm bất thường (Data Anomaly / Overfitting)
    Dù `YOLOv11m` có điểm số cao nhất, nhưng khi đi sâu vào chi tiết, mô hình này đang bộc lộ những dấu hiệu **học vẹt (Overfitting)** nghiêm trọng vào bối cảnh:
    * **Sự hoàn hảo vô lý ở lớp siêu hiếm:** Lớp `van` và `bus` có số lượng cực kỳ ít (lần lượt chỉ có 8 và 21 Bounding Box trong toàn bộ tập Val). Tuy nhiên, `YOLOv11m` lại đạt mAP50 cho `van` lên tới 0.897 và `bus` là 0.975. Khả năng rất cao mô hình không thực sự học được "đặc trưng của xe van/bus", mà chỉ đang học thuộc lòng hình dáng của 1-2 chiếc xe cụ thể chạy ngang qua camera cố định trong các frame ảnh liên tiếp.
    * **Mất cân bằng Precision-Recall ở lớp `cycle`:** Mô hình `v11m` đạt Precision tuyệt đối 1.0 (không bao giờ bắt sai) nhưng Recall chỉ 0.786. Điều này cho thấy mô hình đang bị Over-confidence, chỉ dám dự đoán khi đối tượng quá rõ ràng và bỏ qua các trường hợp hơi mờ hoặc bị che khuất.

    ---

    ### Kết luận: Lựa chọn mô hình cho quá trình Hyperparameter Tuning (Optuna)

    **Quyết định: Chọn `YOLOv11s` làm mô hình cốt lõi để tiến hành tối ưu hóa siêu tham số (Hyperparameter Tuning).**

    **Lý do lựa chọn:**
    1. **Điểm "ngọt" của kiến trúc (Sweet Spot):** Bản `v11n` quá yếu để trích xuất đặc trưng của xe tải/van, trong khi bản `v11m` quá lớn dẫn đến hiện tượng học vẹt (overfit) ngay lập tức vào tập dữ liệu mất cân bằng. `YOLOv11s` cung cấp một nền tảng vững chắc (mAP50 đạt 89.0%) với chi phí tính toán hợp lý.
    2. **Dư địa để tối ưu (Room for Improvement):** Các chỉ số của `YOLOv11s` trên lớp `cycle` (Recall: 0.754) và `van` (Recall: 0.625) cho thấy mô hình này chưa bị "chín ép". Áp dụng Optuna để tìm kiếm các tham số Data Augmentation (như MixUp, HSV, Copy-Paste) sẽ ép `YOLOv11s` tập trung học được các phương tiện này một cách tổng quát hóa mà không bị rơi vào bẫy học thuộc lòng như bản `v11m`.
    """)
    

# Confusion matrix
st.markdown("-----")

st.header("Confusion Matrix")
# with st.expander("Giới thiệu Confusion Matrix"):
st.write("""
        ### Giới thiệu
        Ma trận nhầm lẫn (Confusion Matrix) là công cụ kiểm định quan trọng, giúp đánh giá chi tiết hiệu năng của mô hình thông qua sự kết hợp giữa ngưỡng độ tin cậy (Confidence Threshold) và ngưỡng không gian (IoU Threshold). 
        
        Thay vì chỉ đưa ra độ chính xác tổng thể, ma trận này phân rã kết quả dự đoán thành các nhóm lỗi cụ thể:
        - True Positive (TP - Nhận diện đúng): Mô hình dự đoán chính xác nhãn của đối tượng (ví dụ: nhận diện đúng một phương tiện giao thông) và hộp giới hạn dự đoán có độ trùng khớp với nhãn thực tế đạt yêu cầu (IoU >= Ngưỡng).
        - False Positive (FP - Nhận diện sai): Mô hình nhận diện sai. Điều này xảy ra khi mô hình khoanh vùng một vùng nền (background) thành đối tượng, hoặc đoán đúng vị trí nhưng sai nhãn, hoặc đoán đúng nhãn nhưng vị trí khoanh vùng bị lệch quá nhiều (IoU < Ngưỡng).
        - False Negative (FN - Bỏ sót): Thực tế có đối tượng xuất hiện trong ảnh nhưng mô hình không phát hiện được, hoặc đưa ra dự đoán với độ tin cậy quá thấp.
        - True Negative (TN) thường không được sử dụng trong bài toán, vì phần nền không chứa đối tượng trong một bức ảnh là vô hạn.

        Việc phân tích Confusion Matrix giúp chẩn đoán chính xác điểm yếu của bài toán. Chẳng hạn, một tỷ lệ FP cao cho thấy mô hình quá nhạy cảm và hay nhận diện nhầm, trong khi FN cao chứng tỏ mô hình đang bỏ sót nhiều mục tiêu quan trọng. 
        Từ đó, đưa ra các quyết định điều chỉnh siêu tham số hoặc làm giàu tập dữ liệu huấn luyện một cách có cơ sở.
        """)


st.subheader("Biểu diễn ma trận")
st.subheader("Ví dụ")
col1,col2 = st.columns(2)
with col1:
    st.image(r"result/yolov11n/confusion_matrix/confusion_matrix.png", caption="Ví dụ Confusion matrix")
with col2:
    st.image(r"result/yolov11n/confusion_matrix/confusion_matrix_normalized.png", caption="Ví dụ Confusion matrix normalized")
st.write("""
        Ma trận nhầm lẫn (Confusion Matrix) thường có dạng một bảng lưới  (N+1) x (N+1). 
        Các framework phổ biến triển khai YOLO (như Ultralytics) thường sử dụng quy ước: Trục dọc (Hàng) thể hiện nhãn Thực tế (Ground Truth), trong khi Trục ngang (Cột) thể hiện nhãn do mô hình Dự đoán (Predicted).

        Cấu trúc hình học của ma trận cung cấp cái nhìn toàn cảnh về hiệu năng mô hình thông qua ba vùng chính:

        - Đường chéo chính (Main Diagonal): Kéo dài từ góc trên bên trái xuống góc dưới bên phải, là vùng quan trọng nhất biểu diễn các trường hợp True Positive. Cho biết số lượng (tỷ lệ) đối tượng được mô hình phân loại chính xác nhãn và định vị đạt chuẩn IoU. 
        Trên biểu đồ nhiệt (heatmap), đường chéo này có màu càng đậm (giá trị càng gần 1.0 hoặc 100%) thì chất lượng mô hình càng cao.
        - Các ô ngoài đường chéo (Off-Diagonal Cells): Phản ánh sự nhầm lẫn giữa các lớp đối tượng (Class Confusion). 
        Ví dụ: Giao điểm giữa hàng "Xe tải" và cột "Xe oto" có giá trị cao chứng tỏ mô hình đang phân loại nhầm đặc trưng giữa hai loại phương tiện này, mặc dù đã khoanh vùng đúng vị trí.
        - Hàng và Cột "Background" (Nền): thành phần đặc thù của bài toán Object Detection, giúp bóc tách chi tiết các loại lỗi:
            - Cột Background: Khi thực tế là một vật thể nhưng mô hình lại dự đoán vào cột "Nền", đây chính là False Negative (mô hình đã bỏ lọt, không nhìn thấy đối tượng).
            - Hàng Background: Khi thực tế không có vật thể (Nền) nhưng mô hình lại dự đoán nhầm thành một lớp cụ thể ở các cột tương ứng, đây là False Positive (mô hình bị "ảo giác" hay báo động giả, ví dụ nhận diện một bóng râm thành một phương tiện).
        
        """)



st.subheader("Phân tích Ma trận Nhầm lẫn (Confusion Matrix) của các mô hình YOLOv11")
with st.expander("Confusion Matrix của các mô hình"):
    st.subheader("Confusion Matrix")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("YOLOv11n")
        st.image(r"result/yolov11n/confusion_matrix/confusion_matrix.png", caption="Confusion matrix YOLOv11n")
        st.image(r"result/yolov11n/confusion_matrix/confusion_matrix_normalized.png", caption="Confusion matrix normalized YOLOv11n")
    with col2:
        st.subheader("YOLOv11s")
        st.image(r"result/yolov11s/confusion_matrix/confusion_matrix.png", caption="Confusion matrix YOLOv11s")
        st.image(r"result/yolov11s/confusion_matrix/confusion_matrix_normalized.png", caption="Confusion matrix normalized YOLOv11s")

    with col3:
        st.subheader("YOLOv11m")
        st.image(r"result/yolov11m/confusion_matrix/confusion_matrix.png", caption="Confusion matrix YOLOv11m")
        st.image(r"result/yolov11m/confusion_matrix/confusion_matrix_normalized.png", caption="Confusion matrix normalized YOLOv11m")
    
    
    st.write("""
    Confusion Matrix cung cấp cái nhìn trực quan về khả năng phân loại của mô hình, đặc biệt là cách mô hình xử lý sự nhầm lẫn giữa các lớp phương tiện có hình dáng tương đồng và nền.

    ### Tổng quan
    - Cả 3 phiên bản (n, s, m) đều nhận diện lớp `car` cực kỳ tốt với độ chính xác tuyệt đối (1.00). Điều này dễ hiểu vì ô tô chiếm áp đảo trong tập dữ liệu.
    - Khi tăng số lượng tham số từ bản Nano (`n`) lên Medium (`m`), đường chéo chính của ma trận (đại diện cho tỷ lệ dự đoán đúng - True Positives) ngày càng đậm màu hơn. Số lượng dự đoán sai lệch (các ô ngoài đường chéo) giảm đi rõ rệt.

    ### Phân tích sự nhầm lẫn (Misclassification)

    **Mô hình YOLOv11n (Nano): Bắt yếu với lớp ít**
    - Bản `n` chỉ nhận diện đúng 12% số xe `van` thực tế. Có tới **50% xe van bị đoán nhầm thành `truck`** và **37% bị đoán nhầm thành `car`**. Điều này cho thấy phiên bản Nano không đủ độ sâu để phân biệt.
    - **Bỏ sót đối tượng (False Negatives):** Ở hàng dưới cùng (Predicted = background), mô hình bỏ lỡ khá nhiều phương tiện thực tế: 17% `cycle`, 10%  `bus`, và 9% `truck` bị mô hình coi là nền đường (không nhận diện ra vật thể).

    **Mô hình YOLOv11s (Small): Bước nhảy vọt về hiệu năng**
    - **Khắc phục lỗi bỏ sót:** Điểm sáng lớn nhất là các lớp xe cỡ lớn. Tỷ lệ nhận diện đúng của `bus` tăng mạnh từ 86% lên **95%**, và `truck` tăng từ 70% lên **96%**. Lỗi bỏ sót vật thể (nhận nhầm thành background) giảm gần như triệt để đối với các xe lớn.
    - **Cải thiện xe Van:** Khả năng nhận diện `van` tăng từ 12% lên **50%**. Tỷ lệ đoán nhầm `van` thành `truck` giảm mạnh xuống chỉ còn 12%. 

    **Mô hình YOLOv11m (Medium): Hoàn thiện chi tiết khó**
    - **Làm chủ lớp siêu hiếm (`van`):** thể hiện rõ nhất ở lớp khó nhất. Độ chính xác của `van` được đẩy lên mức **75%** (cao nhất trong 3 mô hình).
    - **Phân loại chính xác `cycle`:** Độ chính xác của lớp `cycle` đạt 86%, hạn chế tối đa việc nhầm lẫn xe với các vật thể nền lề đường.

    ### Vấn đề ảnh không xe (Background False Positives)
    Khi quan sát cột ngoài cùng bên phải (True = background, Predicted = class), cả mô hình đều mắc một điểm chung: **Hay đoán bừa `car`  tại các vùng trống**. 
    * Ví dụ, ở bản chuẩn hóa của `v11n`, trong tổng số các dự đoán sai trên nền trống, có đến 57% là nó vẽ khung và dán nhãn `car`. Bản `s` là 66% và bản `m` là 64%.
    * *Nguyên nhân:* Do lớp `car` có tần suất xuất hiện quá dày đặc, mô hình hình thành bias. Khi gặp một vệt mờ, một bóng đổ hoặc một lùm cây có hình dáng hơi hộp, nó có xu hướng an toàn nhất là dự đoán đó là một chiếc ô tô.

    ### Kết luận
    - Bản **YOLOv11n** hoàn toàn không phù hợp cho bài toán này do bị mù màu trước các xe thương mại nhỏ (`van`).
    - Bản **YOLOv11m** cho khả năng phân biệt hình thể tốt nhất nhưng đánh đổi bằng chi phí tính toán.
    - Bản **YOLOv11s** giải quyết được các điểm nghẽn chí mạng của bản `n` (đặc biệt ở `bus` và `truck`), mang lại sự cân bằng hoàn hảo, là tiền đề tuyệt vời để áp dụng tinh chỉnh (Optuna) nhằm kéo nốt tỷ lệ nhận diện `van` lên cao hơn. """)

# PR Curve
st.markdown("----")
st.header("Precision-Recall Curve")
st.subheader("Giới thiệu")
st.write("""
Precision-Recall Curve (PR Curve) là biểu đồ thể hiện mối quan hệ giữa Precision và Recall khi thay đổi ngưỡng confidence của mô hình.

- Precision phản ánh độ chính xác của các dự đoán.
- Recall phản ánh khả năng phát hiện đầy đủ các đối tượng.

Khi tăng ngưỡng confidence:
- Precision tăng (ít dự đoán sai hơn)
- Recall giảm (bỏ sót nhiều đối tượng hơn)

Ngược lại, khi giảm ngưỡng:
- Recall tăng
- Precision giảm

Đường cong càng nằm gần góc trên bên phải cho thấy mô hình có hiệu năng càng tốt. Diện tích dưới đường cong Precision-Recall được gọi là Average Precision (AP), và trung bình trên các lớp sẽ tạo thành chỉ số mAP.
Trong bài toán phát hiện phương tiện giao thông, Recall thường được ưu tiên hơn nhằm hạn chế việc bỏ sót đối tượng (False Negative), đặc biệt trong các tình huống mật độ giao thông cao.
""")

st.subheader("Đánh giá độ ổn định qua biểu đồ Precision-Recall (PR Curve)")
with st.expander("Biểu đồ PR Curve"):
    st.subheader("Biểu đồ PR Curve của các mô hình")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("YOLOv11n")
        st.image(r"result/yolov11n/pr_curve/BoxPR_curve.png", caption="BoxPR_curve YOLOv11n")
    with col2:
        st.subheader("YOLOv11s")
        st.image(r"result/yolov11s/pr_curve/BoxPR_curve.png", caption="BoxPR_curve YOLOv11s")

    with col3:
        st.subheader("YOLOv11m")
        st.image(r"result/yolov11m/pr_curve/BoxPR_curve.png", caption="BoxPR_curve YOLOv11m")
        
    st.subheader("Nhận xét:")
    st.write("""
    Biểu đồ PR Curve minh họa sự đánh đổi giữa độ chính xác (Precision - bắt đúng vật thể) và độ bao phủ (Recall - không bỏ sót vật thể).
    Diện tích nằm dưới đường cong này chính là chỉ số mAP@0.5. 
    Nhìn vào biểu đồ của 3 mô hình, rút ra các kết luận như sau:

    #### Xu hướng chung: Lớn hơn là khỏe hơn
    Đường nét đậm màu xanh dương (`all classes`) đại diện cho sức mạnh tổng thể của mô hình.
    - Từ **YOLOv11n** (mAP 0.845) sang **YOLOv11s** (mAP 0.886) và cuối cùng là **YOLOv11m** (mAP 0.939),
    đường xanh đậm này ngày càng được đẩy căng về phía góc trên cùng bên phải. 
    Trực quan hóa việc các mô hình lớn có khả năng giữ được độ chính xác cao ngay cả khi phải cố gắng bắt (Recall) nhiều phương tiện nhất có thể.

    #### Sự vượt trội của lớp Car
    Ở cả 3 biểu đồ, đường màu xanh nhạt (`car`) luôn tạo thành một góc vuông gần như hoàn hảo sát mốc 1.0. 
    Điều này khẳng định lại việc dữ liệu mất cân bằng và tập trung ở lớp `car`, 
    bất kỳ phiên bản YOLO nào cũng có thể xử lý hoàn hảo lớp phương tiện này mà không gặp hiện tượng đuối sức.

    #### Hạn chế trong nhận diện lớp Van và Xe cycle
    Sự khác biệt giữa các mô hình nằm trọn ở các đường cong:
    - **Ở mô hình YOLOv11n:** Đường màu tím (`van`) rớt đài từ rất sớm (ngay ở mức Recall 0.4) và có hình dáng bậc thang rất thưa. 
    Khúc gãy này cho thấy mô hình Nano hoàn toàn khó nhận diện xe Van. Đường màu cam (`cycle`) cũng lao dốc thẳng đứng ở cuối.
    - **Ở mô hình YOLOv11s:** Đường màu tím (`van`) đã được kéo dãn ra và bám trụ tốt hơn hẳn (mAP tăng từ 0.681 lên 0.787). 
    Sự chênh lệch giữa các đường cong được thu hẹp lại, tạo ra một cụm đường cong khá đồng đều.
    - **Ở mô hình YOLOv11m:** Ngoại trừ một cú rớt ở giai đoạn cuối của Cycle và Van, 
    tất cả các đường cong đều duy trì ở mức ngang (đỉnh) trong một khoảng thời gian rất dài (đến tận Recall 0.8).
    Khả năng trích xuất đặc trưng của bản `m` thực sự vượt trội.

    #### Kết luận
    Biểu đồ PR Curve là minh chứng rõ nét nhất cho quyết định lựa chọn mô hình trước đó. 
    Giải thích trực quan lý do tại sao **YOLOv11n** bị loại (do đường cong của các lớp thiểu số sụp đổ quá nhanh) 
    và tại sao **YOLOv11s** là lựa chọn hoàn hảo (có độ bao phủ tốt, các đường cong bám sát nhau ở mức chấp nhận được, xây dựng cơ sở đáng tin cậy cho quá trình tối ưu bằng Optuna). """)


st.markdown("-----")
st.header("Tối ưu hóa siêu tham số bằng Optuna")
st.subheader("Kết quả mô hình YOLOv11s sau khi tối ưu bằng Optuna")
col1,col2 = st.columns(2)
with col1:
        st.write("""
            

    ### Tổng quan
        
    | Metric        | Giá trị |
    |--------------|--------|
    | Precision    | 0.885  |
    | Recall       | 0.867  |
    | mAP50        | **0.925** |
    | mAP50-95     | **0.727** |

    """)
        
with col2:
    st.write("""
                
    ### Hiệu năng theo từng lớp

    | Class  | Precision | Recall | mAP50 | mAP50-95 |
    |--------|----------|--------|-------|----------|
    | car    | 0.995    | 0.997  | 0.995 | 0.827    |
    | cycle  | 0.982    | 0.850  | 0.890 | 0.529    |
    | bus    | 0.909    | 0.951  | 0.934 | 0.702    |
    | truck  | 0.855    | 0.913  | 0.953 | 0.819    |
    | van    | 0.682    | 0.625  | 0.852 | 0.756    |

                """)
st.subheader("Nhận xét:")
st.write("""
    - Mô hình đạt hiệu năng tổng thể rất cao với **mAP50 = 0.925** và **mAP50-95 = 0.727**.
    - Lớp **car** tiếp tục duy trì phong độ hoàn hảo (mAP50: 0.995).
    - Đáng chú ý, lớp **cycle** đã đạt Precision cực cao (0.982) và mức Recall được kéo lên 0.850.
    - Các phương tiện lớn/thiểu số như **truck** và **van** đều có mAP50 vượt mốc 0.85, chứng tỏ mô hình học đặc trưng rất tốt.
            """)

st.markdown("-------")
st.subheader("So sánh YOLOv11s trước và sau khi tối ưu bằng Optuna")
col1,col2 = st.columns(2)
with col1:
    st.image(r"result/images/maps11_op.png", caption="mAP50-90 Comparison s11-s11optuna")
with col2:
    st.image(r"result/images/p_r_s11_s11op.png", caption="Precision Recall Comparison")


col1,col2 = st.columns(2)
with col1:
    st.write("""

    ### Tổng quan

    | Metric        | Trước Optuna | Sau Optuna | Mức cải thiện |
    |--------------|-------------|-----------|----------|
    | Precision    | **0.897** | 0.885     | -0.012   |
    | Recall       | 0.857       | **0.867** | +0.010   |
    | mAP50        | 0.890       | **0.925** | +0.035   |
    | mAP50-95     | 0.691       | **0.727** | +0.036   |
    """)
    
with col2:
    st.subheader("Theo từng lớp (mAP50-95)")
    st.write("""
    | Class  | Trước Optuna | Sau Optuna | Mức cải thiện |
    |--------|-------------|-----------|----------|
    | car    | 0.825       | **0.827** | +0.002   |
    | cycle  | 0.484       | **0.529** | +0.045   |
    | bus    | 0.696       | **0.702** | +0.006   |
    | truck  | 0.761       | **0.819** | +0.058   |
    | van    | 0.686       | **0.756** | +0.070   |
    """)
    
st.write("""
    ### Nhận xét:

    - **Sự thành công toàn diện:** Điểm sáng lớn nhất của đợt tối ưu này là **100% các lớp đều được cải thiện chỉ số mAP50-95**. Điều này chứng minh Optuna đã tìm ra bộ tham số giúp Bounding Box ôm sát vật thể hơn trên mọi phương tiện.
    - **Giải cứu lớp thiểu số:** Các lớp khó nhằn nhất đều có bước nhảy vọt ấn tượng. Cụ thể, `van` tăng **+0.070**, `truck` tăng **+0.058**, và `cycle` tăng **+0.045**. Quá trình Tuning đã thực sự ép mô hình chú ý đến những phương tiện hiếm gặp này thay vì chỉ tập trung vào ô tô.
    - **Sự đánh đổi hợp lý (Trade-off):** Mặc dù Precision tổng thể giảm nhẹ (-0.012), nhưng bù lại Recall tăng (+0.010) và điểm mAP tổng vọt lên mạnh mẽ. Việc này thể hiện mô hình đã giảm bớt sự Over-confidence, dám bắt nhiều đối tượng hơn và độ chính xác của các khung hình (IoU) cao hơn hẳn.
    
    ### Kết luận:
    - Quá trình Hyperparameter Tuning bằng Optuna đã phát huy tác dụng xuất sắc, khắc phục triệt để tình trạng mất cân bằng giữa các lớp.
    - Mô hình YOLOv11s hiện tại đạt được sự hài hòa tuyệt vời giữa độ chính xác, khả năng tổng quát hóa và tốc độ, hoàn toàn đủ điều kiện để chốt làm phiên bản cuối cùng mang đi Inference (kiểm thử thực tế) trên tập Test.
    """)
    
st.markdown("------")
st.subheader("Confusion matrix")
col1,col2 = st.columns(2)
with col1:
    st.subheader("YOLOv11s")
    st.image(r"result/yolov11s/confusion_matrix/confusion_matrix.png", caption="Confusion matrix YOLOv11s")
    st.image(r"result/yolov11s/confusion_matrix/confusion_matrix_normalized.png", caption="Confusion matrix normalized YOLOv11s")
with col2:
    st.subheader("YOLOv11s-optuna")
    st.image(r"result/yolov11s-optuna/confusion_matrix/confusion_matrix.png", caption="Confusion matrix YOLOv11s-optuna")
    st.image(r"result/yolov11s-optuna/confusion_matrix/confusion_matrix_normalized.png", caption="Confusion matrix normalized YOLOv11s-optuna")
st.subheader("""Đánh giá hiệu quả sau khi tối ưu tham số cho mô hình (YOLOv11s vs YOLOv11s-optuna)""")

st.write("""
    Việc chạy Optuna bản chất giống như mang mô hình đi "bắt mạch kê đơn" lại. Nhìn vào hai ma trận nhầm lẫn này, có thể thấy rõ thuốc đã ngấm và phát huy tác dụng rất tốt, đặc biệt là ở những ca bệnh khó.

    **Những điểm được cải thiện:** Ở bản `s`, tỷ lệ bắt trúng xe Van chỉ là 50%. Sau khi tối ưu, tỷ lệ này được cải thiện **62%**. Bắt đầu biết phân biệt các đặc điểm nhỏ của xe Van thay vì đoán mò.
    Ở lớp `cycle` độ chính xác tăng từ 80% lên **88%**. Lỗi nhận diện sai đã giảm mạnh từ 17% xuống chỉ còn 12%. Mô hình đã bớt nhát và mạnh dạn khoanh vùng xe máy hơn.

    **Lớp `car` và `bus`:** Vẫn giữ nguyên phong độ đỉnh cao. Xe con đạt tuyệt đối 100%, xe buýt đạt 95%. Điều này chứng tỏ quá trình Optuna không làm hỏng những gì mô hình đã làm tốt từ trước.

    **Lớp`truck` giảm nhẹ:** Độ chính xác của xe tải bị rớt một chút từ 96% xuống 91%. Trong Machine Learning, đây là sự đánh đổi (trade-off) hết sức bình thường. Khi ép mô hình phải chú ý nhiều hơn vào xe Van và xe máy, nó sẽ phải san sẻ bớt sự tập trung, dẫn đến việc thi thoảng nhìn nhầm xe tải thành xe Van hoặc xe con. Tuy nhiên, mức 91% vẫn là một con số rất xuất sắc.
    
    Khi nhìn vào các vùng nhiễu hoặc hình nền, bản Optuna lại càng có xu hướng đoán bừa đó là lớp `car` nhiều hơn (tăng từ 66% lên 77%). Nó thà đoán nhầm là ô tô còn hơn là bỏ sót. 

    #### Kết luận:
    Quá trình Hyperparameter Tuning bằng Optuna **thành công**. Giải quyết được các lớp số ít (cycle, van) mà không làm ảnh hướng đến các lớp số nhiều. 
    Mô hình hiện tại đã cân bằng và toàn diện hơn rất nhiều để mang ra ứng dụng thực tế.  
    """)
st.markdown("-----")
st.header("Precision-Recall Curve")
st.subheader("So sánh Precision-Recall Curve trước và sau Optuna")

col1,col2 = st.columns(2)
with col1:
    st.subheader("YOLOv11s")
    st.image(r"result/yolov11s/pr_curve/BoxPR_curve.png", caption="BoxPR_curve YOLOv11s")
with col2:
    st.subheader("YOLOv11s-optuna")
    st.image(r"result/yolov11s-optuna/pr_curve/BoxPR_curve.png", caption="BoxPR_curve YOLOv11s-optuna")

st.subheader("Nhận xét: ")
st.write("""
Biểu đồ **PR Curve** cho thấy rất rõ việc tối ưu bằng **Optuna** đã giúp mô hình học tốt hơn. Khi so sánh kết quả trước và sau khi tuning, có thể thấy nhiều cải thiện đáng kể.
### Hiệu năng tổng thể được cải thiện rõ rệt
Nhìn vào **đường màu xanh đậm (`all classes`)** đại diện cho toàn bộ mô hình. Sau khi dùng Optuna, đường cong được nâng cao hơn và tiến sát góc trên bên phải hơn.
Điều này cho thấy mô hình vừa giữ được độ chính xác cao (**Precision**) vừa phát hiện được nhiều đối tượng hơn (**Recall**).

- mAP@0.5 tăng từ **0.886** lên **0.925**

=> Đây là dấu hiệu cho thấy mô hình mạnh hơn rõ rệt sau khi tối ưu.

### Các lớp khó nhận diện được cải thiện mạnh
Giá trị lớn nhất của Optuna là giúp mô hình xử lý tốt hơn các lớp trước đây còn yếu.
* **Lớp Cycle**. Trước khi tuning, khả năng nhận diện còn thấp. Sau khi tuning, đường cong tốt hơn rõ rệt. mAP tăng từ **0.798** lên **0.890**
* **Lớp Van**. Đây vẫn là lớp khó nhất, nhưng sau Optuna đường cong ổn định hơn và cao hơn. mAP tăng từ **0.787** lên **0.852**
* **Lớp Truck**. Sau tuning, khả năng nhận diện tốt hơn và ổn định hơn. mAP tăng từ **0.913** lên **0.955**

### Các lớp mạnh vẫn được giữ ổn định
* **Lớp Car**. Vẫn giữ hiệu năng rất cao: mAP = **0.995**. Điều này cho thấy Optuna không làm giảm chất lượng của lớp đang hoạt động tốt.
* **Lớp Bus**. Giảm nhẹ: từ **0.939** xuống **0.934**. Đây là mức giảm rất nhỏ và hoàn toàn chấp nhận được để đổi lại việc các lớp khó được cải thiện mạnh hơn.

### Kết luận
Sau khi dùng Optuna, mô hình trở nên cân bằng hơn giữa các lớp.
- Các lớp yếu như `cycle`, `van`, `truck` được cải thiện rõ rệt.
- Lớp mạnh như `car` vẫn giữ hiệu năng cao.
Điều này chứng minh quá trình tối ưu bằng **Optuna** đã giúp mô hình học tốt hơn và phù hợp hơn khi triển khai thực tế.
""")

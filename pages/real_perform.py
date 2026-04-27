import streamlit as st
import plotly.graph_objects as go
import os
import cv2
import numpy as np
from helpers import compute_iou_simple
from PIL import Image
from ultralytics import YOLO


st.set_page_config(
    page_title="Đánh giá hiệu năng thực tế", 
    page_icon="📈", 
    layout="wide"
)

st.title("📈Đánh giá hiệu năng thực tế")
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

st.header("Kết quả tổng hợp trên tập Test")

col1,col2= st.columns(2)
with col1: 
    st.subheader("Bảng tổng quan hiệu năng của các mô hình")
    st.markdown("""
| Mô hình | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| YOLOv11n | **0.969** | 0.907 | **0.968** | 0.726 |
| YOLOv11s | 0.826 | **0.939** | 0.933 | 0.709 |
| YOLOv11m | 0.918 | 0.926 | 0.966 | **0.763** |
| YOLOv11s-optuna | 0.868 | 0.891 | 0.940 | 0.736 |
                """)
    
with col2:
    st.subheader("So sánh các chỉ số mAP50-95 chi tiết theo từng lớp")
    st.markdown("""
                | Class | Instances | YOLOv11n | YOLOv11s | YOLOv11m | YOLOv11s-optuna |
| :--- | :--- | :--- | :--- | :--- | :--- |
| car | 1533 | 0.810 | 0.834 | **0.840** | 0.834 |
| cycle | 33 | 0.507 | 0.647 | **0.682** | 0.626 |
| bus | 13 | 0.759 | 0.782 | 0.784 | **0.788** |
| truck | 15 | 0.710 | 0.651 | 0.662 | **0.722** |
| van | 2 | **0.846** | 0.629 | **0.846** | 0.712 |                
                """)
st.subheader("Nhận xét và phân tích trên tập test")
st.write("""
         
**Khẳng định giá trị của việc Tuning (v11s vs v11s-optuna):**
Kết quả trên tập Test độc lập đã chứng minh việc sử dụng Optuna là hoàn toàn đúng đắn. 
Mặc dù ở tập Test, Recall tổng thể của bản Optuna (0.891) có giảm nhẹ so với bản gốc (0.939), 
nhưng bù lại Precision tăng (+0.042) và đặc biệt là các chỉ số mAP đều tăng (mAP50-95 tăng từ 0.709 lên 0.736). 
Điều này cho thấy mô hình sau khi tuning đã **tổng quát hóa tốt hơn**, 
Bounding Box ôm sát vật thể hơn trên dữ liệu hoàn toàn mới.

**Sức mạnh của phiên bản lớn (YOLOv11m):**
Đúng với lý thuyết, khi được thử nghiệm trên tập Test, bản `YOLOv11m` thể hiện sức mạnh vượt trội về khả năng phát hiện chi tiết, dẫn đầu ở chỉ số quan trọng và khắt khe nhất là **mAP50-95 (0.763)**. Lớp `cycle` thường rất khó nhận diện vì kích thước nhỏ và hay bị che khuất, nhưng bản `m` đã kéo mAP50-95 lên tới 0.682, cao nhất trong cả 4 mô hình.

**Statistical Noise ở lớp Van & Bus:**
Cột Instances của tập Test, chỉ có vỏn vẹn **2 chiếc xe Van** và **13 chiếc xe Bus**. 
* Lớp `van` của `YOLOv11n` đạt mAP50-95 lên tới 0.846, ngang bằng với bản `m`. 
Tuy nhiên, với tập mẫu chỉ có 2 chiếc xe, con số này không mang nhiều ý nghĩa thống kê (có thể do 2 chiếc xe tải nhỏ này chạy qua ở góc máy quá đẹp và rõ nét). 
* Bản `v11s-optuna` gặp hiện tượng Precision của `van` sụt mạnh xuống 0.409 dù Recall = 1.0. 
Nhận diện đúng 2 chiếc xe van đó, nhưng lại đi nhận thêm một số chiếc xe khác thành xe van.

### Kết luận chung
* **Nếu tài nguyên phần cứng cho phép (Server/GPU xịn):** `YOLOv11m` là sự lựa chọn tốt nhất với độ ổn định cao trên mọi phương tiện.
* **Nếu cần cân bằng giữa Tốc độ và Độ chính xác (Edge AI/Camera tích hợp):** `YOLOv11s-optuna` là nhà vô địch thực sự. 
Mô hình cho ra chất lượng Bounding Box bám sát vật thể rất tốt (thể hiện qua mAP50-95 vượt trội so với bản gốc) mà vẫn giữ được tốc độ FPS cao của dòng Small.
         """)
st.markdown("---")
st.header("Đánh giá chi tiết mô hình YOLOv11s-optuna trên tập test")

st.subheader("Confusion Matrix")
col1,col2 = st.columns(2)
with col1: 
    st.image("result/yolov11s-optuna/test/confusion_matrix.png",caption= "Confusion Matrix YOLOv11s-optuna")

with col2: 
    st.image("result/yolov11s-optuna/test/confusion_matrix_normalized.png",caption= "Confusion Matrix YOLOv11s-optuna")

st.write("""
        
Confusion Matrix trên tập Test là minh chứng rõ nét nhất cho khả năng hoạt động của mô hình `YOLOv11s-optuna` trong môi trường thực tế (với dữ liệu hoàn toàn mới, chưa từng xuất hiện trong quá trình huấn luyện). 

Dựa vào cả hai ma trận, rút ra những kết luận quan trọng sau:

### Hiệu năng cao và đồng đều ở các phương tiện phổ biến
Đường chéo chính của ma trận (từ trên cùng bên trái xuống) thể hiện màu xanh rất đậm, chứng tỏ mô hình có độ tự tin và tính tổng quát hóa cực kỳ cao:
* **Lớp car:** Vẫn giữ vững phong độ thống trị với tỷ lệ đoán trúng lên tới **99%** (1525/1525 xe).
* **Lớp cycle:** Đây là một bước đột phá ấn tượng trên tập Test. Lớp xe máy vốn có kích thước nhỏ, dễ bị che khuất nhưng mô hình đã nhận diện đúng tới **97%** (32/33 xe). Lỗi bỏ sót (False Negative) gần như bị triệt tiêu hoàn toàn (chỉ sót đúng 1 chiếc).
* **Lớp truck và bus:** Cả hai đều đạt mức chính xác rất cao, lần lượt là **93%** và **92%**. Việc phân biệt rạch ròi giữa hai loại xe cỡ lớn này cho thấy mô hình đã học được các đặc trưng hình khối rất tốt.

### Hạn chế về mặt dữ liệu
Nhìn vào ma trận chuẩn hóa, lớp `van` chỉ đạt tỷ lệ nhận diện đúng là 50% và 50% còn lại bị đoán nhầm thành lớp `truck`. 
Có vẻ là một sự sụt giảm hiệu năng.
Tuy nhiên, khi đối chiếu sang ma trận tuyệt đối (bên trái), **toàn bộ tập Test chỉ có vỏn vẹn 2 chiếc xe Van**. 
Việc mô hình nhận diện đúng 1 chiếc và nhầm 1 chiếc thành xe tải đã tạo ra con số 50%. 
Do quy mô mẫu quá nhỏ, tỷ lệ này **không mang ý nghĩa thống kê** và không đủ để kết luận rằng mô hình kém trong việc nhận diện xe Van thực tế.

### Tỷ lệ nhận diện sai cảnh nền (Background False Positives) cực thấp
* Khi nhìn vào cột ngoài cùng bên phải, mô hình vẫn giữ thói quen cũ là có xu hướng đoán nhầm cảnh nền thành ô tô con (57%). 
* Tuy nhiên, nếu xét về số lượng thực tế, mô hình chỉ "nhìn gà hóa thóc" ra **13 chiếc xe con** trên tổng số hàng ngàn khung hình trống. 
Đây là một tỷ lệ sai số nhiễu cực kỳ nhỏ và hoàn toàn chấp nhận được đối với một hệ thống camera giao thông ngoài trời.

### Kết luận chung
Kết quả thử nghiệm trên tập Test đã khẳng định mô hình **YOLOv11s-optuna** không hề bị "học vẹt" (Overfitting) vào tập Validation trước đó. 
Mô hình hoạt động cực kỳ sắc bén, bắt dính hầu hết các phương tiện trên đường với tỷ lệ sai sót vô cùng thấp, 
hoàn toàn đủ tiêu chuẩn để triển khai thành một ứng dụng theo dõi mật độ giao thông tại vòng xoay theo đúng mục tiêu.
        
         """)
st.markdown("------")
st.header("Precision-Recall curve")
st.image("result/yolov11s-optuna/test/BoxPR_curve.png",caption= "Precision-Recall curve YOLOv11s-optuna")

st.write("""
Biểu đồ PR Curve trên tập test là thước đo cuối cùng để khẳng định sức mạnh thực chiến của mô hình YOLOv11s-optuna. 
Nhìn vào hình dáng các đường cong, rút ra những đánh giá như sau:

### Tổng thể xuất sắc (Đường xanh đậm)
Đường `all classes` (màu xanh dương đậm) bao trùm gần như toàn bộ diện tích đồ thị, 
ôm rất sát vào góc trên cùng bên phải với chỉ số **mAP@0.5 đạt 0.940**. 
Quỹ đạo vươn xa của đường cong này chứng minh rằng: Mô hình có khả năng duy trì độ chính xác (Precision) ở mức cực kỳ cao ngay cả khi hệ thống mở rộng phạm vi tìm kiếm để không bỏ sót bất kỳ phương tiện nào (Recall tiến về 1.0).

* Lớp **`car` (0.995)** và **`bus` (0.995)**: Hai đường cong này tạo thành một góc vuông gần như tuyệt đối. 
Điều này cho thấy mô hình không hề gặp một chút khó khăn nào trong việc nhận diện ô tô con và xe buýt trên thực tế. 
Cứ hễ khoanh vùng là chắc chắn đúng.

* Lớp **`cycle` (0.965)**: Nhờ quá trình tối ưu hóa bằng Optuna, đường màu cam của lớp cycle bám sát đỉnh 1.0 và kéo dài liên tục đến tận mốc Recall > 0.95 mới bắt đầu gãy xuống.
Đây là một thành tích cực kỳ ấn tượng đối với một phương tiện có kích thước nhỏ và thường xuyên di chuyển sát nhau tại các vòng xoay.

### Giải mã hình dáng "bậc thang" của lớp Van
* Lớp **`van` (0.828)**: Đường màu tím có hình dáng gập gãy thành những bậc thang rất lớn và dốc.
Về mặt toán học, điều này hoàn toàn hợp lý và giải thích được: Do tập Test chỉ có vỏn vẹn **2 chiếc xe Van**, 
biểu đồ PR Curve không có đủ số điểm dữ liệu để tạo ra một đường cong mượt mà.
Tuy biểu đồ bị gãy khúc do kích thước mẫu nhỏ, nhưng diện tích dưới đường cong (0.828) vẫn là một mức điểm khá tốt, 
cho thấy mô hình xử lý ổn định trên các mẫu thử hiếm hoi này.

### Kết luận
Mô hình YOLOv11s-optuna đã vượt qua bài kiểm tra cuối cùng trên tập Test một cách xuất sắc. 
Sự đồng đều giữa các đường cong (từ phương tiện lớn đến nhỏ) khẳng định hệ thống hoàn toàn sẵn sàng để đưa vào triển khai thực tế, 
phục vụ cho việc theo dõi và phân tích lưu lượng giao thông tại vòng xoay.
         """)

st.markdown("-----")
st.header("Phân tích sai số IoU theo không gian")
st.subheader("Giới thiệu")
st.write("""
         IoU (Intersection over Union) là một chỉ số quan trọng trong bài toán Object Detection, 
được sử dụng để đánh giá độ chính xác của bounding box mà mô hình dự đoán.

IoU đo lường mức độ chồng lắp giữa bounding box dự đoán (Prediction) và bounding box thực tế (Ground Truth), 
được tính bằng tỷ lệ giữa diện tích giao nhau và diện tích hợp của hai vùng này.
""")
st.write("**IoU được tính theo công thức:**")
st.latex(r"IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}")

st.write("""
Giá trị IoU nằm trong khoảng từ 0 đến 1:
- IoU = 1: hai bounding box trùng khớp hoàn toàn
- IoU ≥ 0.5: dự đoán được xem là chấp nhận được
- IoU < 0.5: dự đoán chưa chính xác

Chỉ số này giúp đánh giá khả năng định vị (localization) của mô hình, 
tức là mô hình có vẽ đúng vị trí và kích thước của đối tượng hay không.
""")
# st.write("**IoU được tính theo công thức:**")
# st.latex(r"IoU = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{A \cap B}{A \cup B}")

st.subheader("Trực quan hóa IoU (Intersection over Union)")
st.write("Dưới đây là demo giúp hiểu được cách hoạt động của IoU")


st.markdown("""
    <style>
    .main { background-color: #f7f6f0; }
    div[data-testid="stMetricValue"] { font-size: 24px; font-weight: 500; color: #BA7517; }
    .stat-card {
        background-color: #f1f0e8;
        border-radius: 10px;
        padding: 15px;
        text-align: left;
    }
    .verdict-box {
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 500;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIC TÍNH TOÁN ---
gt_x, gt_y, gt_w, gt_h = 160, 90, 200, 160

def compute_iou(ox, oy, sz):
    pred_w, pred_h = sz, sz
    pred_x = (gt_x + gt_w/2 + ox) - pred_w/2
    pred_y = (gt_y + gt_h/2 + oy) - pred_h/2
    
    xA = max(gt_x, pred_x)
    yA = max(gt_y, pred_y)
    xB = min(gt_x + gt_w, pred_x + pred_w)
    yB = min(gt_y + gt_h, pred_y + pred_h)
    
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    union_area = (gt_w * gt_h) + (pred_w * pred_h) - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou, inter_area, union_area, (pred_x, pred_y, pred_w, pred_h), (xA, yA, inter_w, inter_h)


container = st.container()


col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
with col_ctrl1:
    ox = st.slider("Pred X offset", -80, 80, 40)
with col_ctrl2:
    oy = st.slider("Pred Y offset", -80, 80, 30)
with col_ctrl3:
    sz = st.slider("Pred size", 60, 220, 140)

# Tính toán dữ liệu dựa trên slider
iou, inter, union, pred_box, inter_box = compute_iou(ox, oy, sz)

# Vẽ biểu đồ Plotly (thay cho Canvas)
fig = go.Figure()

# Ground Truth
fig.add_shape(type="rect", x0=gt_x, y0=gt_y, x1=gt_x+gt_w, y1=gt_y+gt_h,
              line=dict(color="#0F6E56", width=2.5))
fig.add_annotation(x=gt_x+5, y=gt_y+10, text="Ground truth", showarrow=False, 
                   font=dict(color="#0F6E56", size=12), xanchor="left")

# Prediction
fig.add_shape(type="rect", x0=pred_box[0], y0=pred_box[1], x1=pred_box[0]+pred_box[2], y1=pred_box[1]+pred_box[3],
              line=dict(color="#BA7517", width=2.5, dash="dash"))
fig.add_annotation(x=pred_box[0]+5, y=pred_box[1]-10, text="Prediction", showarrow=False, 
                   font=dict(color="#BA7517", size=12), xanchor="left")

# Intersection Fill
if inter > 0:
    fig.add_shape(type="rect", x0=inter_box[0], y0=inter_box[1], x1=inter_box[0]+inter_box[2], y1=inter_box[1]+inter_box[3],
                  fillcolor="rgba(186, 117, 23, 0.25)", line=dict(width=0))
    fig.add_annotation(x=inter_box[0]+inter_box[2]/2, y=inter_box[1]+inter_box[3]/2, text="∩", 
                       showarrow=False, font=dict(size=18, color="#3d3d3a"))

# UI Formula & Score ở bên phải (Annotation)
fig.add_annotation(x=480, y=50, text="IoU formula", showarrow=False, font=dict(size=12, color="gray"))
fig.add_annotation(x=480, y=85, text="Intersection<br>────────<br>Union", showarrow=False, font=dict(size=14))
fig.add_annotation(x=480, y=160, text=f"<b>{iou:.3f}</b>", showarrow=False, font=dict(size=36, color="#BA7517"))
fig.add_annotation(x=480, y=190, text="IoU score", showarrow=False, font=dict(size=12, color="gray"))

fig.update_layout(
    xaxis=dict(range=[0, 640], showticklabels=False, showgrid=False, zeroline=False),
    yaxis=dict(range=[320, 0], showticklabels=False, showgrid=False, zeroline=False),
    width=800, height=350, margin=dict(l=0, r=0, t=0, b=0),
    plot_bgcolor="#f7f6f0", paper_bgcolor="#f7f6f0"
)

container.plotly_chart(fig, use_container_width=True)

st.write("") 
m_col1, m_col2, m_col3, m_col4 = st.columns([1, 1, 1, 1.5])

with m_col1:
    st.markdown(f"<div class='stat-card'><small>IoU score</small><br><span style='color:#BA7517; font-size:22px; font-weight:bold;'>{iou:.3f}</span></div>", unsafe_allow_html=True)

with m_col2:
    st.markdown(f"<div class='stat-card'><small>Intersection</small><br><span style='font-size:22px; font-weight:bold;'>{int(inter)} px²</span></div>", unsafe_allow_html=True)

with m_col3:
    st.markdown(f"<div class='stat-card'><small>Union</small><br><span style='font-size:22px; font-weight:bold;'>{int(union)} px²</span></div>", unsafe_allow_html=True)

with m_col4:
    if iou >= 0.7:
        color, bg, txt = "#28a745", "#d4edda", "✓ Good detection (≥ 0.7)"
    elif iou >= 0.5:
        color, bg, txt = "#856404", "#fff3cd", "~ Acceptable (≥ 0.5)"
    else:
        color, bg, txt = "#721c24", "#f8d7da", "✗ Poor detection (< 0.5)"
    
    st.markdown(f"<div class='verdict-box' style='background-color:{bg}; color:{color}; border: 1px solid {color}33;'>{txt}</div>", unsafe_allow_html=True)
st.write("Thay đổi các thanh trượt để xem sự thay đổi của điểm IoU.")


st.subheader("Đánh giá kết quả IoU của mô hình YOLOv11s-optuna trên tập test")

st.image("result/yolov11s-optuna/iou/IOU_s11_optuna.png", caption = "IOU_model_optuna")
st.write("""
        KẾT QUẢ ĐÁNH GIÁ ĐỘ KHỚP KHUNG HÌNH (IoU):
        - Vùng biên trái (x < 0.3): Trung bình IoU = 0.91
        - Vùng bình thường (x >= 0.3): Trung bình IoU = 0.90
        """)
st.markdown("----------")

st.write("""
Từ bước Khám phá dữ liệu (EDA), bản đồ nhiệt (Heatmap) đã chỉ ra rằng mật độ phương tiện tiến vào vòng xoay tập trung rất dày đặc ở khu vực mép trái của camera (tọa độ x < 0.3). 
Để đánh giá xem sự đông đúc và hiệu ứng góc nhìn này có làm giảm chất lượng nhận diện của mô hình hay không.

### Tính ổn định xuyên suốt khung hình (Scatter Plot)
Biểu đồ phân tán (bên trái) cho thấy ở cả hai vùng không gian, các điểm dự đoán (chấm xanh và đỏ) đều tập trung bám rất sát vào dải IoU từ 0.8 đến 1.0. 
Không hề có hiện tượng suy giảm diện rộng khi phương tiện di chuyển dần về các góc khuất của camera. 
Điều này chứng tỏ khung Bounding Box do Optuna tối ưu đã ôm cực kỳ sát vào vật thể bất kể vị trí.

### Đồng đều về mặt thống kê (Boxplot)
Biểu đồ hộp (bên phải) cung cấp góc nhìn rõ nét hơn về sự phân bố:
- Hộp của vùng biên trái (màu xanh) có phần thân khá hẹp và trung vị (đường gạch ngang giữa hộp) nằm ở mức rất cao, ngang ngửa thậm chí nhỉnh hơn một chút so với vùng bình thường (màu đỏ).
- Điều này bác bỏ hoàn toàn lo ngại ban đầu về việc mật độ xe đông ở mép trái sẽ làm giảm độ chính xác của khung hình. 

### Chỉ số thực tế ấn tượng
Điểm IoU trung bình ở khu vực mép trái (x < 0.3) đạt mức **0.91**, nhỉnh hơn một chút so với phần còn lại của khung hình (**0.90**). 
Đối với bài toán phát hiện đối tượng (Object Detection), ngưỡng IoU trung bình đạt trên 0.90 trên tập Test là một kết quả vượt ngoài mong đợi, 
khẳng định hệ thống vẽ khung cực kỳ chuẩn xác và ổn định.

### Kết luận
Mô hình YOLOv11s-optuna đã xử lý hoàn hảo hiện tượng mất cân bằng mật độ giao thông theo không gian. 
Hệ thống hoàn toàn đáp ứng được yêu cầu khắt khe về độ khớp khung hình, sẵn sàng cho các nhiệm vụ phức tạp hơn như đếm xe hoặc theo dõi quỹ đạo (tracking) tại vòng xoay thực tế.
         
         """)



import streamlit as st
import plotly.graph_objects as go
import os
import cv2
import numpy as np
from helpers import compute_iou_simple
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="Demo Error Analysis", 
    page_icon="🧪", 
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

st.title("🧪Demo Error Analysis")
st.write("Phần demo này sẽ trả ra kết quả dự đoán và các lỗi phân tích")
col1, col2, col3 = st.columns(3)
with col1:
    uploaded_image = st.file_uploader("Upload Image",type=["jpg", "png", "jpeg"])
with col2:
    uploaded_labels = st.file_uploader("Upload Label (.txt)",type=["txt"])
with col3:
    uploaded_classes = st.file_uploader("Upload Classes (.txt)",type=["txt"])
 
MODEL_PATHS = {
    "YOLOv11 Nano": r"YOLOv11_training/training_backup_YOLOv11n/weights/best.pt",
    "YOLOv11 Small": r"YOLOv11_training/training_backup_YOLOv11s/weights/best.pt",
    "YOLOv11 Medium": r"YOLOv11_training/training_backup_YOLOv11m/weights/best.pt",
    "YOLOv11 Small(optuna)":r"YOLOv11_training/model_tuning/weights/best.pt",
}
 
model_choice = st.selectbox("Chọn model", list(MODEL_PATHS.keys()))
confidence   = st.slider("Confidence", 0.0, 1.0, 0.25)
 
 
@st.cache_resource
def load_model(path: str):
    return YOLO(path)
 

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter  = max(0, x2 - x1) * max(0, y2 - y1)
    area1  = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2  = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-6)
 
 
def format_counts(counts):
    if not counts:
        return "None"
    return ", ".join(f"{k}: {v}" for k, v in counts.items())
 
 
# ──────────────────────────────────────────
# colors corrected to OpenCV BGR order
#   Convention  |  RGB          |  BGR (OpenCV)
#   TP  green   |  (0,255,0)    |  (0,255,0)      ← same
#   FP  red     |  (255,0,0)    |  (0,0,255)      ← swapped
#   FN  blue    |  (0,0,255)    |  (255,0,0)      ← swapped
#   Loc yellow  |  (255,215,0)  |  (0,215,255)    ← corrected
#   Cls purple  |  (128,0,128)  |  (128,0,128)    ← same
#   Dup orange  |  (255,165,0)  |  (0,165,255)    ← corrected
# ──────────────────────────────────────────
COLOR_TP  = (0,255,0)     # green
COLOR_FP  = (0,0,255)    # red (BGR)
COLOR_FN  = (255,0,0)     # blue  (BGR)
COLOR_LOC = (0,215,255)   # yellow (BGR)
COLOR_CLS = (128,0,128)   # purple
COLOR_DUP = (0,165, 255)   # orange (BGR)
 
 
if st.button("🚀 Predict & Analyze"):
    if uploaded_image and uploaded_labels and uploaded_classes:

       
        uploaded_image.seek(0)
        uploaded_labels.seek(0)
        uploaded_classes.seek(0)
 
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        h, w, _ = img.shape
 
        # Load classes 
        class_names = uploaded_classes.read().decode("utf-8").strip().splitlines()
        class_names = [c.strip() for c in class_names if c.strip()]
 
        # Load ground truth
        gt_boxes  = []
        gt_counts = {}
        for line in uploaded_labels.read().decode("utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id, cx, cy, bw, bh = map(float, parts[:5])
            x1 = int((cx - bw / 2) * w);  y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w);  y2 = int((cy + bh / 2) * h)
            gt_boxes.append({"box": [x1, y1, x2, y2], "class": int(cls_id)})
            name = class_names[int(cls_id)] if int(cls_id) < len(class_names) else str(int(cls_id))
            gt_counts[name] = gt_counts.get(name, 0) + 1
 
        # Draw ground truth
        img_gt = img.copy()
        for g in gt_boxes:
            x1, y1, x2, y2 = g["box"]
            cv2.rectangle(img_gt, (x1, y1), (x2, y2), (255, 255, 255), 2)
            name = class_names[g["class"]] if g["class"] < len(class_names) else str(g["class"])
            cv2.putText(img_gt, name, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
 
        #  Run model
        model=load_model(MODEL_PATHS[model_choice])   # BUG 6 FIX: cached
        results = model.predict(img, conf=confidence, verbose=False)
 
        #  Parse predictions
        pred_boxes  = []
        pred_counts = {}
        for b in results[0].boxes:
            cls_id = int(b.cls)
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            conf_val = float(b.conf)
            pred_boxes.append({"box": [x1, y1, x2, y2], "class": cls_id, "conf": conf_val})
            name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            pred_counts[name] = pred_counts.get(name, 0) + 1
 
        # Error analysis
        TP, FP, FN = [], [], []
        loc_err, cls_err, dup = [], [], []
        matched_gt = set()
 
        for p in pred_boxes:
            best_iou_score = 0.0   
            best_j         = -1
 
            for j, g in enumerate(gt_boxes):
                iou_score = compute_iou(p["box"], g["box"])  
                if iou_score > best_iou_score:
                    best_iou_score = iou_score
                    best_j = j
 
            if best_iou_score >= 0.5:
                if best_j not in matched_gt:
                    matched_gt.add(best_j)
                    g = gt_boxes[best_j]
 
                    if p["class"] != g["class"]:
                        cls_err.append(p)          
                    else:
                        TP.append(p)                
                        if best_iou_score < 0.75:   
                            loc_err.append(p)
                else:
                    dup.append(p)                   
            else:
                FP.append(p)                        
 
        for i, g in enumerate(gt_boxes):
            if i not in matched_gt:
                FN.append(g)                       
 
        #  Draw error analysis image 
        img_error = img.copy()
 
        def draw_boxes(boxes, color, label_prefix=""):
            for o in boxes:
                x1, y1, x2, y2 = o["box"]
                cv2.rectangle(img_error, (x1, y1), (x2, y2), color, 2)
                if label_prefix:
                    cv2.putText(img_error, label_prefix, (x1, max(y1 - 4, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
 
        draw_boxes(TP,COLOR_TP,  "TP")
        draw_boxes(FP,COLOR_FP,  "FP")
        draw_boxes(FN,COLOR_FN,  "FN")
        draw_boxes(loc_err,COLOR_LOC, "Loc")
        draw_boxes(cls_err,COLOR_CLS, "Cls")
        draw_boxes(dup,COLOR_DUP, "Dup")
 
        # Visualization 
        st.header("📷 Visualization")
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), caption="Original",use_container_width=True)
        
        st.subheader("Predict and Error")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(cv2.cvtColor(img_gt,cv2.COLOR_BGR2RGB), caption="Ground Truth",use_container_width=True)
        with c2:
            st.image(results[0].plot(),caption="Prediction",use_container_width=True)
        with c3:
            st.image(cv2.cvtColor(img_error, cv2.COLOR_BGR2RGB), caption="Error Analysis", use_container_width=True)
 
        # Quy ước màu
        st.subheader("Quy ước màu")
        st.markdown("""
        | Color | Error type |
        |-------|-----------|
        | 🟩 Green  | TP — True Positive |
        | 🟥 Red    | FP — False Positive |
        | 🟦 Blue   | FN — False Negative |
        | 🟨 Yellow | Localization error |
        | 🟪 Purple | Classification error |
        | 🟧 Orange | Duplicate detection |
        """)
 
        # ── Metrics
        st.subheader("📊 Metrics")
 
        precision= len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0.0
        recall= len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
 
        col1,col2,col3 = st.columns(3)
        with col1: 
            st.write("Precision", f"{precision:.3f}")
            st.write("Recall",    f"{recall:.3f}")
            st.write("F1-score",  f"{f1:.3f}")
        with col2:
            st.write(f"""
            **Ground Truth:**  
            Total: {len(gt_boxes)} \n
            {format_counts(gt_counts)}
            
            **Prediction:** 
            Total: {len(pred_boxes)} \n
            {format_counts(pred_counts)} """)
        with col3: 
            st.write(f"""
                **Error Analysis:**  
                TP: {len(TP)} \n
                FP: {len(FP)} \n
                FN: {len(FN)}  \n
                Localization errors: {len(loc_err)}  
                Classification errors: {len(cls_err)}  
                Duplicate detections: {len(dup)}
            """)
 
    else:
        st.warning("Vui lòng upload đầy đủ file")
        
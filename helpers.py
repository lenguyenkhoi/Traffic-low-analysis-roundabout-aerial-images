import os
from pathlib import Path
from PIL import Image
import glob
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import seaborn as sns
from ultralytics import YOLO
# Hàm kiểm tra tính nhất quán của ảnh với nhãn
def check_consistency(image_path, label_path, image_extensions=[".jpg", ".jpeg", ".png"], label_extensions=[".txt"]):
    """_summary_ hàm kiểm tra tính nhất quán của file ảnh và file nhãn 

    Args:
        image_path (_type_): đường dẫn đến file ảnh
        label_path (_type_): đường dẫn đến file nhãn
        image_extensions (list, optional): danh sách các đuôi của file ảnh. Defaults to [".jpg", ".jpeg", ".png"].
        label_extensions (list, optional): danh sách các các đuôi của file nhãn. Defaults to [".txt"].

    Returns:
        _type_: 
    """
    # Kiểm tra xem file ảnh có tồn tại không
    if not os.path.exists(image_path):
        print(f"Ảnh không tồn tại: {image_path}")
        return False

    # Kiểm tra xem file nhãn có tồn tại không
    if not os.path.exists(label_path):
        print(f"Nhãn không tồn tại: {label_path}")
        return False

    # 1. Lấy danh sách tên file của ảnh và nhãn. Không lấy đuôi của ảnh và nhãn 
    image_files = [f.stem for f in Path(image_path).iterdir() if f.suffix.lower() in image_extensions] # File ảnh
    label_files = [f.stem for f in Path(label_path).iterdir() if f.suffix.lower() in label_extensions] # File nhãn 

    # Chuyển danh sách sang dạng Set để dễ dàng lọc các ảnh và nhãn không khớp nhau
    set_images = set(image_files) 
    set_labels = set(label_files)

    # 2. Tìm các file bị lệch (Mismatches)
    images_without_labels = set_images - set_labels # Ảnh không có file nhãn tương ứng
    labels_without_images = set_labels - set_images # Nhãn không có file ảnh tương ứng

    # 3. In kết quả chẩn đoán
    print(f"Tổng số file ảnh: {len(set_images)}")
    print(f"Tổng số file nhãn: {len(set_labels)}")

    # kiểm tra nếu không có sự khác về số lượng file ảnh và file nhãn, thì dữ liệu hoàn toàn nhất quán
    if not images_without_labels and not labels_without_images:
        print("File ảnh và file nhãn nhất quán")
    else:
        print("File ảnh và file nhãn không nhất quán")
        # in ra số các file không có nhãn hoặc không có ảnh để kiểm tra thử
        if images_without_labels:
            print(f"Tổng số file ảnh không có file nhãn: {len(images_without_labels)}")
            print("Ví dụ:")
            # In ra tối đa 5 file để kiểm tra thử
            for img in list(images_without_labels)[:5]:
                print(f"{img}.jpg/.png")
        # in ra số các file nhãn không có ảnh để kiểm tra thử
        if labels_without_images:
            print(f"Tổng số file nhãn KHÔNG CÓ ảnh tương ứng: {len(labels_without_images)}")
            print("Ví dụ: ")
            for lbl in list(labels_without_images)[:5]:
                print(f"{lbl}.txt")
    return True

# Hàm đọc file classes.txt và trả về một dictionary ánh xạ class_id sang class_name
def load_classes(class_path):
    """ Hàm load vật thể(class) trong hình từ file classes.txt

    Args:
        class_path (_type_): _description_ đường dẫn file classes.txt

    Returns:
        _type_: _description_ trả về dict {class_id: class_name}
    """
    classes_path = class_path
    class_names = {}
    # kiểm tra file có tồn tại không
    if os.path.exists(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines): 
                name = line.strip()  # Loại bỏ khoảng trắng và ký tự xuống dòng
                if name: # Chỉ lấy các dòng có nhãn
                    class_names[i] = name
    else:
        print("Không tìm thấy file classes.txt trong thư mục.")
    return class_names

# Hàm tạo tọa độ bbox
def coordinate_bbox(x_center, y_center, width, height):
    """_summary_ Hàm tạo bounding box từ tọa độ trung tâm và kích thước

    Args:
        x_center (_type_): Tọa độ x của trung tâm
        y_center (_type_): Tọa độ y của trung tâm
        width (_type_): Chiều rộng của bounding box
        height (_type_): Chiều cao của bounding box

    Returns:
        _type_: Tọa độ của bounding box dưới dạng (x_min, y_min, x_max, y_max)
    """
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return x_min, y_min, x_max, y_max

# EDA 
def analyze_and_plot_eda(data,class_names):
    """_summary_ Hàm EDA cho toàn bộ dataset

    Args:
        data (_type_): _description_ đường dẫn tới thư mục chứa ảnh và file nhãn
    """    
    # Lấy file tọa độ không lấy file classes
    label_files = glob.glob(os.path.join(data, "*.txt"))
    label_files = [f for f in label_files if not f.endswith("classes.txt")]
    
    data = []
    
    # Đọc tọa độ từ các file nhãn
    for toado in label_files:
        with open(toado, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1]) # tọa độ tâm x của bbox, được chuẩn hóa từ 0 đến 1
                    y_center = float(parts[2]) # tọa độ tâm y của bbox, được chuẩn hóa từ 0 đến 1
                    width = float(parts[3]) # chiều rộng của bbox, được chuẩn hóa từ 0 đến 1
                    height = float(parts[4]) # chiều cao của bbox, được chuẩn hóa từ 0 đến 1
                    bbox_area = width * height # diện tích bbox 
                    
                    data.append({
                        "Class_ID": cls_id,
                        "Class_Name": class_names.get(cls_id, f"Class {cls_id}"),
                        "x_center": x_center,
                        "y_center": y_center,
                        "bbox_area": bbox_area
                    })
                    
    if not data:
        print("Không tìm thấy file")
        return
        
    df = pd.DataFrame(data)
    print(f"Số bbox đã phân tích: {len(df)}")
    # Vẽ plot
    plt.figure(figsize=(24, 6))
    sns.set_theme(style="whitegrid")
    
    # Class Distribution
    plt.subplot(1, 3, 1)
    # Sắp xếp các cột từ cao xuống thấp
    order = df["Class_Name"].value_counts().index
    
    #kiểm tra số lượng bbox của từng lớp
    class_counts = df["Class_Name"].value_counts()
    order_df = pd.DataFrame(order, columns=["Class_Name"])
    order_df["Count"] = order_df["Class_Name"].map(class_counts)
    print(order_df)
    
    ax = sns.countplot(data=df, x="Class_Name", order=order, palette="viridis")
    
    plt.title("Class Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Số lượng bbox", fontsize=12)
    
    # Ghi số lượng cụ thể lên đầu mỗi cột
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha="center", va="bottom", fontsize=11, xytext=(0, 5), 
                    textcoords="offset points")

    # Heatmap vị trí tâm vật thể (Spatial Bias)
    plt.subplot(1, 3, 2)
    # Sử dụng Kernel Density Estimation (KDE) của Seaborn để vẽ vùng nhiệt
    sns.kdeplot(x=df["x_center"], y=df["y_center"], cmap="Reds", fill=True, bw_adjust=0.5, thresh=0.05)
    plt.xlim(0, 1)
    plt.ylim(0, 1) # Đảo ngược trục y để phù hợp với hệ tọa độ ảnh
    plt.title("Heatmap mật độ và vị trí phương tiện", fontsize=14, fontweight="bold")
    plt.xlabel("Tọa độ X", fontsize=12)
    plt.ylabel("Tọa độ Y", fontsize=12)

    
    # BBox area distribution
    plt.subplot(1, 3, 3)
    # Sắp xếp các cột từ cao xuống thấp
    ax = sns.histplot(data=df, x="bbox_area", bins=50, kde=True, color="blue")
    plt.title("BBox Area Distribution", fontsize=14, fontweight="bold")
    plt.xlabel("BBox Area (Normalized)", fontsize=12)
    plt.ylabel("Số lượng bbox", fontsize=12)
    plt.tight_layout()
    plt.show()


# Hàm EDA cho tập train và val
def analyze_yolo_eda(dataset_path, class_names, dataset_name="Dataset"):
    """
    Hàm EDA cho 1 tập dữ liệu YOLO (train hoặc val)

    Args:
        dataset_path: đường dẫn tới folder (chứa images/ và labels/)
        class_names: dict {id: class_name}
        dataset_name: tên hiển thị (Train / Val)
    """
    
    label_path = os.path.join(dataset_path, "labels")
    label_files = glob.glob(os.path.join(label_path, "*.txt"))
    label_files = [f for f in label_files if not f.endswith("classes.txt")]
    
    data = []

    # Đọc label
    for file in label_files:
        with open(file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    bbox_area = width * height
                    
                    data.append({
                        "Class_ID": cls_id,
                        "Class_Name": class_names.get(cls_id, f"Class {cls_id}"),
                        "x_center": x_center,
                        "y_center": y_center,
                        "bbox_area": bbox_area
                    })
    if not data:
        print(f"{dataset_name}: Không có dữ liệu!")
        return
    
    df = pd.DataFrame(data)

    print(f"{dataset_name} - Số bbox: {len(df)}")
    print(df["Class_Name"].value_counts())

    # vẽ chart
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(24, 6))

    # Class Distribution
    plt.subplot(1, 3, 1)
    order = df["Class_Name"].value_counts().index
    ax = sns.countplot(data=df, x="Class_Name", order=order, palette="viridis")
    plt.title(f"{dataset_name} - Class Distribution", fontweight="bold")
    plt.xticks(rotation=45)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2., p.get_height()),ha="center", va="bottom", fontsize=10)
    
    # Heatmap
    plt.subplot(1, 3, 2)
    sns.kdeplot(x=df["x_center"],y=df["y_center"], cmap="Reds", fill=True, bw_adjust=0.5, thresh=0.05)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"{dataset_name} - Spatial Distribution", fontweight="bold")

    # BBox Area Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(df["bbox_area"], bins=50, kde=True)
    plt.title(f"{dataset_name} - BBox Area Distribution", fontweight="bold")
    plt.tight_layout()
    plt.show()

# Hàm tính iou
def calculate_iou(box1, box2):
    """Tính IoU giữa 2 Bounding Box định dạng YOLO [x_center, y_center, w, h]"""
    # Chuyển đổi từ tọa độ tâm sang tọa độ góc [x_min, y_min, x_max, y_max]
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    # Tìm tọa độ phần giao nhau (Intersection)
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # Tính diện tích từng Box
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    # Tính IoU
    union_area = b1_area + b2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def compute_iou_simple(box1, box2):
    """_summary_ Hàm tính IoU giữa 2 bounding box định dạng [x1, y1, x2, y2]

    Args:
        box1 (_type_): _description_ bounding box 1 dưới dạng [x1, y1, x2, y2]
        box2 (_type_): _description_ bounding box 2 dưới dạng [x1, y1, x2, y2]

    Returns:
        _type_: _description_ trả về giá trị IoU giữa 2 bounding box
    """
    ix1, iy1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    ix2, iy2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter_area
    
    if union > 0:
        return inter_area / union
    else: 
        return 0



def load_ground_truth(label_path, w_img, h_img):
    """_summary_ hàm load Ground Truth từ file nhãn và chuyển đổi sang định dạng [x1, y1, x2, y2] để tính IoU với dự đoán của mô hình

    Args:
        label_path (_type_): _description_ đường dẫn tới file nhãn
        w_img (_type_): _description_ độ rộng của ảnh (dùng để chuyển đổi tọa độ chuẩn hóa sang pixel)
        h_img (_type_): _description_ độ cao của ảnh (dùng để chuyển đổi tọa độ chuẩn hóa sang pixel)

    Returns:
        _type_: _description_ trả về list các bounding box GT dưới dạng [x1, y1, x2, y2]
    """
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) >= 5:
                    x_c, y_c, w, h = parts[1:5]
                    x1 = int((x_c - w / 2) * w_img)
                    y1 = int((y_c - h / 2) * h_img)
                    x2 = int((x_c + w / 2) * w_img)
                    y2 = int((y_c + h / 2) * h_img)
                    gt_boxes.append([x1, y1, x2, y2])
    return gt_boxes

# Hàm tạo plot
def plot_iou(IMAGES_DIR, LABELS_DIR,MODEL_PATH):
    model = YOLO(MODEL_PATH)
    image_paths = glob.glob(os.path.join(IMAGES_DIR, '*.jpg'))

    iou_results = []

    print("Đang phân tích sai số BBox")

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        label_path = os.path.join(LABELS_DIR, img_name.replace('.jpg', '.txt'))
        
        # 1. Đọc Ground Truth (Nhãn gốc)
        ground_truths = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = list(map(float, line.strip().split()))
                    if len(parts) >= 5:
                        ground_truths.append({
                            'class': int(parts[0]),
                            'bbox': parts[1:5], # [x_c, y_c, w, h]
                            'x_center': parts[1]
                        })
                        
        # chạy dự đoán
        results = model.predict(source=img_path, conf=0.25, verbose=False)
        predictions = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # YOLO v11 xuất BBox định dạng xywhn (chuẩn hóa 0-1)
                pred_bbox = box.xywhn[0].cpu().numpy()
                predictions.append(pred_bbox)

        # Ghép cặp GT và Prediction để tìm IoU cao nhất cho mỗi class
        for gt in ground_truths:
            best_iou = 0
            for pred_bbox in predictions:
                iou = calculate_iou(gt['bbox'], pred_bbox)
                if iou > best_iou:
                    best_iou = iou
                    
            # Phân loại vị trí: "Vùng biên trái" nếu x < 0.3, ngược lại là "Vùng bình thường"
            region = "Vùng biên trái (x < 0.3)" if gt['x_center'] < 0.3 else "Vùng bình thường (x >= 0.3)"
            
            iou_results.append({
                'x_center': gt['x_center'],
                'IoU_Score': best_iou,
                'Region': region
            })

    df_iou = pd.DataFrame(iou_results)

    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")

    # Biểu đồ 1: Scatter plot (Tọa độ X vs IoU)
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_iou, x='x_center', y='IoU_Score', hue='Region', alpha=0.6, palette="Set1")
    plt.axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Ranh giới biên (x=0.3)')
    plt.title('Phân bố IoU theo tọa độ X', fontsize=14, fontweight='bold')
    plt.xlabel('Tọa độ tâm X (Chuẩn hóa)', fontsize=12)
    plt.ylabel('Chỉ số IoU', fontsize=12)
    plt.legend()

    # Biểu đồ 2: Boxplot so sánh chất lượng BBox
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df_iou, x='Region', y='IoU_Score', palette="Set1")
    plt.title('So sánh độ khớp Khung hình (IoU)', fontsize=14, fontweight='bold')
    plt.xlabel('Vị trí trong ảnh', fontsize=12)
    plt.ylabel('Chỉ số IoU', fontsize=12)

    plt.tight_layout()
    plt.show()

    # Tính trung bình để in kết luận
    mean_edge_iou = df_iou[df_iou['Region'] == "Vùng biên trái (x < 0.3)"]['IoU_Score'].mean()
    mean_normal_iou = df_iou[df_iou['Region'] == "Vùng bình thường (x >= 0.3)"]['IoU_Score'].mean()

    print("-" * 50)
    print(f"  KẾT QUẢ ĐÁNH GIÁ ĐỘ KHỚP KHUNG HÌNH (IoU):")
    print(f"- Vùng biên trái (x < 0.3): Trung bình IoU = {mean_edge_iou:.2f}")
    print(f"- Vùng bình thường (x >= 0.3): Trung bình IoU = {mean_normal_iou:.2f}")
    
    
# evaluate IOU all model
def evaluate(IMAGES_DIR,LABELS_DIR):
    """_summary_ hàm đánh giá IoU cho toàn bộ tập Validation và vẽ biểu đồ so sánh giữa các mô hình YOLOv11
    """
    MODEL_DICT = {
    # load models
        "s11": r"YOLOv11_training\training_backup_YOLOv11s\weights\best.pt",
        "m11": r"YOLOv11_training\training_backup_YOLOv11m\weights\best.pt",
        "n11": r"YOLOv11_training\training_backup_YOLOv11n\weights\best.pt",
        "s11-optuna": r"YOLOv11_training\model_tuning\weights\best.pt"
    }
    image_paths = glob.glob(os.path.join(IMAGES_DIR, '*.jpg'))
    if not image_paths:
        print("Không tìm thấy thư mục ảnh")
    else:
        print(f"Bắt đầu đánh giá toàn bộ {len(image_paths)}cho 4 mô hình...\n")
        
        all_results = []

        # Duyệt qua từng mô hình
        for model_name, model_path in MODEL_DICT.items():
            if not os.path.exists(model_path):
                print(f"Bỏ qua {model_name} vì không tìm thấy file {model_path}")
                continue
                
            print(f"Đang đánh giá mô hình: {model_name}...")
            model = YOLO(model_path)
            
            # Duyệt qua từng ảnh trong tập Val
            for i, img_path in enumerate(image_paths):
                img_name = os.path.basename(img_path)
                label_path = os.path.join(LABELS_DIR, img_name.replace('.jpg', '.txt'))
                
                # Đọc kích thước ảnh để chuẩn hóa tọa độ
                img = cv2.imread(img_path)
                if img is None: continue
                h_img, w_img, _ = img.shape
                
                # Lấy Ground Truth
                gt_boxes = load_ground_truth(label_path, w_img, h_img)
                if not gt_boxes: 
                    continue # Bỏ qua ảnh nền không có xe
                    
                # Chạy dự đoán
                results = model.predict(source=img_path, conf=0.25, verbose=False)
                pred_boxes = []
                if len(results) > 0 and len(results[0].boxes) > 0:
                    pred_boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
                    
                # Ghép cặp từng GT với Prediction có IoU cao nhất
                for gt in gt_boxes:
                    best_iou = 0
                    for pred in pred_boxes:
                        iou = compute_iou_simple(gt, pred)
                        if iou > best_iou:
                            best_iou = iou
                    
                    # Lưu kết quả
                    all_results.append({
                        'Model': model_name,
                        'IoU': best_iou
                    })
                    
                # In tiến độ cho đỡ chán
                if (i + 1) % 50 == 0 or (i + 1) == len(image_paths):
                    print(f"   Đã xử lý {i + 1}/{len(image_paths)} ảnh...")

        print("\n Đã hoàn tất đối chiếu BBox! Đang vẽ biểu đồ...")

        # Lưu kết quả vào dataframe
        df = pd.DataFrame(all_results)
        
        # Tính mean IoU cho từng model
        mean_ious = df.groupby('Model')['IoU'].mean().reset_index()
        # Sắp xếp theo tên model trong Dict để giữ thứ tự
        model_order = [m for m in MODEL_DICT.keys() if m in df['Model'].unique()]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.patch.set_facecolor('#0f0f0f')
        for ax in axes:
            ax.set_facecolor('#1a1a1a')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            ax.tick_params(colors='#888', labelsize=10)
            ax.xaxis.label.set_color('#aaa')
            ax.yaxis.label.set_color('#aaa')
            ax.title.set_color('#eeeeee')

        COLORS = ['#4FC3F7', '#FFB74D', '#EF5350', '#4CAF50'] # Xanh, Cam, Đỏ, Xanh lá

        # Biểu đồ 1: Barplot So sánh Mean IoU
        ax1 = axes[0]
        sns.barplot(data=mean_ious, x='Model', y='IoU', order=model_order, palette=COLORS, ax=ax1, alpha=0.85)
        ax1.set_title('1. Mean IoU Comparison', fontsize=14, pad=15, fontweight='bold')
        ax1.set_ylabel('Average IoU Score')
        ax1.set_xlabel('YOLOv11 Variants')
        ax1.set_ylim(0, 1.1)
        
        # Ghi số lên đỉnh cột
        for p in ax1.patches:
            ax1.annotate(f"{p.get_height():.4f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='bottom', color='#eeeeee', fontsize=11, fontweight='bold',
                        xytext=(0, 5), textcoords='offset points')

        # Biểu đồ 2: Violin Plot xem phân bố IoU (Xem model nào có nhiều xe bị IoU = 0)
        ax2 = axes[1]
        sns.violinplot(data=df, x='Model', y='IoU', order=model_order, palette=COLORS, ax=ax2, inner="quartile", linewidth=1.5)
        ax2.set_title('2. Distribution of IoU Scores (Violin Plot)', fontsize=14, pad=15, fontweight='bold')
        ax2.set_ylabel('IoU Score')
        ax2.set_xlabel('YOLOv11 Variants')
        
        # Đường chuẩn IoU = 0.5
        ax2.axhline(0.5, color='#FFC107', linewidth=1, linestyle='--', label='Acceptable Threshold (0.5)')
        ax2.legend(facecolor='#222', edgecolor='#444', labelcolor='#eee', loc='lower right')

        plt.tight_layout()
        plt.savefig('result/images/iou_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
        plt.show()

        # In báo cáo text
        print("-" * 50)
        print("BẢNG TỔNG KẾT ĐỘ KHỚP KHUNG HÌNH (Mean IoU):")
        print("-" * 50)
        for index, row in mean_ious.sort_values(by='IoU', ascending=False).iterrows():
            print(f"{row['Model']:<15} : {row['IoU']:.4f}")
# if __name__ == "__main__":
#     x_min, y_min, x_max, y_max = coordinate_bbox(2,3,4,5)
#     print(x_min,y_min,x_max,y_max)


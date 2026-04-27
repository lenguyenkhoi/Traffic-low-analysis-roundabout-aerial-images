# 🚦Traffic Flow Analysis at Roundabout Using Aerial Images

A computer vision project for **vehicle detection, traffic flow analysis, and error analysis** at roundabouts using **YOLOv11** and **Streamlit**.

This system uses aerial images captured from above to detect vehicles such as:

* Car
* Cycle
* Bus
* Truck
* Van

It also provides an interactive web app for prediction, visualization, and model evaluation.

---

## 📌 Project Highlights

✅ Vehicle detection from aerial images
✅ Traffic density analysis
✅ Multi-model comparison (YOLOv11n / s / m / optimized model)
✅ Error Analysis (TP / FP / FN / Localization / Classification / Duplicate)
✅ Interactive Streamlit dashboard
✅ Visualization with charts and bounding boxes

---

## 🧠 Technologies Used

* Python
* YOLOv11 (Ultralytics)
* Streamlit
* OpenCV
* NumPy
* Plotly
* Pandas
* Matplotlib

---

## 📂 Project Structure

```bash
CAPSTONE_FINAL/
│── app.py
│── helpers.py
│
│── 1.prepaired_data.ipynb
│
│── 2.training-yolo.ipynb
│
│── 3.inference.ipynb
│
│── requirements.txt
│── pages/
│   ├── eda.py
│   ├── model.py
│   ├── model_evaluation.py
│   ├── demo_ea.py
│   └── predict.py
│
│── demo/
│
│── result/
│   ├── images/
│   ├── yolov11m/
│   ├── yolov11n/
│   ├── yolov11s/
│   └──yolov11s-optuna/
│  
│── result_vis/
│
│── YOLOv11_training/ 
│   ├── final_model/ 
│   │     └── weights/ 
│   │         └── best.pt 
│   │ 
│   ├── training_backup_YOLOv11n/ 
│   │     └── weights/ 
│   │         └── best.pt 
│   │ 
│   ├── training_backup_YOLOv11s/ 
│   │     └── weights/ 
│   │         └── best.pt 
│   │ 
│   └── training_backup_YOLOv11m/ 
│         └── weights/ 
│             └── best.pt
```
---
## 📦 Dataset

The dataset used in this project is available on Google Drive:

🔗 https://drive.google.com/drive/folders/1OvV7yxzpcx2wrosF-4-71l0JsHZlYDoj

Dataset includes:

* Aerial traffic images
* YOLO format labels
* Multi-class vehicle annotations

Classes:

* Car
* Cycle
* Bus
* Truck
* Van
---

## 🚀 Features

### 1️⃣ Exploratory Data Analysis (EDA)

* Class distribution
* Sample images
* Vehicle statistics
* Dataset insights

### 2️⃣ Model Training & Evaluation

* YOLOv11 Nano / Small / Medium
* Precision / Recall / mAP
* Loss curves
* Confusion Matrix

### 3️⃣ Prediction

Upload an image and run vehicle detection instantly.

### 4️⃣ Error Analysis

Compare:

* Ground Truth
* Prediction
* Error Visualization

Includes:

* True Positive
* False Positive
* False Negative
* Localization Error
* Classification Error
* Duplicate Detection

Visualize and diagnose model mistakes:

| Type                    | Meaning                            |
| ----------------------- | ---------------------------------- |
| 🟢 True Positive        | Correct detection                  |
| 🔴 False Positive       | Wrong detection                    |
| 🔵 False Negative       | Missed object                      |
| 🟡 Localization Error   | Wrong bounding box                 |
| 🟣 Classification Error | Wrong class                        |
| 🟠 Duplicate            | Multiple detections for one object |

👉 This is the **core strength** of the project

---

## ⚙️ Installation

```bash
git clone https://github.com/lenguyenkhoi/Traffic-low-analysis-roundabout-aerial-images.git
cd traffic-roundabout-analysis
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

## 📈 Model Performance

 | Model       | Precision | Recall | mAP50 | mAP50-95 |
 |------------|----------|--------|-------|----------|
 | YOLOv11n   | 0.787    | 0.854  | 0.824 | 0.603    |
 | YOLOv11s   | 0.903    | 0.868  | 0.899 | 0.694    |
 | YOLOv11m   | 0.902    | 0.857  | 0.894 | 0.683    |
 | YOLOv11s-optuna  | 0.882 | 0.920  | 0.943 |0.711  |

---

## 🎯 Future Improvements

* Real-time video traffic monitoring
* Vehicle counting by lane
* Speed estimation
* Congestion prediction
* Deploy cloud inference API

---

## 👨‍💻 Author

**Khoi Le**
Business Data Science Student
Interested in Data Science, AI, and Data Engineering

---

## ⭐ Support

If you find this project useful:

👉 Give it a **star ⭐ on GitHub**
👉 Or fork it for your own research

---

## 🔥 Why This Project Stands Out

* Real-world application (traffic analysis)
* Strong visualization (Streamlit UI)
* Deep evaluation (Error Analysis, not just accuracy)
* Multi-model experimentation

👉 This is not just a model — it's a **complete AI system**
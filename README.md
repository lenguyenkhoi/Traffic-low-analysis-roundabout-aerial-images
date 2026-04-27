# 🚦Traffic Flow Analysis at Roundabout Using Aerial Images

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv11](https://img.shields.io/badge/Ultralytics-YOLOv11-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Optuna](https://img.shields.io/badge/Optuna-Tuning-blueviolet)

## 📌 Project Overview
This project focuses on building an end-to-end computer vision pipeline to detect and analyze traffic flow at roundabouts using aerial imagery. By leveraging **YOLOv11** and hyperparameter tuning via **Optuna**, the system successfully addresses challenges such as severe class imbalance (e.g., abundant cars vs. rare vans) and high-density vehicle overlapping at the edges of the camera frame.

The final model and insights are deployed as an interactive web dashboard using **Streamlit**, allowing users to explore the data, evaluate model performance, and test predictions.

🔗 **Live Dashboard:** [Traffic Flow Analysis App](https://traffic-low-analysis-roundabout-aerial-images.streamlit.app/)

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
│   ├── ea.py
│   ├── real_perform
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
│   ├── model_tuning/ 
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


## ✨ Key Features
- **Exploratory Data Analysis (EDA):** Interactive heatmaps and spatial distribution analysis revealing traffic density patterns.
- **Model Training & Hyperparameter Tuning:** Comparison between YOLOv11 architectures (Nano, Small, Medium) and the application of Optuna to optimize the `YOLOv11s` model, boosting minority class recognition.
- **Robust Evaluation:** Comprehensive test set evaluation including Confusion Matrix, Precision-Recall curves, and spatial IoU analysis (verifying Bounding Box accuracy at dense frame edges).
- **Interactive Error Analysis:** A dedicated module to investigate False Positives and Classification errors, demonstrating critical thinking and model interpretability.
- **Dark Mode UI:** A professional, sleek interface designed for optimal data visualization.

## 📊 Model Performance
The final deployed model is **YOLOv11s-optuna**, which strikes the perfect balance between inference speed and accuracy. 

**Test Set Results:**
- **Overall mAP@0.5:** `0.940`
- **mAP@0.5-0.95:** `0.736`
- **Localization (IoU):** Maintained `> 0.90` average IoU even in highly dense traffic areas (left edge of the frame, $x < 0.3$).
- **Minority Classes:** Significant improvement in `van` and `cycle` detection post-tuning without sacrificing the performance on majority classes like `car`.

## 🛠️ Tech Stack
- **Computer Vision:** Ultralytics YOLOv11, OpenCV
- **Optimization:** Optuna
- **Data Manipulation & Visualization:** Pandas, Matplotlib, Seaborn
- **Web App Framework:** Streamlit
- **Environment:** NVIDIA CUDA, PyTorch

## 🚀 Local Installation & Setup

To run this Streamlit app locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/traffic-flow-analysis-yolo.git](https://github.com/your-username/traffic-flow-analysis-yolo.git)
   cd traffic-flow-analysis-yolo
   ```
2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

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
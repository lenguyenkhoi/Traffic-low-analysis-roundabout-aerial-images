import streamlit  as st


st.set_page_config(
    page_title="Mô hình", 
    page_icon="🤖", 
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


st.title("🤖 Mô hình")
st.markdown("---------")
st.header("Giới thiệu mô hình YOLO ")
st.write("""
        Là một mô hình phát hiện đối tượng (object detection) thời gian
        thực nổi tiếng trong lĩnh vực thị giác máy tính.
        
        Được phát triển lần đầu bởi Joseph Redmon vào năm 2015,
        YOLO nổi bật nhờ khả năng xử lý hình ảnh chỉ một lần qua mạng
        nơ-ron tích chập (CNN), thay vì các phương pháp truyền thống
        phải quét nhiều lần như R-CNN.
        
        YOLO đạt tốc độ cao, phù hợp cho ứng dụng thời gian thực như
        giám sát video, xe tự lái hoặc nhận diện đối tượng trong ảnh.
        
        YOLO đã phát triển qua nhiều phiên bản, với phiên bản mới
        nhất là YOLO26 từ Ultralytics, tập trung vào hiệu suất trên thiết bị
        edge và low-power.
         """)
st.header("Kiến trúc tổng quát của YOLO")

st.write("""
         Một mô hình YOLO tiêu chuẩn luôn được chia thành 4 thành phần chính: Backbone, Neck, Head và Output.
        
        • **Backbone (Xương sống)**:
        - Vai trò: Trích xuất các đặc trưng (features) từ hình ảnh đầu vào.
        - Cơ chế: Sử dụng các mạng tích chập (CNN) sâu để trích xuất từ các đặc trưng bậc thấp (cạnh,
        màu sắc) đến bậc cao (hình dạng, đối tượng).
        - Các module tiêu biểu: CSPNet (v5/v8), C3k2 (v11), hoặc R-ELAN (v12).
        
        • **Neck (Cổ)**:
        - Vai trò: Kết hợp và trộn các đặc trưng trích xuất từ các tầng khác nhau của Backbone.
        - Cơ chế: Sử dụng các cấu trúc như FPN (Feature Pyramid Network) hoặc PAN (Path Aggregation
        Network) để đảm bảo mô hình nhận diện tốt vật thể ở nhiều quy mô (lớn, trung bình, nhỏ).
        - Nâng cấp ở v12: Tích hợp Area Attention để hiểu ngữ cảnh toàn cục giữa các vùng đặc trưng.
        
        • **Head (Đầu dự đoán)**:
        - Vai trò: Đưa ra dự đoán cuối cùng dựa trên các đặc trưng đã được trộn ở phần Neck.
        - Phân loại:
            - Dense Prediction: Dự đoán trên toàn bộ bản đồ đặc trưng.
            - Anchor-free Head: Dự đoán trực tiếp tọa độ mà không cần khung mẫu (YOLOv8-v12).
        
        • **Output (Đầu ra dữ liệu)**: Mỗi dự đoán của YOLO cho một vật thể thường bao gồm một vector có
        dạng:
        
                                    O = [Pobj , x, y, w, h, C1, C2, ..., Cn] (1)
        
        **Trong đó**:
        - Pobj : Xác suất chứa vật thể (Objectness score).
        - x, y, w, h: Tọa độ tâm và kích thước khung hình (Bounding Box).
        - C1, ..., Cn: Xác suất của các lớp đối tượng (Class scores - ví dụ: chó, mèo, xe).
         """)

st.header("Cách hoạt động của mô hình YOLO ")
st.markdown("![YOLO](https://images.viblo.asia/full/963815a3-930b-4f0b-9897-55fd9215458c.PNG)")
st.markdown("------")
st.header("Mô hình YOLO11")
st.markdown("""
            ## Các phiên bản của YOLOv11 (Object Detection)

YOLOv11 cung cấp nhiều kích cỡ mô hình để cân bằng giữa **tốc độ (FPS)** và **độ chính xác (mAP)**. Việc chọn đúng model rất quan trọng, đặc biệt với:
- Object nhỏ
- Mật độ dày (ví dụ: giao thông từ ảnh flycam)
## Bảng so sánh các phiên bản

| Phiên bản | Tên gọi | Đặc điểm | Ứng dụng |
|----------|--------|---------|----------|
| YOLO11n | Nano | Nhẹ nhất, nhanh nhất, độ chính xác thấp | Edge device (Raspberry Pi, Jetson Nano), real-time |
| YOLO11s | Small | Nhẹ, cân bằng tốt hơn Nano | Mobile, camera giám sát |
| YOLO11m | Medium | Cân bằng tốt giữa tốc độ & accuracy | Baseline cho đa số bài toán |
| YOLO11l | Large | Chính xác cao, model lớn | Detect object nhỏ, bị che |
| YOLO11x | Extra Large | Nặng nhất, chính xác nhất | Research, thi AI, server mạnh |

---

## Arguments (Tham số) cốt lõi
### Tham số Huấn luyện (Training)
- `data`: file `.yaml` chứa thông tin dataset (đường dẫn thư mục train/val/test, số lượng class, tên class).
- `epochs`: số vòng train (100–300)  
- `batch`: Kích thước batch (ví dụ: 16, 32, 64). Nếu bị lỗi Out of Memory (OOM), giảm chỉ số này xuống hoặc đặt batch=-1 để mô hình tự động tìm kích thước tối ưu (AutoBatch).
- `imgsz`: Kích thước ảnh đầu vào là 640
- `device`: GPU (`0`) hoặc `cpu`  
- `optimizer`:  Thuật toán tối ưu hóa (SGD, Adam, AdamW). AdamW thường hoạt động rất tốt trên các bài toán phức tạp.
### Data Augmentation 
Giúp mô hình hiảm overfitting và tăng khả năng tổng quát hóa  
### Các tham số chính:
- `mosaic`: Tính năng gộp 4 ảnh thành 1. Mặc định là 1.0 (100% sử dụng). Rất tốt để giúp mô hình làm quen với các khung hình có mật độ đối tượng dày đặc.
- `mixup`: Trộn 2 ảnh đè lên nhau. Tăng độ phức tạp cho dữ liệu.
- `degrees`: Xoay ảnh một góc ngẫu nhiên (ví dụ: 0.0 đến 180.0). Rất quan trọng khi đối tượng có thể xuất hiện ở bất kỳ góc độ nào.
- `scale`:  Phóng to hoặc thu nhỏ ảnh ngẫu nhiên.
### Suy luận & Đánh giá (Inference / Validation)
- `conf`: Ngưỡng tin cậy (Confidence Threshold). Mặc định là 0.25. Bất kỳ bounding box nào có xác suất dự đoán dưới mức này sẽ bị loại bỏ.
- `iou`: Ngưỡng Intersection Over Union cho thuật toán Non-Maximum Suppression (NMS). Mặc định là 0.7. Dùng để triệt tiêu các bounding box trùng lặp cùng trỏ vào một đối tượng.
- `save`: lưu ảnh/video kết quả  

**Nguồn**: 
- https://docs.ultralytics.com/models/yolo11/
- https://docs.ultralytics.com/guides/yolo-data-augmentation/#flip-left-right-fliplr
- https://blog.roboflow.com/yolov11-how-to-train-custom-data/
            
            """)
st.markdown("------")
st.header("Mô hình và huấn luyện")
st.subheader("Chia dữ liệu theo train, val và test")
st.write("""
         Tạo thư mục output và chia dữ liệu theo train, val và test

Dữ liệu được chia thành ba tập độc lập: **train (70%)**, **validation (20%)** và **test (10%)**. Việc phân chia này đóng vai trò quyết định trong việc huấn luyện và đánh giá mô hình Object Detection:

* **Tập train:** Được sử dụng trực tiếp để huấn luyện mô hình học các đặc trưng của đối tượng.
* **Tập validation:** Được dùng để đánh giá hiệu năng thường xuyên trong quá trình huấn luyện, tính toán các chỉ số như mAP, precision, recall để tinh chỉnh siêu tham số (hyperparameters) và lựa chọn checkpoint tốt nhất.
* **Tập test:** Là tập dữ liệu độc lập chưa từng được mô hình "nhìn thấy". Việc sử dụng tập test riêng biệt giúp khắc phục tình trạng đánh giá quá lạc quan (optimistic bias) khi chỉ dùng tập validation, mang lại cái nhìn khách quan nhất về khả năng tổng quát hóa của mô hình trên hệ thống thực tế.
         """)
st.subheader("Chiến lược phân chia dữ liệu")
st.write("""
         
Một bức ảnh có thể chứa nhiều loại đối tượng khác nhau (ví dụ: vừa có car, vừa có xe bus). Vì vậy, đây là bài toán đa nhãn (multi-label), khác với phân loại ảnh thông thường. Một bức ảnh có thể chứa hàng chục ô tô nhưng chỉ có một vài chiếc xe bus hoặc xe van.

Nếu áp dụng phương pháp chia ngẫu nhiên (Random Split) thông thường, các lớp thiểu số có nguy cơ cực cao bị rơi hết vào tập train và không có mặt ở các tập đánh giá. Điều này làm sai lệch quá trình kiểm thử, khiến chỉ số **mAP** của các lớp này không thể được đo lường chính xác.

Để giải quyết triệt để vấn đề mất cân bằng này, áp dụng thuật toán **Chia phân tầng dựa trên nhãn hiếm nhất (Stratified Split by Rarest Class)** với quy trình như sau:

1.  **Thống kê toàn cục (Global Counting):** Quét toàn bộ file nhãn `.txt` để đếm tổng số lượng Bounding Box của từng lớp trong toàn bộ Dataset.
2.  **Gán nhãn đại diện (Representative Labeling):** Duyệt qua từng bức ảnh. Nếu một ảnh chứa nhiều loại phương tiện, thuật toán sẽ chọn lớp có tổng số lượng ít nhất trên toàn dataset làm "nhãn đại diện" cho ảnh đó. *(Ví dụ: Ảnh có 20 Car và 1 Bus -> Nhãn đại diện của ảnh là Bus).*
3.  **Phân tầng 2 giai đoạn (Two-Stage Stratification):** Do hàm `train_test_split` của `scikit-learn` chỉ chia được làm 2 phần, quá trình được thực hiện hai bước kết hợp với tham số `stratify` bằng chính tập "nhãn đại diện":
    * *Bước 1:* Tách tập tổng thành **Train (70%)** và tập tạm (30%).
    * *Bước 2:* Tiếp tục phân tầng tập tạm để tách ra **Validation (20%)** và **Test (10%)**.

**Kết quả:** Phương pháp này ép hệ thống phải phân bổ đều các phương tiện hiếm (Bus, Van) sang cả 3 tập Train, Val, Test theo đúng tỷ lệ định trước. Cùng lúc đó, theo quy luật số lớn (Law of Large Numbers), các phương tiện phổ biến như Car đi kèm trong các bức ảnh đó cũng sẽ tự động được dàn trải đồng đều. Chiến lược này đảm bảo tập Validation và Test là một phiên bản thu nhỏ hoàn hảo của tập Train về mặt phân phối dữ liệu, giúp quy trình đánh giá mô hình mang tính tin cậy cao.
         
         """)
st.subheader("Training setup")
st.write("""
        Lựa chọn kiến trúc: Khởi tạo mô hình YOLOv11 với các phiên bản Nano (N), Small (s), Medium (m) để tìm ra model nào tối ưu nhất
        
        Trong quá trình huấn luyện, tất cả các mô hình YOLOv11 (n, s, m) đều sử dụng cùng một cấu hình tham số nhằm đảm bảo tính công bằng trong quá trình so sánh hiệu năng.

        #### Cấu hình chung
        - **epochs = 100**: Số vòng lặp huấn luyện đủ lớn để mô hình hội tụ, đồng thời tránh overfitting.
        - **imgsz = 640**: Kích thước ảnh đầu vào được chọn nhằm cân bằng giữa tốc độ và độ chính xác, đặc biệt phù hợp với bài toán phát hiện phương tiện nhỏ trong ảnh aerial.
        - **batch = 16**: Kích thước batch phù hợp với tài nguyên GPU, giúp tối ưu tốc độ huấn luyện mà vẫn đảm bảo ổn định gradient.
        - **device = 0**: Sử dụng GPU để tăng tốc quá trình training.
        - **file yaml**: Khai báo chính xác đường dẫn dữ liệu và danh sách các lớp đối tượng (car,cycle,bus,truck,van) theo đúng thứ tự nhãn của bộ dữ liệu gốc.
        - Thành phần của file yml 
            - `path` : Đường dẫn gốc đến thư mục chứa dataset đã được chia (train/val/test). Các đường dẫn còn lại sẽ được tính tương đối từ đây.
            - `train` : Đường dẫn tới thư mục chứa ảnh của tập Train. (Ví dụ: train/images nghĩa là nằm trong path/train/images)
            - `val` : Đường dẫn tới thư mục chứa ảnh của tập Validation.
            - `test` : Đường dẫn tới thư mục chứa ảnh của tập test.
            - `nc` : (number of classes) Tổng số lớp (đối tượng) trong bài toán. Ở đây: 5 lớp.
            - `names` : Danh sách tên các lớp, được ánh xạ theo index:
            0: car
            1: cycle
            2: bus
            3: truck
            4: van
        ####  Data Augmentation
        Các kỹ thuật augmentation được áp dụng nhằm tăng tính đa dạng của dữ liệu và cải thiện khả năng tổng quát hóa của mô hình:
        - **degrees = 15**: Xoay ảnh ngẫu nhiên trong khoảng ±15°, giúp mô hình thích nghi với các góc nhìn khác nhau.
        - **translate = 0.1**: Dịch chuyển ảnh tối đa 10% theo chiều ngang/dọc.
        - **scale = 0.5**: Phóng to/thu nhỏ đối tượng để mô hình học được nhiều kích thước khác nhau.
        - **mosaic = 1.0**: Kết hợp 4 ảnh thành 1, giúp tăng mật độ object và cải thiện khả năng phát hiện trong môi trường đông đúc.
        - **fliplr = 0.0**: Không lật ngang ảnh do đặc thù dữ liệu giao thông (tránh làm sai ngữ cảnh).

        ####  Optimizer & Learning Rate
        - **optimizer = AdamW**: Bộ tối ưu giúp cải thiện khả năng hội tụ và hạn chế overfitting so với Adam truyền thống.
        - **lr0 = 0.001**: Learning rate ban đầu được chọn ở mức phổ biến, đảm bảo quá trình học ổn định.

        ####  Quản lý training
        - **project**: Thư mục lưu kết quả huấn luyện.
        - **name**: Tên experiment để dễ quản lý và so sánh giữa các mô hình.

        ####  Tổng kết
        Việc sử dụng cùng một cấu hình huấn luyện cho tất cả các mô hình giúp đảm bảo rằng sự khác biệt về hiệu năng (Precision, Recall, mAP) đến từ kiến trúc mô hình, thay vì do thay đổi tham số huấn luyện.
        
        Lựa chọn độ phân giải ảnh (Image Size = 640) 
        - Độ phân giải đầu vào được chọn là 640x640. Giá trị mặc định và phổ biến trong YOLO, giúp cân bằng giữa độ chính xác và tốc độ xử lý.
        - Với độ phân giải, mô hình vẫn giữ được đủ chi tiết để phát hiện các đối tượng nhỏ như phương tiện trong ảnh flycam, đồng thời đảm bảo thời gian suy luận không quá cao khi chạy trên CPU.
        - Nếu sử dụng độ phân giải thấp hơn, mô hình có thể bỏ sót các đối tượng nhỏ. Ngược lại, nếu sử dụng độ phân giải cao hơn, thời gian xử lý sẽ tăng đáng kể mà không cải thiện nhiều về hiệu năng.
         """)
st.markdown("----")
st.header("Hyperparameter tuning (Optuna)")
st.write("""
        Để cải thiện hiệu năng của mô hình, phương pháp tối ưu siêu tham số (Hyperparameter Tuning) được áp dụng thông qua Optuna. 
        Mục tiêu là tìm ra bộ tham số tối ưu giúp nâng cao các chỉ số đánh giá như Recall và mAP, đặc biệt trong bối cảnh dữ liệu có nhiều đối tượng nhỏ và phân bố không đồng đều.

        ### Phương pháp
        Optuna sử dụng chiến lược tìm kiếm thông minh (Bayesian Optimization) để thử nghiệm nhiều tổ hợp tham số khác nhau thay vì brute-force (vét cạn) hoặc gridsearch truyền thống. 
        Quá trình này giúp tiết kiệm thời gian và tài nguyên tính toán, đồng thời nhanh chóng hội tụ về vùng tham số tối ưu.

        ### Các tham số được tối ưu
        Trong quá trình tuning, một số siêu tham số quan trọng được lựa chọn để tối ưu:

        - **learning rate (lr0)**: Điều chỉnh tốc độ học của mô hình
        - **batch size**: Ảnh hưởng đến độ ổn định của gradient
        - **optimizer**: Lựa chọn giữa các thuật toán tối ưu khác nhau
        - **augmentation parameters**:
        - scale
        - translate
        - mosaic

        Việc tối ưu các tham số này giúp mô hình học tốt hơn trong các điều kiện dữ liệu phức tạp.
        Tạo ra một sự đánh đổi (trade-off):
        - Mô hình trở nên “nhạy” hơn (phát hiện nhiều object hơn)
        - Nhưng cũng dễ dự đoán nhầm hơn

        Trong bài toán giao thông, việc tăng Recall thường được ưu tiên hơn, vì bỏ sót phương tiện (False Negative) có thể ảnh hưởng lớn đến các ứng dụng thực tế như giám sát giao thông hoặc phân tích mật độ.
        
        Dự án so sánh hiệu suất của 3 mô hình và tìm ra mô hình tốt nhất để Hyperparameter sau đó đánh giá trên mô hình đã được hyperparmeter
        """)
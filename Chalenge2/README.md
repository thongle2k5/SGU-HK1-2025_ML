<pre>
Challenge2/
├── data/
│   ├── 01_raw/            # Dữ liệu gốc, không bao giờ chỉnh sửa file ở đây
│   ├── 02_intermediate/   # Dữ liệu đã qua xử lý trung gian
│   └── 03_processed/      # Dữ liệu cuối cùng, sẵn sàng cho mô hình
│
├── notebooks/
│   ├── 01-eda.ipynb       # Phân tích khám phá dữ liệu (EDA)
│   ├── 02-feature-engineering.ipynb # Kỹ thuật đặc trưng
│   └── 03-modeling.ipynb  # Huấn luyện và đánh giá mô hình
│
├── src/
│   ├── data/              # Scripts để tải hoặc xử lý dữ liệu
│   │   └── make_dataset.py
│   ├── features/          # Scripts để tạo đặc trưng
│   │   └── build_features.py
│   └── models/            # Scripts để huấn luyện hoặc dự đoán
│       ├── train_model.py
│       └── predict_model.py
│
├── models/                # Các mô hình đã huấn luyện được lưu ở đây
│   ├── logistic_regression_v1.pkl
│   └── random_forest_v2.pkl
│
├── reports/
│   └── figures/           # Các biểu đồ, hình ảnh kết quả
│
├── requirements.txt       # Danh sách các thư viện Python cần thiết
└── README.md              # File giới thiệu tổng quan về dự án
</pre>

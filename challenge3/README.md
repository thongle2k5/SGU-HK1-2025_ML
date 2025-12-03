# Dự đoán Thể loại Nhạc (Music Genre Classification) - Kaggle



## 1. Giới thiệu

Dự án này nhằm giải quyết bài toán **phân loại đa lớp (Multi-class Classification)** để xác định thể loại nhạc của một bài hát (gồm 11 thể loại: Pop, Rock, Classical, Jazz, v.v.) dựa trên các đặc trưng âm thanh (Audio Features) từ Spotify và thông tin siêu dữ liệu (Metadata).

Mục tiêu chính là xây dựng quy trình xử lý dữ liệu chuẩn (Pipeline) và mô hình học máy tối ưu để đạt điểm **F1-Macro Score** cao nhất, đồng thời giải quyết các thách thức về dữ liệu nhiễu và mất cân bằng lớp (Imbalanced Data).

---

## 2. Dữ liệu

- **train.csv**: Chứa 14,396 dòng dữ liệu huấn luyện với 17 đặc trưng và nhãn `Class` (0-10).
- **test.csv**: Chứa 3,600 dòng dữ liệu cần dự đoán.
- **Đặc trưng quan trọng**:
    - **Audio Features**: `danceability`, `energy`, `loudness`, `acousticness`, `instrumentalness`, `tempo`, `valence`...
    - **Metadata**: `Artist Name`, `Track Name`, `Popularity`, `duration`.
    - Nguồn dữ liệu: [https://www.kaggle.com/competitions/shai-music-genre-classification](https://www.kaggle.com/competitions/shai-music-genre-classification)
---

## 3. Cấu trúc thư mục

<pre>
challenge3/
├── data/
│   ├── 01_raw/            # Dữ liệu gốc (train.csv, test.csv) - KHÔNG CHỈNH SỬA
│   ├── 02_processed/      # Dữ liệu sạch sau khi Feature Engineering (train_clean.csv...)
│   └── submission.csv     # Kết quả dự đoán cuối cùng để nộp Kaggle
│
├── notebooks/
│   ├── 01-eda.ipynb                 # Phân tích khám phá dữ liệu (EDA) & Trực quan hóa
│   ├── 02-feature-engineering.ipynb # Thử nghiệm các kỹ thuật xử lý đặc trưng
│   └── 03-modeling.ipynb            # Huấn luyện, tối ưu tham số (Optuna) và đánh giá
│
├── src/
│   ├── data/
│   │   └── make_dataset.py   # Script: Pipeline làm sạch & xử lý dữ liệu (Train/Test đồng bộ)
│   └── models/
│       ├── train_model.py    # Script: Huấn luyện model trên toàn bộ dữ liệu & lưu model
│       └── predict_model.py  # Script: Load model & tạo file dự đoán submission.csv
│
├── models/                   # Nơi lưu trữ Model (.pkl) và danh sách Features
│   ├── xgboost_best_optuna.pkl
│   └── model_features.pkl
│
├── requirements.txt          # Danh sách thư viện Python cần thiết
└── README.md                 # Tài liệu hướng dẫn dự án
</pre>

---

## 4. Hướng dẫn cài đặt & Chạy
 
```bash
Bước 1: Cài đặt môi trường
Cài đặt các thư viện cần thiết (khuyến khích dùng môi trường ảo):
pip install -r requirements.txt 

Bước 2: Chuẩn bị dữ liệu
Đặt file train.csv và test.csv vào thư mục: data/01_raw/

Bước 3: Xử lý dữ liệu (Data Pipeline)
Chạy script để làm sạch dữ liệu, xử lý lỗi đơn vị, encoding và scaling:
python src/data/make_dataset.py
Kết quả: File train_clean.csv và test_clean.csv sẽ được tạo tại data/02_processed/.

Bước 4: Huấn luyện mô hình
Chạy script để huấn luyện mô hình XGBoost với tham số tối ưu:
python src/models/train_model.py
Kết quả: Model và danh sách features sẽ được lưu vào thư mục models/.

Bước 5: Tạo dự đoán (Submission)
Chạy script để dự đoán trên tập test:
python src/models/predict_model.py
Kết quả: File submission.csv sẽ được tạo tại thư mục gốc hoặc data/.
```

## 5. Kết quả mô hình
Mô hình tốt nhất: XGBoost Classifier (Tối ưu hóa với Optuna).

Metric đánh giá: F1-Macro Score (do dữ liệu mất cân bằng).

Validation F1-Macro: ~0.47.

Bộ siêu tham số tối ưu (Best Hyperparameters):
```python
{
    'n_estimators': 947,
    'learning_rate': 0.03247,
    'max_depth': 5,
    'min_child_weight': 4,
    'subsample': 0.808,
    'colsample_bytree': 0.784,
    'gamma': 1.464,
    'reg_alpha': 0.0019,
    'reg_lambda': 1.33e-08,
    'objective': 'multi:softmax',
    'tree_method': 'hist'
}
```
Điểm Kaggle: 


<img width="1947" height="293" alt="Screenshot 2025-11-25 003743" src="https://github.com/user-attachments/assets/1af28f4e-a287-4473-8c8e-71339171d1eb" />



## 6. Phân tích & Nhận xét
Khám phá dữ liệu (EDA Insights)
Lỗi dữ liệu nghiêm trọng: Cột duration bị lẫn lộn đơn vị giữa phút và mili-giây. Đã xử lý bằng cách quy đổi đồng nhất về ms.

Feature quan trọng: instrumentalness giúp phân loại rất tốt nhạc Cổ điển (Class 7). Artist Name mang thông tin quan trọng về dòng nhạc.

Mất cân bằng: Class 10 chiếm đa số, gây khó khăn cho việc nhận diện các Class nhỏ (1, 3, 4).

Kỹ thuật Feature Engineering (FE)
Target Encoding: Áp dụng cho Artist Name (thay vì Drop hoặc One-Hot) giúp tăng đáng kể hiệu suất mô hình.

Transformation: Sử dụng np.log1p cho các cột bị lệch (skewed) như speechiness, instrumentalness.

Outlier Handling: Sử dụng kỹ thuật Capping (1%-99% quantile) cho duration và tempo.

Scaling: Sử dụng StandardScaler để đưa dữ liệu về cùng thang đo.

Mô hình hóa (Modeling)
Sử dụng XGBoost kết hợp với Sample Weights để xử lý vấn đề mất cân bằng lớp.

Sử dụng Optuna (Bayesian Optimization) để tìm kiếm siêu tham số hiệu quả hơn GridSearch.

Tuân thủ quy tắc Anti-Leakage: Mọi tính toán thống kê (Mean, Mode, Scaler Fit) đều thực hiện trên tập Train và áp dụng sang tập Test.

---



Ngày hoàn thành: 26/11/2025

Công cụ: Python, Pandas, Scikit-learn, XGBoost, Optuna, Matplotlib

Hoàn thành bởi: Phan Thanh Thịnh

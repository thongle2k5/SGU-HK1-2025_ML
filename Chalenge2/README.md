# Dự đoán Giá Nhà Ames - Kaggle

## Giới thiệu

Dự án này nhằm **dự đoán giá bán nhà** trong thành phố Ames (Iowa, Mỹ) dựa trên các đặc trưng mô tả ngôi nhà như diện tích, số phòng, chất lượng vật liệu, khu vực, v.v.  
Dữ liệu được lấy từ **cuộc thi "House Prices - Advanced Regression Techniques" trên Kaggle**.

Mục tiêu của dự án là xây dựng một mô hình học máy có khả năng dự đoán chính xác giá nhà và đạt **điểm RMSLE thấp nhất có thể trên Kaggle**.

---

## Dữ liệu

- **train.csv**: chứa thông tin 1460 ngôi nhà, gồm đặc trưng và giá bán `SalePrice`.
- **test.csv**: chứa thông tin 1459 ngôi nhà (không có giá bán).
- Nguồn dữ liệu: [https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

## Cấu trúc thư mục:
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

---
## Hướng dẫn cài đặt

### 1. Tạo môi trường ảo (tùy chọn)
Khuyến khích dùng môi trường ảo để tránh xung đột thư viện:
```bash
python -m venv venv
.\venv\Scripts\activate       # Windows
# hoặc
source venv/bin/activate      # Mac/Linux

Cài đặt các thư viện cần thiết
pip install -r requirements.txt

Hướng dẫn chạy (Workflow)
1. Chuẩn bị dữ liệu 
Đặt file train.csv và test.csv vào thư mục:
data/01_raw/

2. Xử lý dữ liệu
Chạy lệnh:
python src/data/make_dataset.py
Kết quả: dữ liệu đã xử lý được lưu vào data/03_processed/

3. Huấn luyện mô hình
Chạy lệnh:
python src/models/train_model.py
Mô hình tốt nhất sẽ được lưu trong thư mục models/ (xgboost_v1.pkl)

4. Tạo dự đoán (submission)
Chạy lệnh:
python src/models/predict_model.py
File dự đoán được tạo tại thư mục gốc với tên submission.csv
```

## Kết quả sau khi chạy:
Mô hình cuối cùng: XGBoostRegressor

Tham số tối ưu (tìm bằng Optuna):
```python
{
    'n_estimators': 795,
    'learning_rate': 0.041781133811210534,
    'max_depth': 4,
    'subsample': 0.9004238636768309,
    'colsample_bytree': 0.8861701514544139,
    'gamma': 0.0007839147453102702,
    'min_child_weight': 3,
    'random_state': 42
}
```
Hiệu suất (CV RMSE): 0.1242.

Điểm Kaggle (RMSLE): 0.1248 (xếp hạng khá tốt so với baseline).


## Phân tích & Nhận xét:
Biến quan trọng nhất: Chất lượng tổng thể (OverallQual), diện tích sống (GrLivArea), tổng diện tích tầng hầm (TotalBsmtSF), và số chỗ đậu xe (GarageCars) là những yếu tố ảnh hưởng lớn nhất đến giá nhà.

Xử lý SalePrice: Sử dụng log-transform (np.log1p) cho biến mục tiêu SalePrice là bước bắt buộc để giảm độ lệch (skewness) của dữ liệu, giúp mô hình hội tụ tốt hơn.

Xử lý dữ liệu thiếu: Áp dụng các chiến lược điền giá trị thiếu (imputation) dựa trên ngữ cảnh (ví dụ: điền 'None' cho Alley, điền median của LotFrontage dựa theo Neighborhood).

Chuẩn hóa dữ liệu: Sử dụng One-hot encoding cho các biến phân loại (categorical) và StandardScaler cho các biến số (numerical).
Tối ưu hóa: Sử dụng Optuna cho việc tinh chỉnh siêu tham số (hyperparameter tuning) hiệu quả hơn so với GridSearchCV/RandomizedSearchCV.

---
Ngày hoàn thành: 30/10/2025

Công cụ: Python, Pandas, Scikit-learn, XGBoost, Optuna, Matplotlib

Hoàn thành bởi: Phan Thanh Thịnh
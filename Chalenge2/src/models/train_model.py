# src/models/train_model.py

import pandas as pd
import xgboost as xgb
import joblib
import os
import sys

# --- Định nghĩa đường dẫn ---
try:
    # Lấy đường dẫn gốc của dự án (đi lên 2 cấp)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Xử lý nếu chạy trực tiếp
    PROJECT_ROOT = os.path.abspath(os.path.join('.', '..', '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src', 'models')):
        PROJECT_ROOT = os.path.abspath('.') # Thử giả định đang chạy từ thư mục gốc

PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "03_processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models") # Thư mục models cấp gốc

TRAIN_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_clean.csv")
MODEL_OUTPUT_PATH = os.path.join(MODELS_DIR, "xgboost_v1.pkl") # Tên file mô hình

# --- Siêu tham số tốt nhất từ Optuna từ notebook ---
BEST_XGB_PARAMS = {
    'n_estimators': 795,
    'learning_rate': 0.041781133811210534,
    'max_depth': 4,
    'subsample': 0.9004238636768309,
    'colsample_bytree': 0.8861701514544139,
    'gamma': 0.0007839147453102702,
    'min_child_weight': 3,
    'random_state': 42
}

def train_model(train_data_path, model_output_path, params):
    """
    Huấn luyện mô hình XGBoost tốt nhất trên toàn bộ dữ liệu train sạch
    và lưu mô hình.
    """
    print("Bắt đầu quá trình huấn luyện mô hình...")

    # 1. Tải dữ liệu train sạch
    print(f"Đang tải dữ liệu huấn luyện từ: {train_data_path}")
    try:
        df_train = pd.read_csv(train_data_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {train_data_path}")
        print(f"   Hãy chắc chắn bạn đã chạy script xử lý dữ liệu trước.")
        sys.exit(1)

    # 2. Chuẩn bị X_train và y_train
    print("   Chuẩn bị dữ liệu X_train, y_train...")
    y_train = df_train['SalePrice'] # Biến mục tiêu (đã log-transform)
    X_train = df_train.drop(['SalePrice', 'Id'], axis=1) # Đặc trưng

    print(f"   Kích thước X_train: {X_train.shape}, y_train: {y_train.shape}")

    # 3. Khởi tạo và huấn luyện mô hình
    print("🏋️ Đang huấn luyện mô hình XGBoost với siêu tham số tốt nhất...")
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    print("   Huấn luyện hoàn tất.")

    # 4. Lưu mô hình
    print(f"Đang lưu mô hình vào: {model_output_path}")
    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    try:
        joblib.dump(model, model_output_path)
        print("Mô hình đã được lưu thành công.")
    except Exception as e:
        print(f"Lỗi khi lưu mô hình: {e}")
        sys.exit(1)

    print("🏁 Quá trình huấn luyện mô hình hoàn tất.")
    return model # Trả về mô hình nếu cần dùng ngay

# --- Chạy hàm huấn luyện khi script được thực thi ---
if __name__ == "__main__":
    trained_model = train_model(TRAIN_CLEAN_PATH, MODEL_OUTPUT_PATH, BEST_XGB_PARAMS)
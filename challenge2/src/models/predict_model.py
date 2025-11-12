# src/models/predict_model.py

import pandas as pd
import numpy as np
import joblib
import os
import sys

# --- Định nghĩa đường dẫn ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    PROJECT_ROOT = os.path.abspath(os.path.join('.', '..', '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src', 'models')):
        PROJECT_ROOT = os.path.abspath('.')

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "01_raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "03_processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

TRAIN_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_clean.csv") # Cần để lấy cột
TEST_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "test_clean.csv")
TEST_RAW_PATH = os.path.join(RAW_DATA_DIR, "test.csv") # Cần để lấy Id gốc
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_v1.pkl") # Mô hình đã huấn luyện
SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "submission.csv") # File output

def make_predictions(model_path, train_clean_path, test_clean_path, test_raw_path, submission_path):
    """
    Tải mô hình, dữ liệu test, tạo dự đoán và lưu file submission.
    """
    print("Bắt đầu quá trình tạo dự đoán...")

    # 1. Tải mô hình đã huấn luyện
    print(f" Đang tải mô hình từ: {model_path}")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file mô hình {model_path}")
        print(f"   Hãy chắc chắn bạn đã chạy script huấn luyện trước.")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tải mô hình: {e}")
        sys.exit(1)

    # 2. Tải dữ liệu train sạch (chỉ để lấy cấu trúc cột) và test sạch
    print(f"Đang tải dữ liệu đã xử lý từ: {PROCESSED_DATA_DIR}")
    try:
        df_train_clean = pd.read_csv(train_clean_path)
        df_test_clean = pd.read_csv(test_clean_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file train_clean hoặc test_clean.")
        sys.exit(1)

    # 3. Chuẩn bị X_test (Đồng bộ cột với X_train)
    print("   Chuẩn bị dữ liệu X_test và đồng bộ cột...")
    try:
        X_train_cols = df_train_clean.drop(['SalePrice', 'Id'], axis=1).columns
        X_test_raw = df_test_clean.drop('Id', axis=1)
        X_test = X_test_raw.reindex(columns=X_train_cols, fill_value=0)
        print(f"   Kích thước X_test sau khi đồng bộ: {X_test.shape}")
    except KeyError as e:
        print(f"Lỗi: Cột {e} không tìm thấy. Kiểm tra lại file _clean.csv.")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi chuẩn bị X_test: {e}")
        sys.exit(1)


    # 4. Thực hiện dự đoán
    print("Đang tạo dự đoán trên tập test...")
    try:
        preds_test_log = model.predict(X_test)
    except Exception as e:
        print(f"Lỗi trong quá trình dự đoán: {e}")
        sys.exit(1)


    # 5. Chuyển đổi ngược kết quả dự đoán
    print("   Chuyển đổi dự đoán về giá trị gốc...")
    preds_test_final = np.expm1(preds_test_log)

    # 6. Tạo file submission
    print(f"Đang tải file test gốc để lấy Id từ: {test_raw_path}")
    try:
        df_test_original = pd.read_csv(test_raw_path)
        print(f"Đang tạo DataFrame submission...")
        submission = pd.DataFrame({
            'Id': df_test_original['Id'],
            'SalePrice': preds_test_final
        })
    except FileNotFoundError:
         print(f"Lỗi: Không tìm thấy file test gốc {test_raw_path}")
         sys.exit(1)
    except Exception as e:
        print(f"Lỗi khi tạo submission DataFrame: {e}")
        sys.exit(1)

    # 7. Lưu file submission
    print(f"Đang lưu file submission vào: {submission_path}")
    try:
        submission.to_csv(submission_path, index=False)
        print("File submission đã được tạo thành công.")
    except Exception as e:
        print(f"Lỗi khi lưu file submission: {e}")
        sys.exit(1)

    print("🏁 Quá trình tạo dự đoán hoàn tất.")
    return submission # Trả về submission df nếu cần

# --- Chạy hàm dự đoán khi script được thực thi ---
if __name__ == "__main__":
    submission_df = make_predictions(MODEL_PATH, TRAIN_CLEAN_PATH, TEST_CLEAN_PATH, TEST_RAW_PATH, SUBMISSION_PATH)
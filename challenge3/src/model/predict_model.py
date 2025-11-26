import pandas as pd
import joblib
import os
import sys

# Lấy đường dẫn tuyệt đối
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Định nghĩa đường dẫn
INPUT_TEST_PATH = os.path.join(project_root, "data", "02_processed", "test_clean.csv")
RAW_TEST_PATH = os.path.join(project_root, "data", "01_raw", "test.csv")
MODEL_PATH = os.path.join(project_root, "models", "xgboost_best_optuna.pkl")
FEATURES_PATH = os.path.join(project_root, "models", "model_features.pkl")
SUBMISSION_PATH = os.path.join(project_root, "data", "submission.csv")

def predict():    
    print("--- BẮT ĐẦU DỰ ĐOÁN ---")
    print(f"Project root: {project_root}")
    print(f"Model path: {MODEL_PATH}")
    
    # 1. Kiểm tra file
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Lỗi: Không tìm thấy Model tại {MODEL_PATH}")
        print("Hãy chạy train_model.py trước.")
        sys.exit(1)
        
    if not os.path.exists(FEATURES_PATH):
        print(f"❌ Lỗi: Không tìm thấy Feature List tại {FEATURES_PATH}")
        print("Hãy chạy train_model.py trước.")
        sys.exit(1)
    
    if not os.path.exists(INPUT_TEST_PATH):
        print(f"❌ Lỗi: Không tìm thấy test data tại {INPUT_TEST_PATH}")
        print("Hãy chạy make_dataset.py trước.")
        sys.exit(1)
        
    # 2. Load Model và Feature List
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print(f"Model loaded with {len(feature_names)} features")
    
    # 3. Load Data Test (Đã clean)
    print(f"Loading test data from {INPUT_TEST_PATH}...")
    df_test = pd.read_csv(INPUT_TEST_PATH)
    print(f"Test data shape: {df_test.shape}")
    
    # 4. Chuẩn bị X_test
    # Lấy Id ra để làm file submission
    if 'Id' in df_test.columns:
        test_ids = df_test['Id']
    else:
        # Fallback: lấy từ file raw nếu file clean lỡ drop mất Id
        print("Id not found in test_clean.csv, loading from raw test.csv...")
        test_ids = pd.read_csv(RAW_TEST_PATH)['Id']

    # Căn chỉnh cột (Reindex): Bước này cực quan trọng để tránh sai lệch
    # Nó sẽ lấy đúng các cột mà Model đã học, theo đúng thứ tự.
    # Nếu thiếu cột -> Tự điền 0. Nếu thừa cột -> Tự bỏ.
    print("Aligning features...")
    X_test = df_test.reindex(columns=feature_names, fill_value=0)
    print(f"X_test shape after alignment: {X_test.shape}")
    
    # 5. Dự đoán
    print("Predicting...")
    predictions = model.predict(X_test)
    
    # 6. Tạo file Submission
    submission = pd.DataFrame({
        'Id': test_ids,
        'Class': predictions
    })
    
    # Lưu fil
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Đã lưu kết quả dự đoán tại: {SUBMISSION_PATH}")
    print("Preview submission:")
    print(submission.head(10))
    print(f"Total predictions: {len(submission)}")
if __name__ == "__main__":
    predict()
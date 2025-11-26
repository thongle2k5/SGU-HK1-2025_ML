import pandas as pd
import xgboost as xgb
import joblib
import os
import sys
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Lấy đường dẫn tuyệt đối
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

# Định nghĩa đường dẫn
INPUT_PATH = os.path.join(project_root, "data", "02_processed", "train_clean.csv")
MODEL_DIR = os.path.join(project_root, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgboost_best_optuna.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "model_features.pkl")

# Bộ tham số tối ưu (Copy từ kết quả Optuna của bạn)
BEST_PARAMS = {
    'n_estimators': 947,
    'learning_rate': 0.032472472171970086,
    'max_depth': 5,
    'min_child_weight': 4,
    'subsample': 0.8080976090915397,
    'colsample_bytree': 0.7845194789596303,
    'gamma': 1.4647765642795072,
    'reg_alpha': 0.0019309717640234123,
    'reg_lambda': 1.3365684096943992e-08,
    'objective': 'multi:softmax',
    'eval_metric': 'mlogloss',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1,
    'num_class': 11
}

def train():
    print("--- BẮT ĐẦU HUẤN LUYỆN MODEL ---")
    print(f"Project root: {project_root}")
    print(f"Input path: {INPUT_PATH}")
    
    # 1. Load dữ liệu
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {INPUT_PATH}")
        sys.exit(1)
        
    print(f"Loading data from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # 2. Chuẩn bị X, y
    # Drop các cột không dùng như trong Notebook
    cols_to_drop = ['Id', 'Artist Name', 'Class']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=existing_drop)
    y = df['Class']
    
    # Lưu danh sách cột để dùng cho predict (QUAN TRỌNG)
    feature_names = X.columns.tolist()
    
    print(f"Shape: X={X.shape}, y={y.shape}")
    
    # 3. Khởi tạo mô hình
    model = xgb.XGBClassifier(**BEST_PARAMS)
    
    # 4. Đánh giá sơ bộ (Cross Validation)
    print("Đang chạy Cross-Validation (5 Folds) để kiểm tra điểm số...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    print(f"F1-Macro trung bình: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # 5. Train Full Data (Production Model)
    print("Đang huấn luyện mô hình trên toàn bộ dữ liệu...")
    model.fit(X, y)
    
    # 6. Lưu Model và Features
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(feature_names, FEATURES_PATH)
    
    print(f"Đã lưu Model tại: {MODEL_PATH}")
    print(f"Đã lưu Feature List tại: {FEATURES_PATH}")
    print("--- HOÀN TẤT HUẤN LUYỆN ---")

if __name__ == "__main__":
    train()
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_data(input_path):
    """Load dữ liệu raw từ thư mục data/01_raw."""
    train_path = os.path.join(input_path, "train.csv")
    test_path = os.path.join(input_path, "test.csv")
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Loaded train: {df_train.shape}, test: {df_test.shape}")
    return df_train, df_test

def clean_basics(df):
    """
    Xử lý cơ bản không phụ thuộc vào thống kê (Stateless transformations).
    Áp dụng giống hệt nhau cho cả Train và Test.
    """
    df = df.copy()
    
    # 1. Sửa lỗi Duration (lẫn lộn phút/ms)
    mask_min = df['duration_in min/ms'] < 30
    df.loc[mask_min, 'duration_in min/ms'] *= 60 * 1000
    df.rename(columns={'duration_in min/ms': 'duration_in_ms'}, inplace=True)
    
    # 2. Sửa lỗi Loudness (Cắt giá trị dương)
    df['loudness'] = df['loudness'].clip(upper=0)
    
    # 3. Log Transform cho các phân phối lệch
    skew_cols = ['speechiness', 'acousticness', 'liveness', 'instrumentalness']
    for col in skew_cols:
        # np.log1p an toàn hơn np.log (tránh log(0))
        df[col] = np.log1p(df[col])
        
    # 4. Xử lý Imputation cố định (instrumentalness = 0)
    df['instrumentalness'] = df['instrumentalness'].fillna(0)
    
    # 5. Drop cột không cần thiết
    # Lưu ý: Giữ lại 'Id' và 'Artist Name' để xử lý sau hoặc dùng cho submission
    df.drop(columns=['Track Name'], inplace=True, errors='ignore')
    
    return df

def process_data(df_train, df_test):
    """
    Xử lý phụ thuộc thống kê (Stateful transformations).
    QUY TẮC VÀNG: Tính toán trên TRAIN -> Áp dụng lên TEST.
    """
    # --- 1. Missing Values (Mean/Mode từ Train) ---
    pop_mean = df_train['Popularity'].mean()
    key_mode = df_train['key'].mode()[0]
    
    # Train
    df_train['Popularity'] = df_train['Popularity'].fillna(pop_mean)
    df_train['key'] = df_train['key'].fillna(key_mode)
    
    # Test (Dùng số của Train!)
    df_test['Popularity'] = df_test['Popularity'].fillna(pop_mean)
    df_test['key'] = df_test['key'].fillna(key_mode)
    
    # --- 2. Outliers Capping (Quantile từ Train) ---
    for col in ['tempo', 'duration_in_ms']:
        lower = df_train[col].quantile(0.01)
        upper = df_train[col].quantile(0.99)
        
        df_train[col] = df_train[col].clip(lower=lower, upper=upper)
        df_test[col] = df_test[col].clip(lower=lower, upper=upper)

    # --- 3. One-Hot Encoding ---
    # Cần xử lý cẩn thận để cột của Test khớp với Train
    cat_cols = ['key', 'mode', 'time_signature']
    
    df_train = pd.get_dummies(df_train, columns=cat_cols, drop_first=True)
    df_test = pd.get_dummies(df_test, columns=cat_cols, drop_first=True)
    
    # Căn chỉnh cột Test theo Train (Thêm cột thiếu bằng 0, bỏ cột thừa)
    df_test = df_test.reindex(columns=df_train.columns, fill_value=0)
    
    # Vì Test không có cột 'Class', reindex sẽ tạo cột 'Class' toàn số 0 -> Cần drop nó
    if 'Class' in df_test.columns:
        df_test.drop(columns=['Class'], inplace=True)
        
    # --- 4. Scaling (StandardScaler) ---
    # Chọn các cột số cần scale
    scale_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_in_ms', 'Popularity']
    
    scaler = StandardScaler()
    
    # Fit trên Train
    df_train[scale_cols] = scaler.fit_transform(df_train[scale_cols])
    
    # Transform trên Test (Không fit lại!)
    df_test[scale_cols] = scaler.transform(df_test[scale_cols])
    
    return df_train, df_test

def main():
    # Lấy đường dẫn tuyệt đối của file hiện tại
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tính đường dẫn đến thư mục challenge3 (đi lên 2 cấp: src -> challenge3)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Định nghĩa đường dẫn
    raw_path = os.path.join(project_root, "data", "01_raw")
    processed_path = os.path.join(project_root, "data", "02_processed")
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(processed_path, exist_ok=True)
    
    print("--- Bắt đầu quy trình Make Dataset ---")
    print(f"Project root: {project_root}")
    print(f"Raw data path: {raw_path}")
    
    # 1. Load Data
    train, test = load_data(raw_path)
    
    # 2. Clean Basics (Logic cứng)
    print("Applying basic cleaning...")
    train = clean_basics(train)
    test = clean_basics(test)
    
    # 3. Process Data (Logic thống kê & Scaling)
    print("Applying statistical processing (Imputation, Outliers, Scaling, Encoding)...")
    train_proc, test_proc = process_data(train, test)
    
    # 4. Save Data
    train_out_path = os.path.join(processed_path, "train_clean.csv")
    test_out_path = os.path.join(processed_path, "test_clean.csv")
    
    train_proc.to_csv(train_out_path, index=False)
    test_proc.to_csv(test_out_path, index=False)
    
    print(f"--- Hoàn tất! ---")
    print(f"Train processed saved to: {train_out_path}")
    print(f"Test processed saved to: {test_out_path}")
    print(f"Final Train shape: {train_proc.shape}")
    print(f"Final Test shape: {test_proc.shape}")

if __name__ == "__main__":
    main()
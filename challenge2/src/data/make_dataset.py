import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# --- Định nghĩa đường dẫn ---
# Giả định script này nằm trong ProjectRoot/src/data/
# và dữ liệu nằm trong ProjectRoot/data/
try:
    # Lấy đường dẫn gốc của dự án (đi lên 2 cấp từ vị trí script)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    # Xử lý nếu chạy trực tiếp (ví dụ: trong môi trường tương tác không có __file__)
    PROJECT_ROOT = os.path.abspath(os.path.join('.', '..', '..'))
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'src', 'data')):
         # Thử giả định đang chạy từ thư mục gốc
         PROJECT_ROOT = os.path.abspath('.')


RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "01_raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "03_processed")

TRAIN_RAW_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_RAW_PATH = os.path.join(RAW_DATA_DIR, "test.csv")

TRAIN_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_clean.csv")
TEST_CLEAN_PATH = os.path.join(PROCESSED_DATA_DIR, "test_clean.csv")

# --- Hàm xử lý ---

def load_data(train_path, test_path):
    """Tải dữ liệu train và test thô."""
    print("Đang tải dữ liệu gốc...")
    try:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)
        print(f"   Train shape: {df_train.shape}, Test shape: {df_test.shape}")
        return df_train, df_test
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {e.filename}")
        print(f"   Đường dẫn dự án đang dùng: {PROJECT_ROOT}")
        sys.exit(1) # Thoát script nếu không tải được dữ liệu

def fill_missing(df_train, df_test):
    """Điền các giá trị thiếu."""
    print("Đang điền giá trị thiếu...")
    # Tạo bản sao để tránh cảnh báo SettingWithCopyWarning
    df_train_filled = df_train.copy()
    df_test_filled = df_test.copy()

    # Điền 'None' cho các cột categorical có NA mang ý nghĩa "Không có"
    cols_none = [
        'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature',
        'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
    ]
    df_train_filled[cols_none] = df_train_filled[cols_none].fillna('None')
    df_test_filled[cols_none] = df_test_filled[cols_none].fillna('None')

    # Điền 0 cho các cột số có NA mang ý nghĩa "Không có" hoặc giá trị 0
    for col in ['MasVnrArea', 'GarageYrBlt']:
        df_train_filled[col] = df_train_filled[col].fillna(0)
        df_test_filled[col] = df_test_filled[col].fillna(0)

    # Điền mode (chỉ tính trên train) cho Electrical
    electrical_mode = df_train_filled['Electrical'].mode()[0]
    df_train_filled['Electrical'] = df_train_filled['Electrical'].fillna(electrical_mode)
    df_test_filled['Electrical'] = df_test_filled['Electrical'].fillna(electrical_mode)

    # Điền median theo Neighborhood cho LotFrontage
    # Tính median trên train và áp dụng cho cả train và test để tránh data leakage nhẹ
    lotfrontage_median_map = df_train_filled.groupby('Neighborhood')['LotFrontage'].median()
    df_train_filled['LotFrontage'] = df_train_filled.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
    # Áp dụng map từ train cho test
    df_test_filled['LotFrontage'] = df_test_filled.apply(
        lambda row: row['LotFrontage'] if pd.notna(row['LotFrontage']) else lotfrontage_median_map.get(row['Neighborhood'], df_train_filled['LotFrontage'].median()),
        axis=1
    )
    # Xử lý nếu vẫn còn NaN (ví dụ: Neighborhood mới trong test hoặc cả khu vực là NaN)
    overall_median_train = df_train_filled['LotFrontage'].median()
    df_train_filled['LotFrontage'] = df_train_filled['LotFrontage'].fillna(overall_median_train)
    df_test_filled['LotFrontage'] = df_test_filled['LotFrontage'].fillna(overall_median_train)


    return df_train_filled, df_test_filled

def encode_and_align(df_train, df_test):
    """Thực hiện One-hot encoding và đồng bộ cột."""
    print("🔄 Đang thực hiện One-hot encoding và đồng bộ cột...")
    # Lưu Id và SalePrice lại
    train_id = df_train['Id']
    test_id = df_test['Id']
    train_target = df_train['SalePrice']

    # Bỏ Id và SalePrice trước khi encoding
    df_train_enc = df_train.drop(['Id', 'SalePrice'], axis=1)
    df_test_enc = df_test.drop('Id', axis=1)

    # One-hot encoding riêng biệt
    df_train_dummies = pd.get_dummies(df_train_enc, drop_first=True)
    df_test_dummies = pd.get_dummies(df_test_enc, drop_first=True)

    # Đồng bộ cột test theo cột train
    train_cols = df_train_dummies.columns
    df_test_aligned = df_test_dummies.reindex(columns=train_cols, fill_value=0)

    # Thêm Id và SalePrice lại
    df_train_aligned = df_train_dummies
    df_train_aligned['Id'] = train_id
    df_train_aligned['SalePrice'] = train_target # Thêm lại SalePrice gốc

    df_test_aligned['Id'] = test_id

    print(f"   Shapes sau encoding/đồng bộ: Train {df_train_aligned.shape}, Test {df_test_aligned.shape}")
    return df_train_aligned, df_test_aligned

def engineer_features(df):
    """Tạo các đặc trưng mới."""
    print("Đang tạo đặc trưng mới...")
    df_eng = df.copy() # Tạo bản sao
    # Tính toán các feature mới
    df_eng['TotalSF'] = df_eng['TotalBsmtSF'] + df_eng['1stFlrSF'] + df_eng['2ndFlrSF']
    df_eng['TotalBathrooms'] = (df_eng['FullBath'] + 0.5 * df_eng['HalfBath'] +
                                df_eng['BsmtFullBath'] + 0.5 * df_eng['BsmtHalfBath'])
    df_eng['Age'] = df_eng['YrSold'] - df_eng['YearBuilt']
    # Đảm bảo Age không âm
    df_eng['Age'] = df_eng['Age'].apply(lambda x: max(x, 0))
    return df_eng

def transform_target(df_train):
    """Log-transform cột SalePrice."""
    print("Đang log-transform SalePrice...")
    df_train_tf = df_train.copy()
    df_train_tf['SalePrice'] = np.log1p(df_train_tf['SalePrice'])
    return df_train_tf

def scale_features(df_train, df_test):
    """Chuẩn hóa các đặc trưng số."""
    print("🔄 Đang chuẩn hóa đặc trưng số...")
    df_train_sc = df_train.copy()
    df_test_sc = df_test.copy()

    # Xác định các cột số (bỏ Id và SalePrice)
    num_features = df_train_sc.select_dtypes(include=np.number).columns
    # errors='ignore' phòng trường hợp cột không tồn tại
    num_features = num_features.drop(['Id', 'SalePrice'], errors='ignore')

    scaler = StandardScaler()

    # Fit scaler chỉ trên train và transform cả train và test
    df_train_sc[num_features] = scaler.fit_transform(df_train_sc[num_features])

    # Đảm bảo chỉ transform các cột số tồn tại trong test
    num_features_in_test = [col for col in num_features if col in df_test_sc.columns]
    df_test_sc[num_features_in_test] = scaler.transform(df_test_sc[num_features_in_test])

    return df_train_sc, df_test_sc # Không cần trả về scaler nếu không dùng ở ngoài

def save_data(df_train, df_test, train_path, test_path):
    """Lưu dữ liệu đã xử lý."""
    print(f"Đang lưu dữ liệu đã xử lý vào {PROCESSED_DATA_DIR}...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    print("Lưu dữ liệu thành công.")

def main():
    """Hàm chính chạy toàn bộ quy trình xử lý dữ liệu."""
    print("Bắt đầu quy trình xử lý dữ liệu...")

    # 1. Tải dữ liệu
    df_train_raw, df_test_raw = load_data(TRAIN_RAW_PATH, TEST_RAW_PATH)

    # 2. Điền giá trị thiếu
    df_train_filled, df_test_filled = fill_missing(df_train_raw, df_test_raw)

    # 3. One-hot encoding và đồng bộ cột
    df_train_enc, df_test_enc = encode_and_align(df_train_filled, df_test_filled)

    # 4. Feature Engineering
    df_train_eng = engineer_features(df_train_enc)
    df_test_eng = engineer_features(df_test_enc) # Áp dụng tương tự cho test

    # 5. Log-transform cột SalePrice (chỉ train)
    df_train_tf = transform_target(df_train_eng)

    # 6. Chuẩn hóa đặc trưng số
    df_train_scaled, df_test_scaled = scale_features(df_train_tf, df_test_eng)

    # 7. Lưu kết quả
    save_data(df_train_scaled, df_test_scaled, TRAIN_CLEAN_PATH, TEST_CLEAN_PATH)

    print("🏁 Quy trình xử lý dữ liệu hoàn tất.")

# --- Chạy hàm main khi script được thực thi ---
if __name__ == "__main__":
    main()
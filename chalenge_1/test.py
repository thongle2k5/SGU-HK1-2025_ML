import pandas as pd
import numpy as np
import re
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# --- 1. DATA INGESTION & COMBINATION ---

# Tải dữ liệu
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Lưu lại PassengerId của tập test để tạo file submission
passenger_id = test['PassengerId']

# Tách cột target 'Survived' và gộp 2 tập dữ liệu lại
y_train = train['Survived']
X_train = train.drop('Survived', axis=1)
all_data = pd.concat([X_train, test], ignore_index=True)

# Ghi chú: Sử dụng 'all_data' cho tất cả các bước tiền xử lý để đảm bảo tính nhất quán

print("Data loaded and combined successfully. Shape:", all_data.shape)

# --- 2. FEATURE ENGINEERING & DATA CLEANING ---

# 2.1. Feature Engineering: Title (Chức danh)
def get_title(name):
    """Trích xuất Title (Mr, Miss, Master, v.v.) từ cột Name."""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

all_data['Title'] = all_data['Name'].apply(get_title)

# Nhóm các Title hiếm (Rare)
all_data['Title'] = all_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
all_data['Title'] = all_data['Title'].replace('Mlle', 'Miss')
all_data['Title'] = all_data['Title'].replace('Ms', 'Miss')
all_data['Title'] = all_data['Title'].replace('Mme', 'Mrs')

# 2.2. Feature Engineering: FamilySize
all_data['FamilySize'] = all_data['SibSp'] + all_data['Parch'] + 1
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1

# 2.3. Imputation (Điền giá trị thiếu)
# Age: Điền bằng median dựa trên Title (thực hiện trên all_data)
# Đây là một phương pháp imputation phức tạp hơn so với chỉ dùng median của toàn bộ cột
all_data['Age'] = all_data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

# Embarked: Điền bằng mode (giá trị xuất hiện nhiều nhất)
all_data['Embarked'] = all_data['Embarked'].fillna(all_data['Embarked'].mode()[0])

# Fare: Điền bằng median (thiếu 1 giá trị trong tập test)
all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median())

# Cabin: Biến thành boong (chữ cái đầu tiên) và điền 'Missing' cho NaN
all_data['Cabin'] = all_data['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Missing')

# 2.4. Loại bỏ các cột không cần thiết
all_data = all_data.drop(['Name', 'Ticket'], axis=1)

print("Feature Engineering and Imputation completed.")

# --- 3. ENCODING CATEGORICAL FEATURES ---

# Mã hóa (One-Hot Encoding) các biến phân loại
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'Cabin']
all_data = pd.get_dummies(all_data, columns=categorical_features, drop_first=True)

# Loại bỏ PassengerId (chỉ cần thiết cho file submission)
all_data = all_data.drop('PassengerId', axis=1)

print("Categorical features encoded.")

# --- 4. DATA SPLITTING & MODEL TRAINING ---

# Tách lại Train và Test đã được tiền xử lý
X_train_processed = all_data.iloc[:len(train)]
X_test_processed = all_data.iloc[len(train):]

# 4.1. Khởi tạo mô hình (Sử dụng XGBoost - mạnh mẽ và phổ biến)
# Bạn có thể thử các tham số khác để tối ưu hơn
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 4.2. Huấn luyện mô hình
print("Start training XGBoost model...")
model.fit(X_train_processed, y_train)
print("Model trained successfully.")

# --- 5. PREDICTION & SUBMISSION ---

# 5.1. Dự đoán trên tập Test
predictions = model.predict(X_test_processed)

# 5.2. Tạo file Submission
submission = pd.DataFrame({
    'PassengerId': passenger_id,
    'Survived': predictions.astype(int) # Đảm bảo Survived là kiểu int (0 hoặc 1)
})

submission.to_csv('submission.csv', index=False)

print("\n--- RESULTS ---")
print("Submission file 'submission.csv' created successfully.")
print("You can now upload 'submission.csv' to Kaggle.")
print(f"Number of Survived predictions: {submission['Survived'].sum()}")
print(f"Number of Died predictions: {len(submission) - submission['Survived'].sum()}")
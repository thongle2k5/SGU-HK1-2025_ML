# Nhập môn máy học (Introduction to Machine Learning)

Chào mừng đến với repository chính thức của **Nhóm 7**. Đây là nơi lưu trữ toàn bộ mã nguồn, báo cáo thực hành và các dự án (Challenges) được thực hiện trong môn học **Nhập môn Máy học**, dưới sự hướng dẫn của **Giảng viên Đỗ Như Tài**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-green)
![Course](https://img.shields.io/badge/Course-Machine_Learning-orange)


## Thành viên nhóm 7:
- 3123410360 - Phan Thanh Thịnh (Nhóm trưởng)
- 3123410362 - Lê Văn Thông
- 3123410363 - Võ Hoàng Thông

---
## Cấu trúc Repository

Repository này được tổ chức theo từng bài Lab và các thử thách (Challenge) của môn học:

### Các Dự án Chính (Challenges)

| Thư mục | Tên dự án | Mô tả & Công nghệ | Trạng thái |
|:---|:---|:---|:---:|
| **[`challenge1/`](./challenge1)** | **Phân loại SVM** | Ứng dụng thuật toán Support Vector Machine (SVM) để phân loại dữ liệu. | Đã hoàn thành |
| **[`challenge2/`](./challenge2)** | **House Price Prediction** | Dự đoán giá nhà (Regression).  <br> *Công nghệ: XGBoost, Optuna, Feature Engineering nâng cao.*  | Đã hoàn thành |
| **[`challenge3/`](./challenge3)** | **Music Genre Classification** | Phân loại thể loại nhạc (Multi-class Classification). <br> *Công nghệ: XGBoost, Optuna, Feature Engineering nâng cao.* | Đã hoàn thành |

### Bài tập Thực hành (Labs & Assignments)

* **[`Lab05/`](./Lab05)**: Các bài thực hành nền tảng về Tiền xử lý dữ liệu và Hồi quy tuyến tính.
* **[`Lab07/`](./Lab07)**: Báo cáo thực hành và bài tập mở rộng (File báo cáo 2.3).
* **[`Lab08/k-nearest`](./Lab08/k-nearest)**: Triển khai thuật toán K-Nearest Neighbors (KNN) từ cơ bản đến nâng cao.
* **[`PimaIndiansDiabetes/`](./PimaIndiansDiabetes)**: Phân tích và xây dựng mô hình dự đoán tiểu đường trên tập dữ liệu Pima Indians.

---

## Công nghệ sử dụng

Dự án sử dụng ngôn ngữ **Python** và các thư viện Data Science phổ biến:

* **Xử lý dữ liệu:** `Pandas`, `NumPy`
* **Trực quan hóa:** `Matplotlib`, `Seaborn`
* **Machine Learning:** `Scikit-learn`
* **Advanced Models:** `XGBoost`, `CatBoost`
* **Optimization:** `Optuna`
* **Tools:** `Jupyter Notebook`, `VS Code`

---

## Hướng dẫn chạy

Để chạy các dự án trong repository này, vui lòng cài đặt các thư viện cần thiết:

```bash
# Clone repository
git clone [https://github.com/username/ten-repo.git](https://github.com/username/ten-repo.git)

# Di chuyển vào thư mục
cd ten-repo

# Cài đặt thư viện (nếu có file requirements.txt ở từng challenge)
pip install -r requirements.txt

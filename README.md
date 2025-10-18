## Cấu trúc thư mục dự án

```bash
Tên_Du_An_ML/
│
├── data/
│   ├── 01_raw/          # Dữ liệu gốc, không thay đổi (immutable)
│   ├── 02_interim/      # Dữ liệu đã làm sạch, xử lý trung gian
│   └── 03_processed/    # Dữ liệu đã sẵn sàng để train (chia train/test/validation)
│
├── notebooks/           # Các tệp Jupyter Notebooks (khám phá dữ liệu, thử nghiệm nhanh)
│
├── src/                 # Source Code (được module hóa, sạch sẽ)
│   ├── __init__.py      # Giúp Python nhận dạng đây là một package
│   ├── data_prep.py     # Script xử lý và tiền xử lý dữ liệu
│   ├── models.py        # Định nghĩa kiến trúc mô hình
│   ├── train.py         # Logic huấn luyện mô hình chính
│   └── utils.py         # Các hàm helper, tiện ích chung
│
├── models/              # Lưu trữ mô hình đã train
│   └── my_best_model.pkl  # (hoặc .h5, .pth)
│
├── experiments/         # Lưu trữ kết quả, logs, metrics
│
├── config/              # Tệp cấu hình (YAML, JSON)
│
├── .gitignore           # Bỏ qua tệp/thư mục không cần đẩy lên Git
├── requirements.txt     # Danh sách thư viện Python cần thiết
└── README.md            # Mô tả dự án, hướng dẫn cài đặt và chạy


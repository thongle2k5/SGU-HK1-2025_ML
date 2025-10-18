Tên_Du_An_ML/
|
├── data/
│ ├── 01_raw/ # Dữ liệu gốc, không thay đổi (immutable)
│ ├── 02_interim/ # Dữ liệu đã làm sạch, xử lý trung gian
│ └── 03_processed/ # Dữ liệu đã sẵn sàng để train (chia train/test/validation)
|
├── notebooks/ # Các tệp Jupyter Notebooks (dành cho khám phá dữ liệu, thử nghiệm nhanh)
|
├── src/ # Source Code (Code đã được module hóa, sạch sẽ)
│ ├── **init**.py # Giúp Python nhận dạng đây là một package
│ ├── data_prep.py # Script xử lý và tiền xử lý dữ liệu
│ ├── models.py # Định nghĩa kiến trúc mô hình (ví dụ: lớp Neural Network)
│ ├── train.py # Script chứa logic huấn luyện mô hình chính
│ └── utils.py # Các hàm helper, tiện ích chung
|
├── models/ # Nơi lưu trữ mô hình đã train (checkpoints, final models)
│ └── my_best_model.pkl (hoặc .h5, .pth)
|
├── experiments/ # Lưu trữ kết quả, logs, metrics của các lần chạy thử nghiệm
|
├── config/ # Các tệp cấu hình (YAML, JSON) cho siêu tham số, đường dẫn
|
├── .gitignore # Danh sách các tệp/thư mục không muốn đẩy lên Git (ví dụ: data/raw, models/)
├── requirements.txt # Danh sách các thư viện Python cần thiết (để tái tạo môi trường)
└── README.md # Mô tả dự án, hướng dẫn cài đặt và chạy

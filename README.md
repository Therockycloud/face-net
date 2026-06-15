# 📊 Ứng Dụng Nhận Diện Khuôn Mặt Thành Viên Nhóm

Ứng dụng nhận diện khuôn mặt cục bộ hiệu năng cao sử dụng thư viện **Streamlit** làm giao diện điều khiển và **FaceNet (InceptionResnetV1)** từ `facenet-pytorch` làm bộ trích xuất đặc trưng (embedding 512 chiều). 

Hệ thống hoạt động theo cơ chế **Face Embedding** (nhận diện qua so sánh khoảng cách đặc trưng vector) nên không cần huấn luyện lại mô hình khi thêm người mới, hỗ trợ đăng ký nhiều góc mặt để tăng độ chính xác và bảo mật.

---

## 🌟 Tính năng nổi bật

- **Nhận diện trực quan**: Cho phép chụp trực tiếp từ Webcam hoặc tải tệp ảnh lên. Tự động vẽ khung bounding box và gán nhãn tên kèm phần trăm độ tin cậy.
- **Quy trình đăng ký 5 góc mặt**: Dẫn dắt người dùng chụp 5 góc mặt mẫu qua camera (Nhìn thẳng, Quay trái, Quay phải, Ngẩng lên, Cúi xuống) để nhận diện nhạy hơn ở mọi tư thế.
- **Tải tệp hàng loạt**: Cho phép kéo thả cùng lúc nhiều ảnh mẫu của thành viên. Hệ thống tự động xác minh từng ảnh chứa đúng 1 khuôn mặt trước khi lưu.
- **Giao diện Light Mode Cyan/Green**: Tông màu Cyan chủ đạo và Green dịu mắt, tối giản và cực kỳ hiện đại.
- **Quản lý dữ liệu trực tiếp**: Xem thư viện ảnh đã lưu của từng thành viên và xóa dữ liệu trực quan ngay trên giao diện.

---

## 📂 Cơ sở dữ liệu Cục bộ
Dữ liệu khuôn mặt được tổ chức đơn giản và an toàn trong thư mục máy tính của bạn:
```text
data/
└── registered_faces/
    ├── NguyenVanA/
    │   ├── front.png
    │   ├── left.png
    │   └── right.png
    └── TranThiB/
        └── 1718432380789.png
```
> [!NOTE]
> Để tăng độ nhạy nhận diện khi camera đặt lệch góc, hãy đăng ký đủ các góc mặt (thẳng, trái, phải, lên, xuống) của thành viên.

---

## 🚀 Hướng Dẫn Chạy & Cài Đặt

### 1. Chạy trực tiếp bằng Python / Conda

Yêu cầu môi trường có cài đặt Python 3.12+ (khuyên dùng Miniforge hoặc Anaconda).

* **Cài đặt thư viện**:
  ```bash
  pip install -r requirements.txt
  ```

* **Khởi động ứng dụng**:
  ```bash
  streamlit run app.py
  ```
  *(Ứng dụng sẽ tự động mở trên trình duyệt tại: `http://localhost:8501`)*

* **Chạy bộ kiểm thử (pytest)**:
  ```bash
  pytest test_face_engine.py -v
  ```

---

### 2. Sử dụng Docker (Khuyên dùng cho triển khai nhanh)

Docker giúp đóng gói và chạy ứng dụng mà không cần cấu hình Python hay cài đặt thư viện trên máy của bạn.

* **Build Docker Image**:
  ```bash
  docker build -t face-rec-app .
  ```

* **Khởi động Container**:
  Gắn thư mục dữ liệu `data` ra ngoài máy vật lý để lưu trữ dữ liệu vĩnh viễn:
  ```bash
  docker run -d -p 8501:8501 -v "$(pwd)/data:/app/data" --name face-rec-container face-rec-app
  ```
  *(Truy cập giao diện tại: `http://localhost:8501`)*

* **Tạm dừng / Dừng Container**:
  Khi không sử dụng nữa, bạn có thể tạm dừng container:
  ```bash
  docker stop face-rec-container
  ```

* **Chạy lại Container**:
  Để khởi chạy lại container đã dừng trước đó:
  ```bash
  docker start face-rec-container
  ```

* **Xóa Container**:
  Nếu bạn muốn gỡ bỏ hoàn toàn container (không ảnh hưởng đến ảnh mẫu đã lưu trong thư mục `data` cục bộ):
  ```bash
  docker stop face-rec-container && docker rm face-rec-container
  ```

---

> [!WARNING]
> **Độ bảo mật**: Hệ thống đã được tích hợp bộ lọc chống tấn công ghi đè và duyệt ngược thư mục (Path Traversal) qua tên đăng ký. Tuy nhiên, hãy đảm bảo rằng tên thành viên chỉ chứa các ký tự chữ cái không dấu, số, hoặc dấu gạch ngang/gạch dưới để hệ thống hoạt động ổn định nhất.

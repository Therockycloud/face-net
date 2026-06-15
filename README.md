# 📸 Ứng Dụng Nhận Diện Khuôn Mặt Thành Viên Nhóm (Streamlit + FaceNet)

Ứng dụng nhận diện khuôn mặt cục bộ sử dụng mô hình pre-trained **InceptionResnetV1** (FaceNet) từ thư viện `facenet-pytorch` để trích xuất đặc trưng (embedding 512 chiều) và so khớp độ tương đồng Cosine. Hệ thống hoạt động theo cơ chế Face Embedding (nhận diện nhanh mà không cần huấn luyện lại từ đầu) và hỗ trợ lưu trữ nhiều ảnh mẫu (nhiều góc mặt) cho mỗi thành viên.

Giao diện đồ họa được thiết kế bằng **Streamlit** với tông màu Light Mode (Cyan/Green) sạch sẽ, hiện đại.

---

## 🛠️ Tính năng chính
1. **🔍 Nhận Diện Face**: Cho phép tải ảnh lên (`st.file_uploader`) hoặc chụp trực tiếp qua Webcam (`st.camera_input`). Nhận diện và vẽ khung bao (bounding box) kèm tên thành viên và phần trăm độ tin cậy.
2. **👥 Danh Sách Thành Viên**: Hiển thị lưới hình ảnh của các thành viên đã lưu trong cơ sở dữ liệu và cho phép xoá thành viên.
3. **➕ Đăng Ký Mới**: Nhập tên thành viên mới, chụp ảnh webcam hoặc tải ảnh chân dung lên. Hệ thống sẽ tự động kiểm tra xem ảnh có chứa **đúng 1 khuôn mặt** hay không trước khi đồng ý lưu.

---

## 📂 Cấu trúc Cơ sở dữ liệu Cục bộ
Dữ liệu thành viên được lưu trực tiếp dưới dạng thư mục trong máy của bạn:
```text
data/
└── registered_faces/
    ├── NguyenVanA/
    │   ├── 1718432345123.png
    │   └── 1718432360256.png
    └── TranThiB/
        └── 1718432380789.png
```
*Bạn có thể lưu nhiều ảnh của một người (trực diện, nghiêng trái, nghiêng phải, nhìn lên/xuống) vào thư mục của họ để tăng độ nhạy nhận diện.*

---

## 🚀 Cách Chạy Ứng Dụng

### Cách 1: Chạy trực tiếp bằng Python/Conda (Khuyên dùng khi Dev)

1. **Cài đặt thư viện**:
   Chạy lệnh sau để cài đặt các package trong file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Chạy ứng dụng Streamlit**:
   Chạy lệnh sau tại thư mục gốc của dự án:
   ```bash
   streamlit run app.py
   ```
   *Streamlit sẽ tự động mở trình duyệt web tại địa chỉ: `http://localhost:8501`*

3. **Chạy kiểm thử (Unit Tests)**:
   Để chạy bộ test kiểm tra độ chính xác và bảo mật của FaceEngine:
   ```bash
   pytest test_face_engine.py -v
   ```

---

### Cách 2: Chạy bằng Docker (Tiện lợi, cô lập môi trường)

Docker giúp bạn chạy ứng dụng mà không cần cài đặt Python, PyTorch hay các thư viện bổ trợ trên máy chủ.

1. **Build Docker Image**:
   Mở terminal tại thư mục dự án và build image với tên `face-rec-app`:
   ```bash
   docker build -t face-rec-app .
   ```

2. **Chạy Docker Container**:
   Để lưu trữ vĩnh viễn dữ liệu khuôn mặt (không bị mất khi dừng container), bạn cần gắn ổ đĩa (`volume`) từ thư mục `data` trên máy chủ vào container:
   ```bash
   docker run -d -p 8501:8501 -v "$(pwd)/data:/app/data" --name face-rec-container face-rec-app
   ```
   - `-p 8501:8501`: Mở cổng 8501 ra ngoài.
   - `-v "$(pwd)/data:/app/data"`: Gắn thư mục `data` cục bộ vào container để lưu trữ lâu dài.
   - Truy cập giao diện tại: `http://localhost:8501`

---

## 💡 Lưu ý khi sử dụng bộ ảnh mẫu
Để hệ thống nhận diện tốt nhất:
- **Đăng ký**: Khi đăng ký thành viên mới, hãy chọn ảnh **trực diện, đủ sáng, rõ mặt** làm ảnh đầu tiên.
- **Tăng độ nhạy**: Bạn có thể vào tab **Đăng Ký Mới**, gõ lại đúng tên thành viên cũ và chụp/upload thêm ảnh góc nghiêng của họ. Hệ thống sẽ tự động lưu thêm ảnh mẫu vào thư mục của họ mà không ghi đè ảnh cũ.

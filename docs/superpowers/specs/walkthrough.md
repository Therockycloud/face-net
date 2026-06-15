# Walkthrough: Streamlit Face Recognition Dashboard with Guided 5-Angle Registration

This walkthrough documents the completed implementation of the Face Recognition Dashboard using Streamlit and `facenet-pytorch`.

## Changes Made
We developed a complete directory-based Face Recognition system with a Light Mode (Cyan/Green) web dashboard. The key files added and updated are:

1. **[requirements.txt](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/FaceRecognition/requirements.txt)**:
   - Configures dependencies: `facenet-pytorch`, `streamlit`, `opencv-python`, and `pytest`.

2. **[face_engine.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/FaceRecognition/face_engine.py)**:
   - Contains the core backend logic in `FaceRecognitionEngine`.
   - **Optimization**: Combined face counting and cropping inside `detect_face` to return a 3-tuple `(cropped_face_img, bbox, face_count)`. This runs the MTCNN deep learning model only **once** per photo (halving the processing latency).
   - Computes 512-dimensional face embedding vectors using a pre-trained `InceptionResnetV1` (on `vggface2`).
   - Runs fast, O(1) database registration and deletion with incremental cache updates in RAM (avoiding full database reloads).
   - Computes cosine similarity scores, handling full `[-1.0, 1.0]` ranges safely.
   - Includes robust directory security (preventing path traversal attacks via `Path(name).name.strip()` and explicit folder name validation).
   - Automatically handles image mode conversions (`.convert('RGB')`) to prevent model channel crashes.
   - Supports saving images with custom semantic filenames (e.g. `front.png`, `left.png`).

3. **[test_face_engine.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/FaceRecognition/test_face_engine.py)**:
   - A complete pytest suite utilizing the native `tmp_path` fixture for safe, concurrent, and isolated TDD.
   - Tests directory creation, face detection edge cases, embedding dimensionalities, cosine similarities, cache loading, incremental registration/deletion, path traversal security, and face counting.

4. **[.streamlit/config.toml](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/FaceRecognition/.streamlit/config.toml)**:
   - Configures the Streamlit server settings and applies a custom Light Mode theme (Cyan main colors `#0891B2` / `#ECFEFF` and Green success accents `#22C55E`).

5. **[app.py](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/FaceRecognition/app.py)**:
   - Implements the complete Streamlit UI with three tabs (no emojis used for a clean, professional aesthetic):
     - **Nhận Diện Face Tab**: Webcam snapshots and file uploads, drawing green bounding boxes and names directly on images (with similarity scores clamped at 0.0% to avoid negative numbers).
     - **Danh Sách Thành Viên Tab**: Displays grid cards of registered members and their reference images (under a safe `with` context manager to prevent file descriptor leaks), alongside a secure Delete button that triggers persistent success notifications across reruns using Streamlit session state.
     - **Đăng Ký Mới Tab**:
       - **Guided Webcam Capture**: Dues the user through a 5-step capture flow (Chính giữa, Nghiêng Trái, Nghiêng Phải, Ngẩng Lên, Cúi Xuống) with on-screen guidance and a progress bar, validating that exactly 1 face is present on each capture.
       - **Multi-File Upload**: Supports drag-and-drop of multiple files, validating exactly 1 face is present on each, saving all valid ones, and automatically clearing the widget selection upon success.
       - **Input Validation**: Uses regex `^[a-zA-Z0-9_ -]+$` to sanitize name inputs.

6. **[Dockerfile](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/FaceRecognition/Dockerfile)**:
   - Packages the app, exposing port `8501`.

7. **[README.md](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/FaceRecognition/README.md)**:
   - Premium guide in Vietnamese with Docker container controls (start/stop/remove) and git push instructions.

---

## What Was Tested & Validation Results
We ran the pytest test suite in the active Conda environment:
```bash
/opt/homebrew/Caskroom/miniforge/base/bin/pytest test_face_engine.py -v
```
All 7 tests passed successfully:
1. `test_engine_initialization_creates_db_dir`: Verified directory creation.
2. `test_detect_face_returns_none_for_blank_image`: Verified detection failures handle empty/blank inputs gracefully.
3. `test_get_embedding_returns_512_dimensional_tensor`: Verified FaceNet model output shapes and CPU residency.
4. `test_cosine_similarity`: Verified cosine similarity range calculations.
5. `test_register_and_load_and_delete_member`: Verified database caching, file outputs, and matching predictions.
6. `test_path_traversal_protection`: Verified path traversal validation blocks and sanitizes dangerous input names.
7. `test_count_faces`: Verified face counting logic for a blank image.

---

## How to Run the Application (Docker)
1. **Dừng container cũ (nếu có)**:
   ```bash
   docker stop face-rec-container && docker rm face-rec-container
   ```
2. **Khởi động container mới với tính năng 5 góc mặt**:
   ```bash
   docker run -d -p 8501:8501 -v "$(pwd)/data:/app/data" --name face-rec-container face-rec-app
   ```
3. **Mở giao diện**: Truy cập **[http://localhost:8501](http://localhost:8501)**.

import streamlit as st
from PIL import Image, ImageDraw
import re
from face_engine import FaceRecognitionEngine

# Title and Layout Setup
st.set_page_config(page_title="Nhận Diện Khuôn Mặt Nhóm", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        color: #0891B2;
        font-family: 'Fira Sans', sans-serif;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">NHẬN DIỆN KHUÔN MẶT THÀNH VIÊN</h1>', unsafe_allow_html=True)

# Display persistent success messages if they exist in session state
if 'success_message' in st.session_state:
    st.success(st.session_state.success_message)
    del st.session_state.success_message

# Initialize Engine
if 'engine' not in st.session_state:
    with st.spinner("Đang tải mô hình FaceNet (InceptionResnetV1)..."):
        st.session_state.engine = FaceRecognitionEngine(db_dir="./data/registered_faces")

engine = st.session_state.engine

# Tabs
tab1, tab2, tab3 = st.tabs(["Nhận Diện Face", "Danh Sách Thành Viên", "Đăng Ký Mới"])

# ==========================================
# TAB 1: RECOGNITION
# ==========================================
with tab1:
    st.header("Nhận Diện Khuôn Mặt")
    
    source = st.radio("Chọn nguồn ảnh:", ["Webcam Chụp Trực Tiếp", "Upload File Ảnh từ Máy Tính"])
    
    img_file = None
    if source == "Webcam Chụp Trực Tiếp":
        img_file = st.camera_input("Chụp ảnh khuôn mặt của bạn")
    else:
        img_file = st.file_uploader("Chọn ảnh (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
        
    if img_file is not None:
        try:
            pil_img = Image.open(img_file).convert("RGB")
            st.subheader("Kết quả phân tích:")
            
            with st.spinner("Đang phát hiện và nhận diện khuôn mặt..."):
                # Detect face
                cropped_face, bbox, face_count = engine.detect_face(pil_img)
                
                if cropped_face is not None:
                    # Recognize face
                    name, score = engine.recognize_face(cropped_face, threshold=0.60)
                    display_score = max(0.0, score)
                    
                    # Draw bbox on image
                    draw_img = pil_img.copy()
                    draw = ImageDraw.Draw(draw_img)
                    x1, y1, x2, y2 = bbox
                    
                    # Draw thick green box
                    draw.rectangle([x1, y1, x2, y2], outline="#22C55E", width=4)
                    
                    # Draw label text background
                    label = f"{name} ({display_score:.2f})"
                    draw.text((x1, max(0, y1 - 20)), label, fill="#22C55E")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(draw_img, caption="Ảnh Đã Nhận Diện", use_column_width=True)
                    with col2:
                        st.image(cropped_face, caption="Mặt Cắt Được (MTCNN)", width=160)
                        if name != "Unknown":
                            st.success(f"Chào mừng: **{name}** (Độ tin cậy: {display_score:.2%})")
                        else:
                            st.warning(f"Khuôn mặt lạ hoặc chưa được đăng ký (Độ tin cậy cao nhất: {display_score:.2%})")
                else:
                    st.error("Không tìm thấy khuôn mặt nào trong ảnh. Vui lòng chụp/chọn ảnh rõ mặt hơn.")
        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {e}")

# ==========================================
# TAB 2: GALLERY
# ==========================================
with tab2:
    st.header("Quản Lý Thành Viên Nhóm")
    
    # Reload button
    if st.button("Tải lại Danh Sách"):
        engine.load_db()
        st.rerun()
        
    # Check if empty
    db_path = engine.db_dir
    if not db_path.exists():
        members = []
    else:
        members = [d.name for d in db_path.iterdir() if d.is_dir()]
    
    if len(members) == 0:
        st.info("Chưa có thành viên nào được đăng ký trong hệ thống.")
    else:
        st.write(f"Hiện có **{len(members)}** thành viên đã được lưu trữ:")
        
        for member in sorted(members):
            member_dir = db_path / member
            images = [f.name for f in member_dir.iterdir() if f.is_file() and f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            
            with st.container():
                st.markdown(f"### {member}")
                if len(images) > 0:
                    cols = st.columns(min(len(images), 5))
                    for idx, img_name in enumerate(sorted(images)):
                        if idx < 5:
                            img_path = member_dir / img_name
                            with Image.open(img_path) as img:
                                cols[idx].image(img, caption=f"Ảnh mẫu {idx+1}", width=120)
                else:
                    st.write("Không có ảnh mẫu nào.")
                
                # Delete Button
                if st.button(f"Xóa Thành Viên: {member}", key=f"del_{member}"):
                    engine.delete_member(member)
                    st.session_state.success_message = f"Đã xóa thành viên '{member}' khỏi cơ sở dữ liệu."
                    st.rerun()
                st.markdown("---")

# ==========================================
# TAB 3: REGISTRATION
# ==========================================
with tab3:
    st.header("Đăng Ký Thành Viên Mới")
    
    # 5 angles defined
    STEPS = [
        {"label": "Chính giữa (Nhìn thẳng vào camera)", "filename": "front.png"},
        {"label": "Quay đầu sang bên TRÁI", "filename": "left.png"},
        {"label": "Quay đầu sang bên PHẢI", "filename": "right.png"},
        {"label": "Ngẩng đầu LÊN TRÊN", "filename": "up.png"},
        {"label": "Cúi đầu XUỐNG DƯỚI", "filename": "down.png"}
    ]
    
    # Initialize registration states
    if 'reg_name' not in st.session_state:
        st.session_state.reg_name = ""
    if 'reg_step' not in st.session_state:
        st.session_state.reg_step = 0
    if 'reg_images' not in st.session_state:
        st.session_state.reg_images = []
        
    # Reset helper
    def reset_registration_state():
        st.session_state.reg_name = ""
        st.session_state.reg_step = 0
        st.session_state.reg_images = []

    # Display registration messages if they exist
    if 'reg_success_message' in st.session_state:
        st.success(st.session_state.reg_success_message)
        del st.session_state.reg_success_message
    if 'reg_error_message' in st.session_state:
        st.error(st.session_state.reg_error_message)
        del st.session_state.reg_error_message

    # Show form if registration name is not locked yet
    if not st.session_state.reg_name:
        new_name = st.text_input("Nhập Tên Thành Viên (Chữ cái không dấu, số, gạch ngang, gạch dưới):")
        reg_source = st.radio("Chọn nguồn đăng ký:", ["Chụp 5 góc qua Webcam", "Upload file ảnh mẫu (Chọn 1 hoặc nhiều file)"])
        
        if reg_source == "Chụp 5 góc qua Webcam":
            if st.button("Bắt đầu đăng ký góc mặt"):
                cleaned_name = new_name.strip()
                if not cleaned_name:
                    st.error("Vui lòng điền tên trước khi bắt đầu.")
                elif not re.match(r"^[a-zA-Z0-9_ -]+$", cleaned_name):
                    st.error("Tên thành viên chỉ được chứa các ký tự chữ cái không dấu, số, dấu gạch ngang, gạch dưới hoặc khoảng trắng.")
                else:
                    # Lock name and start step 0
                    st.session_state.reg_name = cleaned_name
                    st.session_state.reg_step = 0
                    st.session_state.reg_images = []
                    st.rerun()
        else:
            # File Upload Source (supports multiple files)
            reg_files = st.file_uploader("Upload ảnh chân dung mẫu (có thể chọn nhiều file):", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="reg_multi_upload")
            
            if st.button("Đăng Ký qua Ảnh File"):
                cleaned_name = new_name.strip()
                if not cleaned_name:
                    st.error("Vui lòng điền tên trước khi đăng ký.")
                elif not re.match(r"^[a-zA-Z0-9_ -]+$", cleaned_name):
                    st.error("Tên thành viên chỉ được chứa các ký tự chữ cái không dấu, số, dấu gạch ngang, gạch dưới hoặc khoảng trắng.")
                elif not reg_files:
                    st.error("Vui lòng tải lên ít nhất một ảnh để đăng ký.")
                else:
                    valid_crops = []
                    for f in reg_files:
                        try:
                            pil_img = Image.open(f).convert("RGB")
                            cropped_face, _, face_count = engine.detect_face(pil_img)
                            if face_count == 1:
                                if cropped_face is not None:
                                    valid_crops.append(cropped_face)
                            else:
                                st.warning(f"File {f.name} bị bỏ qua vì không chứa đúng 1 khuôn mặt (tìm thấy {face_count} khuôn mặt).")
                        except Exception as e:
                            st.warning(f"Lỗi khi xử lý file {f.name}: {e}")
                    
                    if not valid_crops:
                        st.error("Không phát hiện thấy khuôn mặt hợp lệ trong bất kỳ ảnh nào được tải lên. Hãy chọn ảnh rõ mặt.")
                    else:
                        success_count = 0
                        for crop in valid_crops:
                            if engine.register_member(cleaned_name, crop):
                                success_count += 1
                        
                        if success_count > 0:
                            st.session_state.reg_success_message = f"Đăng ký thành công thành viên: **{cleaned_name}** với {success_count} ảnh mẫu hợp lệ!"
                            if 'reg_multi_upload' in st.session_state:
                                st.session_state.reg_multi_upload = []
                            st.rerun()
                        else:
                            st.error("Đăng ký thất bại. Không thể lưu bất kỳ khuôn mặt nào vào cơ sở dữ liệu.")
    else:
        # Guided Webcam Capture Flow in progress
        st.markdown(f"### Đang đăng ký góc mặt cho thành viên: **{st.session_state.reg_name}**")
        
        step_idx = st.session_state.reg_step
        if step_idx < 5:
            current_step = STEPS[step_idx]
            
            # Progress bar
            st.progress((step_idx) / 5)
            st.info(f"**Bước {step_idx + 1}/5**: Vui lòng chụp góc mặt: **{current_step['label']}**")
            
            # Camera snapshot
            reg_file = st.camera_input(f"Chụp góc: {current_step['label']}", key=f"cam_step_{step_idx}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Hủy đăng ký", key="cancel_reg"):
                    reset_registration_state()
                    st.rerun()
            
            if reg_file is not None:
                try:
                    pil_img = Image.open(reg_file).convert("RGB")
                    cropped_face, bbox, face_count = engine.detect_face(pil_img)
                    if face_count == 1:
                        if cropped_face is not None:
                            with col2:
                                if st.button(f"Lưu góc chụp: {current_step['label']}", key=f"save_step_{step_idx}"):
                                    # Save cropped image and its filename into session state
                                    st.session_state.reg_images.append((cropped_face, current_step['filename']))
                                    st.session_state.reg_step += 1
                                    st.rerun()
                            st.image(cropped_face, caption="Ảnh mặt cắt được hợp lệ", width=160)
                    elif face_count == 0:
                        st.error("Không thể phát hiện khuôn mặt nào trong ảnh. Hãy thử lại.")
                    else:
                        st.error("Phát hiện nhiều hơn 1 khuôn mặt. Ảnh đăng ký chỉ được phép chứa duy nhất 1 khuôn mặt.")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            # All 5 steps completed - Save images
            with st.spinner("Đang lưu các khuôn mặt đã chụp vào cơ sở dữ liệu..."):
                success_count = 0
                for crop, filename in st.session_state.reg_images:
                    if engine.register_member(st.session_state.reg_name, crop, filename=filename):
                        success_count += 1
            
            if success_count > 0:
                st.session_state.reg_success_message = f"Đăng ký thành công thành viên: **{st.session_state.reg_name}** với đủ {success_count}/5 góc mặt mẫu!"
            else:
                st.session_state.reg_error_message = f"Đăng ký thành viên thất bại. Không lưu được góc mặt nào."
            
            # Reset state
            reset_registration_state()
            st.rerun()

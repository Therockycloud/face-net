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

st.markdown('<h1 class="main-title">📸 NHẬN DIỆN KHUÔN MẶT THÀNH VIÊN</h1>', unsafe_allow_html=True)

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
tab1, tab2, tab3 = st.tabs(["🔍 Nhận Diện Face", "👥 Danh Sách Thành Viên", "➕ Đăng Ký Mới"])

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
                cropped_face, bbox = engine.detect_face(pil_img)
                
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
                st.markdown(f"### 👥 {member}")
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
    
    new_name = st.text_input("Nhập Tên Thành Viên (Không dấu, không khoảng trắng đặc biệt):")
    reg_source = st.radio("Chọn nguồn đăng ký:", ["Chụp qua Webcam", "Upload file ảnh mẫu"])
    
    reg_file = None
    if reg_source == "Chụp qua Webcam":
        reg_file = st.camera_input("Chụp ảnh chân dung của thành viên")
    else:
        reg_file = st.file_uploader("Upload ảnh chân dung:", type=["jpg", "png", "jpeg"], key="reg_upload")
        
    if st.button("Đăng Ký"):
        if not new_name.strip():
            st.error("Vui lòng điền tên trước khi đăng ký.")
        elif not re.match(r"^[a-zA-Z0-9_ -]+$", new_name.strip()):
            st.error("Tên thành viên chỉ được chứa các ký tự chữ cái không dấu, số, dấu gạch ngang, gạch dưới hoặc khoảng trắng.")
        elif reg_file is None:
            st.error("Vui lòng chụp hoặc tải ảnh lên để đăng ký.")
        else:
            try:
                pil_img = Image.open(reg_file).convert("RGB")
                with st.spinner("Đang phân tích khuôn mặt..."):
                    # Check face detection
                    cropped_face, bbox = engine.detect_face(pil_img)
                    
                    if cropped_face is not None:
                        success = engine.register_member(new_name.strip(), cropped_face)
                        if success:
                            st.success(f"Đăng ký thành công thành viên: **{new_name.strip()}**!")
                            st.image(cropped_face, caption="Khuôn mặt đã lưu trữ", width=160)
                        else:
                            st.error("Không thể đăng ký thành viên. Vui lòng thử lại.")
                    else:
                        st.error("Không thể phát hiện khuôn mặt nào trong ảnh đăng ký. Hãy chụp chính diện rõ mặt.")
            except Exception as e:
                st.error(f"Lỗi đăng ký: {e}")

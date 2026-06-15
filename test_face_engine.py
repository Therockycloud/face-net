from pathlib import Path
import numpy as np
from PIL import Image
from face_engine import FaceRecognitionEngine

def test_engine_initialization_creates_db_dir(tmp_path: Path) -> None:
    test_db_dir = tmp_path / "registered_faces"
    engine = FaceRecognitionEngine(db_dir=test_db_dir)
    assert test_db_dir.exists()

def test_detect_face_returns_none_for_blank_image(tmp_path):
    engine = FaceRecognitionEngine(db_dir=tmp_path)
    # Create a blank black image
    blank_img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    cropped_face, bbox = engine.detect_face(blank_img)
    assert cropped_face is None
    assert bbox is None

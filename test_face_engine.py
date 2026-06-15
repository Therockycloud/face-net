from pathlib import Path
import numpy as np
from PIL import Image
import torch
from face_engine import FaceRecognitionEngine

def test_engine_initialization_creates_db_dir(tmp_path: Path) -> None:
    test_db_dir = tmp_path / "registered_faces"
    engine = FaceRecognitionEngine(db_dir=test_db_dir)
    assert test_db_dir.exists()

def test_detect_face_returns_none_for_blank_image(tmp_path: Path) -> None:
    engine = FaceRecognitionEngine(db_dir=tmp_path)
    # Create a blank black image
    blank_img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    cropped_face, bbox = engine.detect_face(blank_img)
    assert cropped_face is None
    assert bbox is None

def test_get_embedding_returns_512_dimensional_tensor(tmp_path: Path) -> None:
    engine = FaceRecognitionEngine(db_dir=tmp_path)
    dummy_face = Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
    emb = engine.get_embedding(dummy_face)
    assert emb is not None
    assert emb.shape == (512,)
    assert emb.device.type == "cpu"

def test_cosine_similarity(tmp_path: Path) -> None:
    engine = FaceRecognitionEngine(db_dir=tmp_path)
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v2 = torch.tensor([1.0, 0.0, 0.0])
    v3 = torch.tensor([0.0, 1.0, 0.0])
    
    assert abs(engine.compute_similarity(v1, v2) - 1.0) < 1e-5
    assert abs(engine.compute_similarity(v1, v3) - 0.0) < 1e-5



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
    cropped_face, bbox, face_count = engine.detect_face(blank_img)
    assert cropped_face is None
    assert bbox is None
    assert face_count == 0

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


def test_register_and_load_and_delete_member(tmp_path: Path) -> None:
    engine = FaceRecognitionEngine(db_dir=tmp_path)
    
    # Assert database starts empty
    engine.load_db()
    assert len(engine.cache) == 0
    
    # Save a fake face image
    dummy_face = Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
    
    # Recognize face when cache is empty
    name, score = engine.recognize_face(dummy_face, threshold=0.5)
    assert name == "Unknown"
    assert score == -1.0
    
    # Register member
    success = engine.register_member("UserA", dummy_face)
    assert success is True
    assert (tmp_path / "UserA").exists()
    
    # Reload and assert cached
    engine.load_db()
    assert "UserA" in engine.cache
    assert len(engine.cache["UserA"]) == 1
    
    # Recognize face
    name, score = engine.recognize_face(dummy_face, threshold=0.5)
    assert name == "UserA"
    assert score >= 0.5
    
    # Delete member
    engine.delete_member("UserA")
    assert not (tmp_path / "UserA").exists()
    
    # Reload and assert empty
    engine.load_db()
    assert "UserA" not in engine.cache


def test_path_traversal_protection(tmp_path: Path) -> None:
    engine = FaceRecognitionEngine(db_dir=tmp_path)
    dummy_face = Image.fromarray(np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8))
    
    # These should be rejected and return False
    assert engine.register_member(".", dummy_face) is False
    assert engine.register_member("..", dummy_face) is False
    assert engine.register_member("", dummy_face) is False
    
    # Path extraction should keep absolute or relative traversal parts safe (only using name)
    assert engine.register_member("/etc/passwd", dummy_face) is True
    assert (tmp_path / "passwd").exists()
    
    # Try deleting with unsafe names (should not raise exceptions)
    engine.delete_member(".")
    engine.delete_member("..")
    engine.delete_member("")


def test_count_faces(tmp_path: Path) -> None:
    engine = FaceRecognitionEngine(db_dir=tmp_path)
    blank_img = Image.fromarray(np.zeros((200, 200, 3), dtype=np.uint8))
    assert engine.count_faces(blank_img) == 0



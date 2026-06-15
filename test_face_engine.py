from pathlib import Path
from face_engine import FaceRecognitionEngine

def test_engine_initialization_creates_db_dir(tmp_path: Path) -> None:
    test_db_dir = tmp_path / "registered_faces"
    engine = FaceRecognitionEngine(db_dir=test_db_dir)
    assert test_db_dir.exists()

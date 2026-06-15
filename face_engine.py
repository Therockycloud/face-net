from pathlib import Path
from typing import Any, Dict, Union

class FaceRecognitionEngine:
    """Engine for face recognition operations, including registration and cache management."""

    def __init__(self, db_dir: Union[str, Path] = "./data/registered_faces") -> None:
        """Initialize the engine, creating the database directory if it does not exist.

        Args:
            db_dir: Path to the database directory where face encodings are stored.
        """
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.cache: Dict[str, Any] = {}

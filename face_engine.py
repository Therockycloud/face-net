import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List, Optional
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import shutil
import time

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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=False, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.load_db()

    def detect_face(self, pil_image: Image.Image) -> Tuple[Optional[Image.Image], Optional[List[int]], int]:
        """
        Detects a single face in an image.
        Returns:
            cropped_face: PIL Image of the cropped face resized to 160x160, or None.
            box: Bounding box coordinates [left, top, right, bottom], or None.
            face_count: Number of faces detected in the image.
        """
        pil_image = pil_image.convert('RGB')
        boxes, _ = self.mtcnn.detect(pil_image)
        face_count = len(boxes) if boxes is not None else 0
        if face_count > 0:
            box = boxes[0]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            width, height = pil_image.size
            x1 = max(0, min(width, x1))
            y1 = max(0, min(height, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            
            if x2 > x1 and y2 > y1:
                cropped = pil_image.crop((x1, y1, x2, y2))
                cropped_resized = cropped.resize((160, 160), Image.Resampling.LANCZOS)
                return cropped_resized, [x1, y1, x2, y2], face_count
        return None, None, face_count

    def count_faces(self, pil_image: Image.Image) -> int:
        """Counts the number of faces in an image.

        Args:
            pil_image: A PIL Image to scan for faces.

        Returns:
            The number of detected faces.
        """
        pil_image = pil_image.convert('RGB')
        boxes, _ = self.mtcnn.detect(pil_image)
        return len(boxes) if boxes is not None else 0

    def get_embedding(self, cropped_face_img: Image.Image) -> torch.Tensor:
        """Extracts a 512-dimensional embedding vector from a cropped face image.

        Args:
            cropped_face_img: PIL Image containing the cropped and resized face.

        Returns:
            A 512-dimensional torch.Tensor on the CPU representing the face embedding.
        """
        cropped_face_img = cropped_face_img.convert('RGB')
        img_tensor = self.transform(cropped_face_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.resnet(img_tensor).squeeze(0).cpu()
        return embedding

    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Computes cosine similarity between two embedding tensors.

        Args:
            emb1: The first embedding tensor.
            emb2: The second embedding tensor.

        Returns:
            The cosine similarity score as a float.
        """
        return F.cosine_similarity(emb1.unsqueeze(0).cpu(), emb2.unsqueeze(0).cpu()).item()

    def load_db(self) -> None:
        """
        Scans self.db_dir, extracts embeddings for all images, and updates self.cache.
        """
        self.cache = {}
        if not self.db_dir.exists():
            return
        
        for member_dir in self.db_dir.iterdir():
            if not member_dir.is_dir():
                continue
            
            name = member_dir.name
            embeddings = []
            for img_path in member_dir.iterdir():
                if img_path.is_file() and img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    try:
                        with Image.open(img_path) as img:
                            rgb_img = img.convert('RGB')
                            cropped_resized = rgb_img.resize((160, 160), Image.Resampling.LANCZOS)
                            emb = self.get_embedding(cropped_resized)
                            embeddings.append(emb)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            if embeddings:
                self.cache[name] = embeddings

    def register_member(self, name: str, cropped_face_img: Image.Image, filename: Optional[str] = None) -> bool:
        """
        Saves cropped face image to database directory and updates the cache.
        """
        safe_name = Path(name).name.strip()
        if not safe_name or safe_name in (".", ".."):
            return False
        
        member_dir = self.db_dir / safe_name
        member_dir.mkdir(parents=True, exist_ok=True)
        
        if not filename:
            filename = f"{int(time.time() * 1000)}.png"
        file_path = member_dir / filename
        
        try:
            cropped_face_img.save(file_path)
        except Exception as e:
            print(f"Error saving image for {safe_name}: {e}")
            return False
        
        emb = self.get_embedding(cropped_face_img)
        if safe_name not in self.cache:
            self.cache[safe_name] = []
        self.cache[safe_name].append(emb)
        return True

    def delete_member(self, name: str) -> None:
        """
        Removes a member's folder from the disk and updates the cache.
        """
        safe_name = Path(name).name.strip()
        if not safe_name or safe_name in (".", ".."):
            return
        
        member_dir = self.db_dir / safe_name
        if member_dir.exists() and member_dir.is_dir():
            shutil.rmtree(member_dir)
        
        if safe_name in self.cache:
            del self.cache[safe_name]

    def recognize_face(self, cropped_face_img: Image.Image, threshold: float = 0.60) -> Tuple[str, float]:
        """
        Compares query cropped face with all embeddings in cache.
        Returns:
            name: Matched member name or "Unknown"
            best_score: Highest cosine similarity score
        """
        if not self.cache:
            return "Unknown", -1.0
        
        query_emb = self.get_embedding(cropped_face_img)
        best_name = "Unknown"
        best_score = -1.0
        
        for name, embeddings in self.cache.items():
            for ref_emb in embeddings:
                score = self.compute_similarity(query_emb, ref_emb)
                if score > best_score:
                    best_score = score
                    if score >= threshold:
                        best_name = name
                        
        return best_name, best_score




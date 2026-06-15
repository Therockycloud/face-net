import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from typing import Any, Dict, Tuple, Union, List, Optional
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

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

    def detect_face(self, pil_image: Image.Image) -> Tuple[Optional[Image.Image], Optional[List[int]]]:
        """
        Detects a single face in an image.
        Returns:
            cropped_face: PIL Image of the cropped face resized to 160x160, or None.
            box: Bounding box coordinates [left, top, right, bottom], or None.
        """
        pil_image = pil_image.convert('RGB')
        boxes, _ = self.mtcnn.detect(pil_image)
        if boxes is not None and len(boxes) > 0:
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
                return cropped_resized, [x1, y1, x2, y2]
        return None, None

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



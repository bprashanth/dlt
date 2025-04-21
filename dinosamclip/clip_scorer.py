import torch
import clip
from PIL import Image
import logging


class CLIPScorer:

    def __init__(self, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.logger = logging.getLogger('CLIPScorer')
        self.logger.info(f"Initialized CLIP scorer on device: {self.device}")

    def score(self, image_path, prompt):
        self.logger.info(f"Scoring image: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([prompt]).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                similarity = torch.nn.functional.cosine_similarity(
                    image_features, text_features)

            return similarity.item()
        except Exception as e:
            self.logger.error(f"Error scoring image: {e}")
            return 0.0

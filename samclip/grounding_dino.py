import torch
from torchvision.ops import box_convert
import os
import matplotlib.pyplot as plt
from groundingdino.util.inference import load_model, predict
import torchvision.transforms as T
import logging
from PIL import Image
from utils import get_color
import numpy as np
from groundingdino.datasets import transforms as GT


class GroundingDINOBoxGenerator:
    def __init__(self, dino_config,  dino_weights, device=None):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(dino_config, dino_weights).to(self.device)
        self.logger = logging.getLogger('GroundingDINOBoxGenerator')
        self.logger.info(
            f"Initialized GroundingDINO box generator on device: {self.device}")

    def generate_boxes(self, image_rgb, text_prompt, output_dir="output", box_threshold=0.25, text_threshold=0.25):
        # Convert numpy array to PIL Image if not already
        image_pil = Image.fromarray(image_rgb) if isinstance(
            image_rgb, np.ndarray) else image_rgb

        transform = GT.Compose([
            GT.RandomResize([800], max_size=1333),
            GT.ToTensor(),
            GT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Transform the image
        image_tensor, _ = transform(image_pil, None)
        image_tensor = image_tensor.to(self.device)

        # Get original size for scaling boxes later
        w, h = image_pil.size

        self.logger.info(f"Generating boxes for image with shape: {h}x{w}")
        self.logger.info(
            f"Using text_prompt: {text_prompt}, box_threshold: {box_threshold}, text_threshold: {text_threshold}")

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        # Scale boxes to original image size
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes, in_fmt="cxcywh",
                            out_fmt="xyxy").int().cpu().numpy()

        self.logger.info(f"Generated {len(boxes)} boxes")

        self.apply_boxes(image_rgb, boxes, phrases,
                         logits, output_dir)

        return boxes, logits.cpu().numpy(), phrases

    def apply_boxes(self, image_rgb, boxes, phrases, logits, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)

        # Create a unique color for each phrase
        colors = [get_color() for _ in range(len(boxes))]

        for box, phrase, score, color in zip(boxes, phrases, logits, colors):
            x1, y1, x2, y2 = box
            # Draw rectangle
            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False,
                linewidth=2,
                edgecolor=color
            )
            plt.gca().add_patch(rect)

            # Add text label with score
            text = f"{phrase}: {score:.2f}"
            plt.text(
                x1, y1-5,
                text,
                color=color,
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.8,
                          edgecolor='none', pad=1.5)
            )

        plt.axis('off')
        output_path = os.path.join(output_dir, "groundingdino_boxes.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        self.logger.info(f"Saved box visualization to {output_path}")

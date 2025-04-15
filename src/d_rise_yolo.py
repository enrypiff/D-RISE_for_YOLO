import os
import torch
from ultralytics import YOLO
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from vision_explanation_methods.explanations.drise import DRISE_saliency
from vision_explanation_methods.explanations.common import DetectionRecord, GeneralObjectDetectionModelWrapper
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class YOLOv8Wrapper(GeneralObjectDetectionModelWrapper):
    """Wrapper for YOLOv8 to make it compatible with D-RISE."""
    def __init__(self, model_path: str):
        self.model = YOLO(model_path, verbose=False)
        self.model.to(device="cuda" if torch.cuda.is_available() else "cpu")  # Move model to GPU if available
        self.num_classes = self.model.model.yaml['nc']  # Number of classes in the model

    def predict(self, x: torch.Tensor, conf_threshold: float = 0.5) -> list:
        """
        Run predictions and return detections in DetectionRecord format.
        
        Args:
            x (torch.Tensor): Input image tensor.
            conf_threshold (float): Confidence threshold for predictions.


        Returns:
            list: List of DetectionRecord objects containing bounding boxes, scores, and class scores.
        """

        # Run YOLOv8 predictions
        results = self.model.predict(source=x, conf=conf_threshold, verbose=False)
        # Convert results to DetectionRecord format
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.clone().detach().to(dtype=torch.float32)  # [N, 4]
            scores = result.boxes.conf.clone().detach().to(dtype=torch.float32)  # [N]
            labels = result.boxes.cls.clone().detach().to(dtype=torch.int64)  # [N]
            
            class_scores = torch.zeros((len(labels), self.num_classes), dtype=torch.float32)
            for i, label in enumerate(labels):
                class_scores[i, label] = scores[i]
            detections.append(DetectionRecord(bounding_boxes=boxes, objectness_scores=scores, class_scores=class_scores))
        return detections


def save_saliency_images(
        image: Image.Image,
        saliency_map: np.ndarray,
        bbox: np.ndarray,
        score: float,
        label: str,
        output_dir: str,
        image_name: str,
        index: int
        ):
    """
    Save the original image, saliency mask, and merged saliency mask.
    
    Args:
        image (Image.Image): The original image.
        saliency_map (np.ndarray): The saliency map.
        bbox (np.ndarray): The bounding box coordinates.
        score (float): The confidence score.
        label (str): The class label.
        output_dir (str): The directory to save the images.
        image_name (str): The name of the image file.
        index (int): The index of the detection.    
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create a visualization with matplotlib
    plt.figure(figsize=(15, 5))
    
    # Original image with bounding boxes
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title(f"Original Image - {label}")
    plt.axis('off')
    
    # Draw bounding box for this detection
    x, y, x2, y2 = bbox
    width, height = x2 - x, y2 - y
    plt.gca().add_patch(Rectangle((x, y), width, height, fill=False, edgecolor='red', linewidth=2))
    plt.gca().text(x, y - 10, f"{label} - {score:.2f}", color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    # Overlay saliency map on original image
    plt.subplot(1, 3, 2)
    saliency_mask = saliency_map['detection'].cpu().numpy().transpose(1, 2, 0)
    saliency_mask = np.squeeze(saliency_mask)  # Ensure it's 2D for colormap
    saliency_mask = saliency_mask[:, :, 0]  # Use only the first channel  
    plt.imshow(image)
    plt.imshow(saliency_mask, alpha=0.5, cmap='jet')  # Use the same colormap for consistency
    plt.colorbar(label='Saliency Score', orientation='horizontal', pad=0.1)
    plt.title(f"Overlay - {label}")
    plt.axis('off')

    # Cropped image with saliency overlay
    plt.subplot(1, 3, 3)
    cropped_saliency = saliency_mask[int(y):int(y2), int(x):int(x2)]  # Crop the saliency map to the bounding box
    image_array = np.array(image)  # Convert PIL Image to NumPy array
    cropped_image = image_array[int(y):int(y2), int(x):int(x2)]  # Crop the original image to the bounding box
    plt.imshow(cropped_image)
    plt.imshow(cropped_saliency, alpha=0.3, cmap='jet')  # Overlay saliency map on cropped image
    plt.colorbar(label='Saliency Score', orientation='horizontal', pad=0.1)
    plt.title(f"Cropped Overlay - {label}")
    plt.axis('off')

    # Save the visualization in the output folder
    plt.tight_layout()
    output_file_name = os.path.basename(image_name)
    output_file_name = os.path.splitext(output_file_name)[0]  # Remove file extension
    output_file_name = f"{output_file_name}_{label}_{index}.png"  # Append index to filename
    output_file_path = os.path.join(output_dir, output_file_name)
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()
import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from vision_explanation_methods.explanations.drise import DRISE_saliency
from src.d_rise_yolo import YOLOv8Wrapper, save_saliency_images
from tqdm import tqdm
import argparse


def generate_saliency_maps_yolov8(
        images_folder_path: str, 
        model_path: str, 
        output_dir: str, 
        conf_threshold: float = 0.5,
        img_size: int = 640,
        mask_num: int = 500,
        mask_res: int = 8,
        mask_padding: int = None,
        ):
    """
    Generate saliency maps for a YOLOv8 model.

    Args:
        images_folder_path (str): Path to the folder containing images.
        model_path (str): Path to the YOLOv8 model file.
        output_dir (str): Path to the folder where results will be saved.
        conf_threshold (float): Confidence threshold for displaying detections.
        img_size (int): Size to resize images for the model.
        mask_num (int): Number of masks to generate for D-RISE. More is slower but gives higher quality mask.
        mask_res (int): Resolution of the base mask. High resolutions will give finer masks, but more need to be run.
        mask_padding (int): Padding for the mask.

    """
    
    # Load the YOLOv8 model
    yolo_wrapper = YOLOv8Wrapper(model_path=model_path)

    # Get the list of image files in the folder and its subfolders
    image_files = []
    for root, _, files in os.walk(images_folder_path):
        image_files.extend([os.path.join(root, f) for f in files if f.endswith(('.jpg', '.png', '.jpeg'))])

    print(f"Found {len(image_files)} images in {images_folder_path}.")

    # Process each image file
    for image_file in tqdm(image_files, desc="Processing images"):
        image = Image.open(image_file).convert("RGB")
        if image is None:
            print(f"Failed to load image {image_file}")
            continue
        
        image = image.resize((img_size, img_size))  # Resize to model input size
        image_tensor = ToTensor()(image).unsqueeze(0).to(device="cuda" if torch.cuda.is_available() else "cpu")  # Add batch dimension and move to GPU

        detections = yolo_wrapper.predict(image_tensor, conf_threshold=conf_threshold)
        if not detections or len(detections[0].bounding_boxes) == 0:
            print(f"No detections found for image: {image_file}")
            continue

        # Generate saliency maps using D-RISE
        saliency_maps = DRISE_saliency(
            model=yolo_wrapper,
            image_tensor=image_tensor,
            target_detections=detections,
            number_of_masks=mask_num,  # Adjust for quality vs. speed
            mask_res=(mask_res, mask_res), # Resolution of the masks
            mask_padding=mask_padding, # Padding for the masks
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True
        )

        # Save images for each detection
        for i, saliency_map in enumerate(saliency_maps[0]):
            bbox = detections[0].bounding_boxes[i].cpu().numpy()
            score = detections[0].objectness_scores[i].item()
            label = int(torch.argmax(detections[0].class_scores[i]))
            label = yolo_wrapper.model.model.names[label]  # Get class name from index
            save_saliency_images(image, saliency_map, bbox, score, label, output_dir, image_file, i)


# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run object detection and D-RISE explanation.")
    parser.add_argument("--images_folder_path", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the object detection model.")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Path to the folder where results will be saved.")
    parser.add_argument("--conf_threshold", type=float, default=0.50, help="Confidence threshold for displaying detections.")
    parser.add_argument("--img_size", type=int, default=640, help="Size to resize images for the model.")
    parser.add_argument("--masks_num", type=int, default=500, help="Number of masks to generate for D-RISE.")
    parser.add_argument("--mask_res", type=int, default=8, help="Resolution of the base mask.")
    parser.add_argument("--maskpadding", type=int, default=None, help="Padding for the mask.")

    args = parser.parse_args()

    # Check if the model path is provided, if not, use a default model
    if args.model_path is None:
        args.model_path = "yolov8n.pt"  # Default model path

    if torch.cuda.is_available():
        print("Using GPU for processing.")
    else:
        print("Using CPU for processing.")

    generate_saliency_maps_yolov8(
        images_folder_path=args.images_folder_path,
        model_path=args.model_path,
        output_dir=args.output_folder_path,
        conf_threshold=args.conf_threshold,
        img_size=args.img_size,
        mask_num=args.masks_num,
        mask_res=args.mask_res,
        mask_padding=args.maskpadding,
    )
import os
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
from vision_explanation_methods.explanations.drise import DRISE_saliency
from src.d_rise_yolo import YOLOv8Wrapper
from src.visualization import generate_saliency_maps_yolov8
from src.d_rise_modified import DRISE_saliency_with_debug, set_seed
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt



# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run object detection and D-RISE explanation.")
    parser.add_argument("--images_folder_path", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--model_path", type=str, required=False, help="Path to the object detection model.")
    parser.add_argument("--output_folder_path", type=str, required=True, help="Path to the folder where results will be saved.")
    parser.add_argument("--conf_threshold", type=float, default=0.30, help="Confidence threshold for displaying detections.")
    parser.add_argument("--img_size", type=int, default=640, help="Size to resize images for the model.")
    parser.add_argument("--masks_num", type=int, default=500, help="Number of masks to generate for D-RISE.")
    parser.add_argument("--mask_res", type=int, default=16, help="Resolution of the base mask.")
    parser.add_argument("--mask_padding", type=int, default=None, help="Padding for the mask.")
    parser.add_argument("--save_masks", type=bool, default=False, help="Whether to save generated masks.")
    parser.add_argument("--save_masked_images", type=bool, default=False, help="Whether to save images with masks applied.")
    parser.add_argument("--save_predictions", type=bool, default=False, help="Whether to save prediction results on masked images.")
    parser.add_argument("--save_individual_saliency", type=bool, default=False, help="Whether to save individual saliency maps.")
    parser.add_argument("--debug_sample_count", type=int, default=45, help="Number of debug samples to save.")
    parser.add_argument("--deterministic_generation", type=bool, default=False, help="Whether to use deterministic generation.")
    parser.add_argument("--resolution_decrease_factor", type=float, default=0.5, help="Factor to decrease resolution by if flat saliency maps are detected.")
    parser.add_argument("--max_resolution_attempts", type=int, default=3, help="Maximum number of attempts to find a suitable mask resolution.")
    parser.add_argument("--mark_high_intensity", type=bool, default=False, help="Whether to mark high intensity areas in the saliency maps.")
    parser.add_argument("--mark_high_intensity_threshold_mid", type=float, default=0.8, help="Threshold for mid intensity.")
    parser.add_argument("--mark_high_intensity_threshold_high", type=float, default=0.9, help="Threshold for high intensity.")

    args = parser.parse_args()

    # Check if the model path is provided, if not, use a default model
    if args.model_path is None:
        args.model_path = "yolov8n.pt"  # Default model path

    if torch.cuda.is_available():
        print("Using GPU for processing.")
    else:
        print("Using CPU for processing.")

    if args.deterministic_generation:
        set_seed(42)
        print("Using deterministic generation.")

    generate_saliency_maps_yolov8(
        images_folder_path=args.images_folder_path,
        model_path=args.model_path,
        output_dir=args.output_folder_path,
        conf_threshold=args.conf_threshold,
        img_size=args.img_size,
        mask_num=args.masks_num,
        mask_res=args.mask_res,
        mask_padding=args.mask_padding,
        save_masks=args.save_masks,
        save_masked_images=args.save_masked_images,
        save_predictions=args.save_predictions,
        save_individual_saliency=args.save_individual_saliency,
        debug_sample_count=args.debug_sample_count,
        resolution_decrease_factor=args.resolution_decrease_factor,
        max_resolution_attempts=args.max_resolution_attempts,
        mark_high_intensity=args.mark_high_intensity,
        mark_high_intensity_threshold_mid=args.mark_high_intensity_threshold_mid,
        mark_high_intensity_threshold_high=args.mark_high_intensity_threshold_high
    )
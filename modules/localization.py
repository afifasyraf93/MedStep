import cv2
import numpy as np
import torch
import base64
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
import io

from modules.detection import build_model, PATHOLOGIES, TRANSFORM


def get_target_layer(model, model_name="densenet121"):
    """Return the target layer for Grad-CAM based on model architecture."""
    if model_name == "densenet121":
        return [model.features.denseblock4]
    elif model_name == "resnet50":
        return [model.layer4[-1]]
    elif model_name == "efficientnet_b0":
        return [model.features[-1]]
    elif model_name == "mobilenet_v3":
        return [model.features[-1]]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def apply_intensity_threshold(heatmap, percentile=70):
    """Zero out activations below the given percentile.
    Only keep the top (100 - percentile)% of activations.
    """
    # TODO: compute the threshold value at the given percentile
    # set all values below threshold to 0
    # return the masked heatmap
    threshold = np.percentile(heatmap, percentile)
    heatmap = np.where(heatmap >= threshold, heatmap, 0)
    return heatmap


def apply_lung_mask(heatmap, original_gray):
    """Use Otsu thresholding on the X-ray to create a lung region mask.
    Zero out heatmap activations outside the detected chest region.
    """
    # TODO:
    # 1. Convert original_gray to uint8 (0-255)
    # 2. Apply Otsu threshold: cv2.threshold with cv2.THRESH_OTSU
    # 3. Apply morphological operations to clean up the mask
    #    - cv2.morphologyEx with cv2.MORPH_CLOSE to fill holes
    #    - cv2.dilate to expand slightly
    # 4. Multiply heatmap by (mask / 255) to zero out outside regions
    # 5. Return masked heatmap
    gray_uint8 = (original_gray * 255).astype(np.uint8)
    _, mask = cv2.threshold(gray_uint8, 0, 255, 
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    heatmap = heatmap * (mask / 255.0)
    return heatmap


def generate_heatmap(model, image_tensor, pathology_index, 
                     original_image_array, model_name="densenet121"):
    """Generate a Grad-CAM heatmap for a specific pathology.
    
    Args:
        model: loaded PyTorch model
        image_tensor: preprocessed image tensor [1, 3, 224, 224]
        pathology_index: index of target pathology (0-5)
        original_image_array: numpy array of original image, shape [224, 224, 3], 
                               values 0-1 float32
        model_name: which architecture to use for target layer
    
    Returns:
        heatmap_array: numpy array of final heatmap [224, 224, 3]
        base64_str: base64 encoded side-by-side PNG
    """
    target_layers = get_target_layer(model, model_name)

    # Grad-CAM target — focus on specific pathology output
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    targets = [ClassifierOutputTarget(pathology_index)]

    # Generate raw Grad-CAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0]  # shape [224, 224]

    # TODO: call apply_intensity_threshold on grayscale_cam
    # TODO: get grayscale version of original for lung mask
    #       Hint: convert original_image_array to grayscale
    #       np.mean(original_image_array, axis=2) gives grayscale
    # TODO: call apply_lung_mask with grayscale_cam and the grayscale image
    grayscale_cam = apply_intensity_threshold(grayscale_cam, percentile=70)
    original_gray = np.mean(original_image_array, axis=2)
    grayscale_cam = apply_lung_mask(grayscale_cam, original_gray)

    # Overlay heatmap on original image
    heatmap_overlay = show_cam_on_image(
        original_image_array, grayscale_cam, use_rgb=True)

    # TODO: create side by side image
    # Convert original_image_array to uint8 (multiply by 255)
    # Stack original and heatmap_overlay side by side using np.hstack
    # Add a title/label — use matplotlib to add text "Original" and "Heatmap"
    original_uint8 = (original_image_array * 255).astype(np.uint8)
    side_by_side = np.hstack([original_uint8, heatmap_overlay])

    # TODO: encode final image to base64
    # Use PIL to save to bytes buffer
    # Use base64.b64encode to encode
    img = Image.fromarray(side_by_side)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return heatmap_overlay, base64_str


def generate_all_heatmaps(model, image_tensor, detections, 
                           original_image_array, model_name="densenet121"):
    """Generate heatmaps for all detected pathologies.
    
    Args:
        detections: dict of {pathology_name: probability}
        threshold: only generate heatmap if probability >= 0.5
    
    Returns:
        dict of {pathology_name: base64_str}
    """
    # TODO: iterate over PATHOLOGIES
    # for each pathology where detections[pathology] >= 0.5
    # get pathology_index using PATHOLOGIES.index(pathology)
    # call generate_heatmap and store base64_str in results dict
    # return results dict
    results = {}
    for pathology in PATHOLOGIES:
        if detections.get(pathology, 0) >= 0.5:
            idx = PATHOLOGIES.index(pathology)
            _, base64_str = generate_heatmap(
                model, image_tensor, idx, 
                original_image_array, model_name)
            results[pathology] = base64_str
    return results
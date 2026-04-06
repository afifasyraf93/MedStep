import cv2
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from modules.detection import TRANSFORM, PATHOLOGIES


def get_target_layer(model):
    """
    Return the target layer for Grad-CAM.
    For DenseNet121, the last conv layer is features.denseblock4
    """
    return [model.features.denseblock4.denselayer16.conv2]


def generate_heatmap(model, image_path, pathology, device="cuda"):
    """
    Generate Grad-CAM heatmap for a specific pathology.

    Args:
        model:        loaded DenseNet121 model in eval mode
        image_path:   path to chest X-ray image
        pathology:    string name e.g. "pneumonia"
        device:       "cuda" or "cpu"

    Returns:
        heatmap_overlay: PIL Image with heatmap overlaid on original
        heatmap_raw:     numpy array of raw heatmap (0-1)
        cam_score:       float, mean activation score
    """
    # Get pathology index
    if pathology not in PATHOLOGIES:
        raise ValueError(f"Unknown pathology: {pathology}. "
                         f"Must be one of {PATHOLOGIES}")
    class_idx = PATHOLOGIES.index(pathology)

    # Load and preprocess image
    orig_image = Image.open(image_path).convert("RGB")
    orig_array = np.array(orig_image.resize((224, 224))) / 255.0
    orig_array = orig_array.astype(np.float32)

    input_tensor = TRANSFORM(orig_image).unsqueeze(0).to(device)

    # Handle DenseNet classifier wrapper
    # pytorch-grad-cam needs the model to output raw logits
    target_layers = get_target_layer(model)
    targets       = [ClassifierOutputTarget(class_idx)]

    # Generate CAM
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]  # shape: (224, 224)

    # Overlay heatmap on original image
    overlay = show_cam_on_image(
        orig_array,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.6   # 60% original, 40% heatmap
    )

    heatmap_pil = Image.fromarray(overlay)
    cam_score   = float(grayscale_cam.mean())

    return heatmap_pil, grayscale_cam, cam_score


def generate_all_heatmaps(model, image_path, detection_results,
                           device="cuda", threshold=0.3):
    """
    Generate heatmaps for all detected pathologies.

    Args:
        model:             loaded model
        image_path:        path to image
        detection_results: dict from detection.predict()
        device:            cuda or cpu
        threshold:         only generate for pathologies above this prob

    Returns:
        dict: {pathology: {"heatmap": PIL, "score": float}}
              only includes detected pathologies
    """
    results = {}

    for pathology, info in detection_results.items():
        if info["probability"] >= threshold:
            try:
                heatmap, raw_cam, score = generate_heatmap(
                    model, image_path, pathology, device
                )
                results[pathology] = {
                    "heatmap":  heatmap,
                    "raw_cam":  raw_cam,
                    "score":    score,
                    "prob":     info["probability"]
                }
                print(f"  Grad-CAM generated for {pathology} "
                      f"(prob={info['probability']:.3f}, "
                      f"score={score:.3f})")
            except Exception as e:
                print(f"  WARNING: Grad-CAM failed for "
                      f"{pathology}: {e}")

    return results

def generate_all_heatmaps(model, image_path, detection_results,
                           device="cuda", threshold=0.3):
    results = {}

    for pathology, info in detection_results.items():
        if info["probability"] >= threshold:
            try:
                heatmap, raw_cam, score = generate_heatmap(
                    model, image_path, pathology, device
                )

                # Flag low-quality heatmaps
                quality = "good" if score >= 0.05 else "low"

                results[pathology] = {
                    "heatmap": heatmap,
                    "raw_cam": raw_cam,
                    "score":   score,
                    "prob":    info["probability"],
                    "quality": quality
                }
                print(f"  Grad-CAM [{quality}] {pathology} "
                      f"(prob={info['probability']:.3f}, "
                      f"score={score:.3f})")
            except Exception as e:
                print(f"  WARNING: Grad-CAM failed for "
                      f"{pathology}: {e}")

    return results

def save_heatmap(heatmap_pil, output_path):
    """Save heatmap PIL image to disk."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    heatmap_pil.save(output_path)
    return output_path

def heatmap_to_base64(heatmap_pil):
    """Convert PIL heatmap to base64 string for API transmission."""
    import io
    import base64
    buffer = io.BytesIO()
    heatmap_pil.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")
import os
import base64
import json
from io import BytesIO
from PIL import Image
from groq import Groq
from dotenv import load_dotenv
from modules.detection import PATHOLOGIES

load_dotenv("api.env")

def _encode_image_base64(image_input):
    """
    Convert image to base64 string.
    Accepts PIL Image or file path.
    """
    if isinstance(image_input, str):
        with open(image_input, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    elif isinstance(image_input, Image.Image):
        buffer = BytesIO()
        image_input.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")
    else:
        raise ValueError("image_input must be a file path or PIL Image")


def _format_detection_summary(detection_results, threshold=0.3):
    """Format detection results into readable text for the prompt."""
    detected   = []
    not_detected = []

    for path, info in detection_results.items():
        prob = info["probability"]
        if prob >= threshold:
            detected.append(f"{path.replace('_', ' ')} "
                            f"({prob:.1%} confidence)")
        else:
            not_detected.append(f"{path.replace('_', ' ')} "
                                 f"({prob:.1%})")

    summary = ""
    if detected:
        summary += "DETECTED: " + ", ".join(detected)
    else:
        summary += "DETECTED: none above threshold"

    summary += "\nNOT DETECTED: " + ", ".join(not_detected)
    return summary


def generate_findings(client, image_b64, detection_summary,
                       heatmap_b64=None):
    """
    First API call — generate radiological findings.
    Returns findings text.
    """
    # Build image content list
    content = []

    # Add original X-ray
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_b64}"
        }
    })

    # Add heatmap if available
    if heatmap_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{heatmap_b64}"
            }
        })

    detection_block = f"""
AI Detection Results:
{detection_summary}
"""

    content.append({
        "type": "text",
        "text": f"""You are an experienced radiologist reviewing a 
chest X-ray for medical education purposes.

{detection_block}

The second image (if provided) is a Grad-CAM heatmap showing which 
regions the AI focused on. Red/yellow areas indicate high attention.

Please provide FINDINGS only — describe what you observe in the 
chest X-ray including:
- Lung fields (any opacities, consolidations, effusions)
- Heart size and borders
- Costophrenic angles
- Any notable abnormalities

Be concise and use standard radiological terminology.
Write 3-5 sentences maximum.
Do NOT write an impression yet."""
    })

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": content}],
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


def generate_impression(client, findings, detection_summary):
    """
    Second API call — generate impression based on findings.
    Text only, no image needed.
    Returns impression text.
    """
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": f"""You are an experienced radiologist 
writing a radiology report for medical education purposes.

AI Detection Results:
{detection_summary}

Radiological Findings:
{findings}

Based on the findings above, write a brief IMPRESSION section only.
The impression should:
- Summarize the key findings in 2-3 sentences
- State the most likely diagnoses
- Note any recommendations if appropriate
- Use standard radiological report language

Do NOT repeat the findings. Write the impression only."""
            }
        ],
        max_tokens=200
    )

    return response.choices[0].message.content.strip()\
    .replace("IMPRESSION:", "").strip()


def generate_explanation(image_path, detection_results,
                          heatmap_pil=None, threshold=0.3):
    """
    Main function — generates full radiological explanation.

    Args:
        image_path:        path to chest X-ray
        detection_results: dict from detection.predict()
        heatmap_pil:       PIL Image of best heatmap (optional)
        threshold:         detection threshold

    Returns:
        dict with keys:
            findings, impression, full_report,
            detection_summary, success
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "success":    False,
            "error":      "GROQ_API_KEY not found in .env file",
            "findings":   "",
            "impression": "",
            "full_report": ""
        }

    client = Groq(api_key=api_key)

    try:
        # Encode images
        image_b64   = _encode_image_base64(image_path)
        heatmap_b64 = None
        if heatmap_pil is not None:
            heatmap_b64 = _encode_image_base64(heatmap_pil)

        detection_summary = _format_detection_summary(
            detection_results, threshold
        )

        print("  Generating findings...")
        findings = generate_findings(
            client, image_b64, detection_summary, heatmap_b64
        )

        print("  Generating impression...")
        impression = generate_impression(
            client, findings, detection_summary
        )

        full_report = f"FINDINGS:\n{findings}\n\nIMPRESSION:\n{impression}"

        return {
            "success":            True,
            "findings":           findings,
            "impression":         impression,
            "full_report":        full_report,
            "detection_summary":  detection_summary
        }

    except Exception as e:
        return {
            "success":    False,
            "error":      str(e),
            "findings":   "",
            "impression": "",
            "full_report": ""
        }
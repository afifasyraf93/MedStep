import os
import base64
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv("api.env")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
GROQ_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_TOKENS  = 512
# ──────────────────────────────────────────────────────────────────────────────

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def format_detections(detections):
    """Convert detections dict to readable string for prompt."""
    # TODO: build a string listing detected pathologies
    # only include pathologies where probability >= 0.5
    # format: "Detected: pneumonia (0.92), pleural_effusion (0.78)"
    # if nothing detected: "No pathologies detected"
    detected = [
        f"{p} ({detections[p]:.2f})"
        for p in detections
        if detections[p] >= 0.5
    ]
    if detected:
        return "Detected: " + ", ".join(detected)
    return "No pathologies detected"


def generate_findings(image_base64, detections_text):
    """Generate the Findings section of the radiology report."""
    prompt = f"""You are an AI radiology assistant helping medical students learn.

    Detection results: {detections_text}

    Describe the radiographic findings visible in this chest X-ray in 2-3 sentences.
    Use proper medical terminology.
    Focus only on observable features — do NOT state a diagnosis yet.
    Mention relevant anatomical regions and any visible abnormalities."""

    # TODO: call client.chat.completions.create
    # model=GROQ_MODEL
    # messages with role "user" containing:
    #   - image_url content with base64 image
    #   - text content with prompt
    # max_tokens=MAX_TOKENS
    # return response.choices[0].message.content.strip()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }],
        max_tokens=MAX_TOKENS
    )
    return response.choices[0].message.content.strip()


def generate_impression(findings_text, detections_text):
    """Generate the Impression section based on findings."""
    prompt = f"""You are an AI radiology assistant helping medical students learn.

    Radiographic findings: {findings_text}
    Detection results: {detections_text}

    Based on the findings above, provide a clinical impression in 2-3 sentences.
    State the likely diagnosis with supporting evidence from the findings.
    Use professional medical language appropriate for a radiology report."""

    # TODO: call client.chat.completions.create
    # text-only this time — no image needed
    # same model and max_tokens
    # return response.choices[0].message.content.strip()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=MAX_TOKENS
    )
    return response.choices[0].message.content.strip()


def generate_report(image_path, detections):
    """Generate complete radiology report with findings and impression.

    Args:
        image_path: path to the chest X-ray image file
        detections: dict of {pathology: probability}

    Returns:
        dict with keys: "findings", "impression", "detections_text"
    """
    # TODO: encode image to base64
    # TODO: format detections to text
    # TODO: call generate_findings
    # TODO: call generate_impression using findings result
    # TODO: return dict with findings, impression, detections_text
    image_base64     = encode_image_to_base64(image_path)
    detections_text  = format_detections(detections)
    findings         = generate_findings(image_base64, detections_text)
    impression       = generate_impression(findings, detections_text)
    return {
        "findings":        findings,
        "impression":      impression,
        "detections_text": detections_text
    }
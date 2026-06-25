import os
import time
import requests
from PIL import Image

# Hugging Face Free Public Inference API endpoint for CLIP
API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch16"

def analyze_spacecraft_image(image_path):
    """
    Sends the target image to Hugging Face's free inference infrastructure
    to calculate zero-shot classifications without crashing Render's RAM.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target processing image not found at: {image_path}")

    # Standardize image layout
    with Image.open(image_path) as img:
        img.convert("RGB").save(image_path)

    # Our exact structural validation labels
    CANDIDATE_LABELS = [
        "solar panels",
        "a parabolic satellite dish antenna",
        "a camera lens or optical sensor",
        "rocket thrusters or propulsion nozzles",
        "background noise"
    ]

    # Read the image bits to transmit over the network
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    payload = {
        "inputs": img_bytes,
        "parameters": {"candidate_labels": CANDIDATE_LABELS}
    }

    # Attempt to query the public API container (handles cold starts safely)
    for attempt in range(3):
        response = requests.post(API_URL, json=payload)
        result = response.json()
        
        # If the model is sleeping on Hugging Face, it tells us to wait a few seconds
        if isinstance(result, dict) and "estimated_time" in result:
            time.sleep(5)
            continue
        break

    if not isinstance(result, list):
        raise RuntimeError(f"Hugging Face API Error: {result}")

    # Parse predictions exactly how the rest of your app expects them
    parsed_predictions = {}
    for prediction in result:
        confidence_percentage = round(prediction['score'] * 100, 2)
        if confidence_percentage >= 15.0:
            parsed_predictions[prediction['label']] = confidence_percentage

    return parsed_predictions
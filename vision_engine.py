import os
from PIL import Image
from transformers import pipeline

# Load a tiny, ultra-lightweight version of the CLIP pipeline
detector = pipeline("zero-shot-image-classification", model="hf-internal-testing/tiny-random-clip")

def analyze_spacecraft_image(image_path):
    """
    Analyzes the image locally using a memory-optimized CLIP model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target processing image not found at: {image_path}")

    # Standardize image layout
    with Image.open(image_path) as img:
        img.convert("RGB").save(image_path)

    # Classification labels
    CANDIDATE_LABELS = [
        "solar panels",
        "a parabolic satellite dish antenna",
        "a camera lens or optical sensor",
        "rocket thrusters or propulsion nozzles",
        "background noise"
    ]

    # Run local inference
    result = detector(image_path, candidate_labels=CANDIDATE_LABELS)

    # Parse predictions
    parsed_predictions = {}
    for label, score in zip(result['labels'], result['scores']):
        confidence_percentage = round(score * 100, 2)
        if confidence_percentage >= 15.0:
            parsed_predictions[label] = confidence_percentage

    return parsed_predictions
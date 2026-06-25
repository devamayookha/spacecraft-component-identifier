import os
from PIL import Image
from transformers import pipeline

print("Loading CLIP AI Model... (This may take a minute on the first run)")

# Ensure the model utilizes CPU safely and cleanly without warning flags
try:
    detector = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error initializing CLIP model: {e}")
    detector = None

def analyze_spacecraft_image(image_path):
    """
    Opens the uniquely saved image file and passes it to the local 
    neural network for zero-shot component extraction.
    """
    if detector is None:
        raise RuntimeError("The CLIP classification model is not loaded or initialized.")

    # 1. Open the FRESH file path provided by app.py
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target processing image not found at: {image_path}")
        
    image = Image.open(image_path).convert("RGB")

    # 2. Strict aerospace candidate labels matching our backend inference dictionary
    CANDIDATE_LABELS = [
        "solar panels",
        "a parabolic satellite dish antenna",
        "a camera lens or optical sensor",
        "rocket thrusters or propulsion nozzles",
        "background noise"
    ]

    # 3. Process the actual image object through the model
    raw_results = detector(image, candidate_labels=CANDIDATE_LABELS)

    # 4. Filter out weak detections (anything lower than 15%) to prevent noise
    parsed_predictions = {}
    for prediction in raw_results:
        confidence_percentage = round(prediction['score'] * 100, 2)
        if confidence_percentage >= 15.0:
            parsed_predictions[prediction['label']] = confidence_percentage

    return parsed_predictions
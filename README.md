# Spacecraft Component Identifier & Mission Classifier

A lightweight, locally-hosted computer vision application that extracts engineering subsystems from aerospace telemetry images and uses heuristic logic to infer their operational mission profiles.

## Project Overview & Motivation

As an AI & Data Science student exploring the intersection of computer vision and aerospace engineering, I built this prototype to bridge the gap between low-level visual features and high-level deductive reasoning.

In aerospace design, a spacecraft's physical architecture is directly dictated by its mission requirements. This application automates the process of analyzing structural hardware configurations to classify whether an unlabelled spacecraft is optimized for orbital communication, deep-space exploration, or planetary observation.

---

## Tech Stack

- **Backend Framework:** Flask (Python 3.8+)
- **Deep Learning Engine:** Hugging Face Transformers (OpenAI's CLIP Model: `clip-vit-base-patch32`)
- **Image Processing:** Pillow (PIL)
- **Frontend Architecture:** Vanilla HTML5, CSS3, Asynchronous JavaScript (Fetch API)

---

## How It Works

1. **Visual Grounding:** The user uploads a spacecraft image via a responsive, dark-theme web dashboard.
2. **Zero-Shot Feature Extraction:** The Flask backend passes the image stream to a locally cached CLIP model, which calculates cross-modal matching probabilities against explicit hardware subcomponents.
3. **Heuristic Inference Engine:** A rule-based Python layer processes the resulting text-image matching matrix, scoring the system's design traits to deduce the overarching operational mission profile.
4. **Reactive UI Rendering:** Asynchronous JavaScript updates the viewport with percentage matches and engineering explanations without requiring a page reload.

---

## Current Scope & Limitations (Version 1)

This prototype is calibrated to recognize four core aerospace subsystems:

- Solar panels
- Parabolic satellite dish antennas
- Camera lenses / optical sensors
- Rocket thrusters / propulsion nozzles

> **Note:** Because the pipeline uses a closed zero-shot classification matrix, the application is scoped to operate reliably only on satellites and rocket propulsion stages. Uploading non-aerospace images (e.g., everyday objects or animals) will produce false positives, as the model is forced to distribute confidence scores across the nearest available structural match.

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/spacecraft-component-identifier.git
cd spacecraft-component-identifier
```

### 2. Install Dependencies

Ensure you have Python 3.8+ installed, then run:

```bash
pip install Flask Pillow torch transformers
```

### 3. Launch the Server

```bash
python app.py
```

### 4. Open the Interface

Navigate to `http://127.0.0.1:5000` in your web browser.

---

## Project Presentation & UI Structure

The application features a minimalist, dark-theme terminal interface. When an image is supplied:

- A dynamic telemetry parsing state is triggered (`loading...`)
- The interface resolves into a structured breakdown displaying the inferred **Mission Profile**
- A deductive reasoning paragraph justifies the classification based on component groupings
- Individual subsystems are listed with their corresponding confidence scores (e.g., `Rocket Thrusters: 97.04% Match`)

---

## Future Enhancements (Version 2 Roadmap)

- **Confidence Calibration (Anti-False Positive Filter):** Integrate a neutral "Background / Everyday Earthbound Object" label into the classification array so the model can safely reject non-aerospace images.
- **Bounding-Box Object Detection:** Transition from zero-shot global image classification to localized detection by fine-tuning a small-footprint model like YOLO nano.
- **Expanded Hardware Taxonomy:** Broaden the component array to identify specialized deep-space structures, including star trackers, reaction wheels, and magnetometer booms.

---

## Author

**B R Devamayookha**  
AI & Data Science Student  
[GitHub](https://github.com/devamayookha) · [LinkedIn](https://linkedin.com/in/b-r-devamayookha-627305375)
import os
import time  # <--- Added for unique filenames
from flask import Flask, render_template, request, jsonify
from PIL import Image
from vision_engine import analyze_spacecraft_image
from inference_engine import infer_mission

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'spacecraft_image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['spacecraft_image']
    print("DEBUG: Received file named:", file.filename)
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        # Prevent caching and overlaps by creating a unique filename
        filename, file_extension = os.path.splitext(file.filename)
        unique_filename = f"{filename}_{int(time.time())}{file_extension}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(file_path)
        
        try:
            with Image.open(file_path) as img:
                img.verify() 
        except Exception:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': 'Invalid image file'}), 400

        try:
            # 1. Vision Analysis (Now points to a completely unique path)
            analysis_results = analyze_spacecraft_image(file_path)
            
            # 2. Mission Inference Logic
            mission_results = infer_mission(analysis_results)
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        finally:
            # Safely clean up the exact file we just processed
            if os.path.exists(file_path):
                os.remove(file_path)

        return jsonify({
            'status': 'Success',
            'detected_components': analysis_results,
            'mission_inference': mission_results
        })

if __name__ == '__main__':
    app.run(debug=True)
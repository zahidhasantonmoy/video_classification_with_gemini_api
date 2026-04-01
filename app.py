import os
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from PIL import Image
from decord import VideoReader, cpu

# API Key Setup
API_KEY = "AIzaSyBbCawu7bH5Uz9ldNEaIjToLS_ZiV0swbc"
genai.configure(api_key=API_KEY)

app = Flask(__name__)

def extract_frames(video_path, num_frames=4):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(frame) for frame in frames]

def get_working_model():
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
    
    for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
        if preferred in available_models:
            return preferred
    return available_models[0] if available_models else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected video"})

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_file.save(temp_video.name)
            temp_path = temp_video.name

        frames = extract_frames(temp_path)
        model_name = get_working_model()
        
        if not model_name:
            return jsonify({"error": "No supported Gemini models found."})

        model = genai.GenerativeModel(model_name)

        prompt = """
        Analyze these frames from a food video. Return ONLY valid JSON and nothing else.
        Structure:
        {
          "cuisine": "Bangladeshi / Indian / Asian / Others",
          "taste_tags": ["spicy", "savory", "sweet", "creamy"],
          "food_type": "rice / noodles / biryani / curry / snack",
          "meal_type": "breakfast / lunch / dinner / snack",
          "dining_mode": "restaurant / homemade / street_food"
        }
        """
        
        response = model.generate_content(frames + [prompt])
        
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        result = json.loads(raw_text.strip())
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
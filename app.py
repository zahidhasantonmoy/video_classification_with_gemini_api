import os
import json
import tempfile
import numpy as np
import cv2  # decord এর বদলে OpenCV ব্যবহার করা হচ্ছে
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from PIL import Image

# API Key Environment Variable থেকে নেওয়া হচ্ছে
API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

app = Flask(__name__)

def extract_frames(video_path, num_frames=3):
    """
    OpenCV ব্যবহার করে ফ্রেম এক্সট্রাক্ট করা হচ্ছে।
    এটি পুরো ভিডিও র‍্যামে লোড করে না, তাই সার্ভার ক্র্যাশ করবে না।
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pil_frames = []

    if total_frames > 0:
        indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        for idx in indices:
            # নির্দিষ্ট ফ্রেমে গিয়ে শুধু ওই ছবিটি রিড করা
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # OpenCV BGR ফরম্যাটে ছবি দেয়, সেটাকে RGB তে কনভার্ট করা
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                # মেমোরি বাঁচাতে সাইজ ছোট করা হচ্ছে
                img.thumbnail((512, 512)) 
                pil_frames.append(img)
                
    cap.release()
    return pil_frames

def get_working_model():
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
        if preferred in available_models:
            return preferred
    return available_models[0] if available_models else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if not API_KEY:
        return jsonify({"error": "API Key is missing in environment variables!"})

    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No selected video"})

    temp_path = ""
    try:
        # ভিডিও সাময়িকভাবে সেভ করা
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            video_file.save(temp_video.name)
            temp_path = temp_video.name

        frames = extract_frames(temp_path)
        
        if not frames:
            return jsonify({"error": "Could not read video frames. Try a different MP4 file."})

        model_name = get_working_model()
        if not model_name:
            return jsonify({"error": "No supported Gemini models found."})

        model = genai.GenerativeModel(model_name)
        prompt = """
        Analyze these frames from a food video. Return ONLY valid JSON and nothing else.
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
        raw_text = raw_text.replace("```json", "")
        raw_text = raw_text.replace("```", "")
            
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
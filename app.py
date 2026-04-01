import os
import json
import base64
from io import BytesIO
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from PIL import Image

API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

app = Flask(__name__)

# Flask-এ রিকোয়েস্ট সাইজ লিমিট বাড়ানো হলো (যাতে ৩টি Base64 ছবি আসতে পারে)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB

def get_working_model():
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    for preferred in ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']:
        if preferred in available_models:
            return preferred
    return available_models[0] if available_models else None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_frames', methods=['POST'])
def analyze_frames():
    if not API_KEY:
        return jsonify({"error": "API Key is missing in environment variables!"})

    data = request.json
    if not data or 'frames' not in data:
        return jsonify({"error": "No frames received from frontend."})

    base64_frames = data['frames']
    if not base64_frames or len(base64_frames) == 0:
        return jsonify({"error": "Empty frames array."})

    try:
        pil_frames = []
        for b64_str in base64_frames:
            # Base64 স্ট্রিং থেকে "data:image/jpeg;base64," অংশটুকু বাদ দেওয়া
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            
            image_data = base64.b64decode(b64_str)
            img = Image.open(BytesIO(image_data))
            # মেমোরি ও টোকেন বাঁচাতে আরও ছোট করা হলো
            img.thumbnail((300, 300))
            pil_frames.append(img)

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
        response = model.generate_content(pil_frames + [prompt])
        
        raw_text = response.text.strip()
        raw_text = raw_text.replace("```json", "")
        raw_text = raw_text.replace("```", "")
            
        result = json.loads(raw_text.strip())
            
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
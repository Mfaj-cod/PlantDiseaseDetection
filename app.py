import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
from modules.logging import logger
from modules.create_model import create_model_loss_optim
from modules.utils import ConvertToRGB

try:
    import google.generativeai as genai
except ImportError: 
    genai = None


app = Flask(__name__)


# model architecture
num_classes = 23  #the number of classes
model, _, _ = create_model_loss_optim(num_classes)

# Loading saved weights
model_path = "artifacts/model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
logger.info(f"Loaded model weights from {model_path}")

# to evaluation mode
model.eval()

# transforms using same mean and std as used during training
transform = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# label mapping
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


# Loading environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME")
gemini_model = None




if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        logger.info(f"Configured Gemini model '{GEMINI_MODEL_NAME}'.")
    except Exception as exc:
        gemini_model = None
        logger.exception("Failed to configure Gemini model: %s", exc)

elif not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not set. Remedy suggestions will be unavailable.")

elif genai is None:
    logger.warning("google-generativeai package not installed. Remedy suggestions will be unavailable.")





def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        logger.info(f"Predicted class: {class_names[predicted.item()]}")
        return class_names[predicted.item()]




def get_disease_remedy(disease_name: str):
    # Only proceed when there's an actual prediction
    if not disease_name or disease_name == "No Prediction" or disease_name.strip() == "":
        return None  # Do NOT trigger Gemini

    if gemini_model is None:
        return "Remedy suggestions unavailable. Check Gemini configuration."

    prompt = (
        "You are an agricultural expert. In 2-4 sentences, describe an actionable treatment and "
        "prevention plan for the plant disease '{disease}'. Include specific chemical or organic "
        "treatments, cultural practices, and preventive tips and if the disease includes 'healthy', don't recommend any chemical just give some suggestions to keep it healthy. All in Hindi and simple language "
        "because it's for Indian farmers."
    ).format(disease=disease_name.replace("_", " "))

    try:
        response = gemini_model.generate_content(prompt)
        text = (getattr(response, "text", None) or "").strip()

        if not text:
            logger.warning("Gemini returned an empty remedy for %s.", disease_name)
            return "No remedy suggestion returned by Gemini."
        
        # Convert **text** to <strong>text</strong>
        parts = text.split("**")
        formatted_text = ""
        toggle = True
        for part in parts:
            if toggle:
                formatted_text += part
            else:
                formatted_text += f"<strong>{part}</strong>"
            toggle = not toggle

        return formatted_text
    
    except Exception as exc:
        logger.exception("Gemini remedy generation failed for %s: %s", disease_name, exc)
        return "Unable to retrieve remedy suggestion at this time."



@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded.", remedy=None)
        
        file = request.files['file']
        if not file:
            return render_template('index.html', prediction="No image selected.", remedy=None)
        
        img_bytes = file.read()
        prediction = predict_image(img_bytes).replace('_', ' ')
        logger.info(f"Image prediction: {prediction}")

        remedy = get_disease_remedy(prediction)
        return render_template('index.html', prediction=prediction, remedy=remedy)
    
    logger.info("problem in rendering uploaded page.")
    return render_template('index.html', prediction=None, remedy=None)




@app.route('/capture', methods=['POST'])
def capture_predict():
    file = request.files.get('file')
    if not file:
        return render_template('index.html', prediction="No image captured.", remedy=None)
    
    prediction = predict_image(file.read()).replace('_', ' ')
    logger.info(f"Captured image prediction: {prediction}")
    
    remedy = get_disease_remedy(prediction)
    return render_template('index.html', prediction=prediction, remedy=remedy)




if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)

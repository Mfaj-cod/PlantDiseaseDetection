import os
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
from modules.logging import logger
from modules.create_model import create_model_loss_optim  # <-- Import your function

# Initialize Flask app
app = Flask(__name__)


# Create the same model architecture
num_classes = 23  #the number of classes you trained with
model, _, _ = create_model_loss_optim(num_classes)

# Load your saved weights
model_path = "artifacts/model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
logger.info(f"Loaded model weights from {model_path}")
# Set to evaluation mode
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Define label mapping (adjust to your dataset)
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']


def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        logger.info(f"Predicted class: {class_names[predicted.item()]}")
        return class_names[predicted.item()]



@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded.")
        file = request.files['file']
        if not file:
            return render_template('index.html', prediction="No image selected.")
        
        img_bytes = file.read()
        prediction = predict_image(img_bytes)
        logger.info(f"Image prediction: {prediction}")
        return render_template('index.html', prediction=prediction)
    
    logger.info("problem in rendering uploaded page.")
    return render_template('index.html', prediction=None)


@app.route('/capture', methods=['POST'])
def capture_predict():
    file = request.files.get('file')
    if not file:
        return render_template('index.html', prediction="No image captured.")
    prediction = predict_image(file.read())
    logger.info(f"Captured image prediction: {prediction}")
    return render_template('index.html', prediction=prediction)



if __name__ == '__main__':
    logger.info("Starting Flask app...")
    print(app.url_map)
    app.run(debug=True)

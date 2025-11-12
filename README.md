<div align="center">

# 🌿✨ **PLANT DISEASE DETECTION**  
### _AI-powered Deep Learning Web App for Leaf Disease Classification_  

🧠 Built with **PyTorch**, **Flask**, and **Computer Vision**  
📸 Upload or Capture live plant images and get instant predictions  

![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-%23000.svg?logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![UI](https://img.shields.io/badge/UI-Dark%20Mode-black)

</div>

---

<div align="center">
  <img src="https://user-images.githubusercontent.com/11388939/172873216-2e004c1e-81da-4e4f-a00f-2d5648a52764.gif" width="80%">
</div>

---

## 🧭 **Overview**

This project is a **Deep Learning–powered web application** that detects **plant leaf diseases** in real time 🌱.  
It uses a **Convolutional Neural Network (CNN)** trained on plant disease datasets and provides an interactive **Flask web interface** for predictions.

---

## ⚙️ **Tech Stack**

| 🧩 Category | 🛠️ Tools |
|-------------|----------|
| 💻 **Frontend** | HTML5, CSS3, JavaScript (Webcam API) |
| ⚙️ **Backend** | Flask |
| 🧠 **AI Model** | PyTorch (CNN) |
| 🧾 **Data Processing** | NumPy, PIL, TorchVision |
| 🎨 **Design** | Responsive Dark UI |
| ☁️ **Deployment Ready** | Render / Streamlit Cloud / Hugging Face Spaces |

---

## 🧠 **Model Architecture**
<ol>
    <li>Input (3×224×224)</li>
    <li>Conv Block 1
        <ol>
            <li>Conv2D(3 → 16, 3×3)</li>
            <li>ReLU</li>
            <li>MaxPool(2×2)</li>
        </ol>
    </li>
    <li>Conv Block 2
        <ol>
            <li>Conv2D(16 → 32, 3×3)</li>
            <li>ReLU</li>
            <li>MaxPool(2×2)</li>
        </ol>
    </li>
    <li>Conv Block 3
        <ol>
            <li>Conv2D(32 → 64, 3×3)</li>
            <li>ReLU</li>
            <li>MaxPool(2×2)</li>
        </ol>
    </li>
    <li>Classifier
        <ol>
            <li>Flatten</li>
            <li>Dropout(0.5)</li>
            <li>Linear(642828 → 500)</li>
            <li>ReLU</li>
            <li>Dropout</li>
            <li>Linear(500 → num_classes)</li>
        </ol>
    </li>
</ol>

---

📉 **Loss:** CrossEntropyLoss  
⚡ **Optimizer:** Adam (lr=0.001)  
🎯 **Accuracy:** ~93% Validation Accuracy  

---

## 🚀 **Web App Features**

🌾 Upload an image from your device  
📸 Capture a live image using your webcam  
🤖 Get instant disease predictions powered by CNN  
🌗 Beautiful dark UI design  
💬 Ready for cloud deployment  

---

## 🌿 Training Highlights

| Metric                 | Value         |
| :--------------------- | :------------ |
| 🧮 Training Accuracy   | 95%           |
| 🧾 Validation Accuracy | 93%           |
| 🧠 Loss Function       | Cross-Entropy |
| ⚡ Optimizer           | Adam          |
| 🕒 Epochs              | 8             |

---


## 🎯 Live Camera Mode
<div align="center">

<p style="font-size: 1.1rem; margin: 12px 0;">💡 Take a live photo using your webcam directly in the browser:</p>

<div style="background: #0b1220; padding: 20px; border-radius: 10px; max-width: 600px; margin: 20px auto; color: #dbe9d9;">
    <p style="margin: 10px 0;"><strong>Click 📸 Capture Photo</strong></p>
    <p style="margin: 10px 0;"><strong>Then click Predict from Camera</strong></p>
    <p style="margin: 10px 0;"><strong>Get instant results using your trained CNN model 🚀</strong></p>
</div>

</div>


---

## ❤️ Acknowledgements
<div class="acknowledgements" style="background:#0b1220;color:#dbe9d9;padding:16px;border-radius:10px;max-width:760px;margin:12px auto;font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
    <h3 style="margin:0 0 8px 0;text-align:center;">🌟 Special thanks to</h3>
    <ul style="list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:10px;">
        <li style="display:flex;align-items:center;gap:12px;padding:10px;border-radius:8px;background:linear-gradient(90deg,rgba(255,255,255,0.02),transparent);">
            <span style="font-size:1.4rem;">🌱</span>
            <div>
                <strong>WorldQuant University</strong>
                <div style="font-size:0.95rem;color:#a9c4a7;">for the Deep Learning Foundations</div>
            </div>
        </li>
        <li style="display:flex;align-items:center;gap:12px;padding:10px;border-radius:8px;background:linear-gradient(90deg,rgba(255,255,255,0.02),transparent);">
            <span style="font-size:1.4rem;">🔥</span>
            <div>
                <strong>PyTorch</strong>
                <div style="font-size:0.95rem;color:#a9c4a7;">for making model building intuitive</div>
            </div>
        </li>
        <li style="display:flex;align-items:center;gap:12px;padding:10px;border-radius:8px;background:linear-gradient(90deg,rgba(255,255,255,0.02),transparent);">
            <span style="font-size:1.4rem;">🧩</span>
            <div>
                <strong>Flask</strong>
                <div style="font-size:0.95rem;color:#a9c4a7;">for the minimalistic yet powerful web backend</div>
            </div>
        </li>
        <li style="display:flex;align-items:center;gap:12px;padding:10px;border-radius:8px;background:linear-gradient(90deg,rgba(255,255,255,0.02),transparent);">
            <span style="font-size:1.4rem;">👩‍💻</span>
            <div>
                <strong>You</strong>
                <div style="font-size:0.95rem;color:#a9c4a7;">for taking the time to make plants healthier <span style="margin-left:6px;">🌿</span></div>
            </div>
        </li>
    </ul>
</div>

---

## 🌟 If you like this project...

⭐ Star it on GitHub
🍴 Fork it
🚀 Share it

<img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge&logo=pytorch"/> <img src="https://img.shields.io/badge/Web-Framework-000000?style=for-the-badge&logo=flask"/> <img src="https://img.shields.io/badge/Frontend-HTML/CSS/JS-yellow?style=for-the-badge&logo=html5"/> </div> ```



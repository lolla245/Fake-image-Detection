from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, template_folder='.')

# Load model (make sure path correct)
model = load_model("model/fake_image_model.h5")

IMG_SIZE = 128

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    return "FAKE IMAGE" if prediction > 0.5 else "REAL IMAGE"


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']

    path = os.path.join("static", file.filename)
    file.save(path)

    result = predict_image(path)

    image_path = path.replace("\\", "/")

    return render_template("index.html", prediction=result, image_path=image_path)


# IMPORTANT for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # change 10000 → 5000
    app.run(host="0.0.0.0", port=port)        # remove debug=True
# Fake Image Detection

A web app that classifies uploaded images as **REAL** or **FAKE (AI-generated)** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras, served through a Flask web interface.

## Features

- Upload an image (PNG, JPG, WEBP) through a clean web UI
- CNN model predicts whether the image is real or AI-generated
- Instant visual result with confidence-based styling
- Drag-and-drop or click-to-upload support

## Tech Stack

- **Model:** TensorFlow / Keras (CNN with Conv2D, MaxPooling, Dense, Dropout layers)
- **Backend:** Flask
- **Image processing:** OpenCV, NumPy
- **Frontend:** HTML, CSS, vanilla JavaScript

## Project Structure

```
Fake-image-Detection/
├── app.py                  # Flask application
├── index.html              # Frontend UI
├── Imgdetect.ipynb         # Model training notebook
├── model/
│   └── fake_image_model.h5 # Trained CNN model
├── Dataset/                # Training images (real/fake categories)
├── static/                 # Uploaded images (runtime)
├── requirements.txt        # Python dependencies
└── runtime.txt             # Python runtime version (for deployment)
```

## How It Works

1. **Data preparation** — Images from labeled folders (`real`, `fake`) are loaded, resized to 128×128, and normalized to a 0–1 pixel range.
2. **Model** — A CNN with 3 convolution + pooling blocks extracts visual features, followed by dense layers with dropout to reduce overfitting. A final sigmoid output gives a probability between 0 (real) and 1 (fake).
3. **Training** — The model is trained with the Adam optimizer and binary crossentropy loss, validated on a held-out test split.
4. **Inference** — `app.py` loads the saved model, preprocesses any uploaded image the same way as training data, and returns a REAL/FAKE prediction.

## Setup & Installation

### Prerequisites
- Python 3.10 or 3.11 (TensorFlow does not yet support the very latest Python releases)

### Steps

```bash
# Clone the repo
git clone https://github.com/lolla245/Fake-image-Detection.git
cd Fake-image-Detection

# Create a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Then open your browser at:
```
http://localhost:5000
```

## Usage

1. Open the web app in your browser.
2. Click or drag an image into the upload zone.
3. Click **Analyze Image**.
4. View the prediction — REAL or FAKE — along with the uploaded image.

## Model Training

The model was trained in `Imgdetect.ipynb`, which covers:
- Loading and preprocessing the image dataset
- Building the CNN architecture
- Training for 5 epochs with an 80/20 train-test split
- Plotting training vs. validation accuracy
- Saving the trained model as `fake_image_model.h5`

To retrain the model with your own dataset, update the `DATA_DIR` and `categories` variables in the notebook and re-run all cells.

## Notes

- The `venv/` folder is excluded from version control via `.gitignore` — install dependencies fresh using `requirements.txt`.
- Uploaded images are temporarily stored in the `static/` folder.
- For deployment (e.g., on Render), the app reads the `PORT` environment variable and binds to `0.0.0.0`.

## Disclaimer

This model is trained on a specific dataset and is intended for educational/demonstration purposes. Prediction accuracy depends on the training data and may not generalize to all types of real or AI-generated images.

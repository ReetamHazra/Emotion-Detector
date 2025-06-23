# 🎧 Emotion Detector from Audio

This project is a deep learning-based application that detects human emotions from audio files using a trained neural network. It features a Keras-based model for classification and a Flask web interface for interaction.

## 🚀 Features

- Detects emotions like **angry**, **happy**, **sad**, **neutral**, etc.
- Extracts relevant features from audio using `librosa`.
- Simple web interface for uploading `.wav` files.
- Backend powered by Flask and Keras.
- Model trained on [RAVDESS/TESS/CREMA-D/Custom] datasets.

## 🧠 Technologies Used

- Python 3.8+
- Flask
- Keras / TensorFlow
- Librosa
- NumPy, Pandas
- Scikit-learn
- HTML/CSS for basic frontend

## 🗂️ Project Structure

```
emotion_detector/
├── static/                # CSS, JS files
├── templates/             # HTML templates (index.html, result.html)
├── model/                 # Saved model (.h5) file
├── app.py                 # Flask application
├── feature_extraction.py  # Audio preprocessing & feature extraction
├── utils.py               # Helper functions (e.g., prediction logic)
├── requirements.txt
└── README.md
```

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/emotion_detector.git
cd emotion_detector
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask App

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser to use the app.

## 📊 Emotion Classes

The model is trained to detect the following emotions:

- 😠 Angry
- 😢 Sad
- 😐 Neutral
- 😄 Happy
- 😲 Surprise
- 😨 Fear
- 🤢 Disgust
- 😴 Calm

*(Dependent on dataset used)*

## 📈 Model Training (Optional)

If you want to retrain the model:

```bash
python train_model.py
```

Make sure you have the dataset in the correct format and path.

## 📌 Requirements

See `requirements.txt`. Major packages include:

- Flask
- Keras
- librosa
- numpy
- scikit-learn

## 🙌 Acknowledgements

- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [Librosa](https://librosa.org/)
- [Keras](https://keras.io/)
- [Flask](https://flask.palletsprojects.com/)

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

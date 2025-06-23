from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import librosa
import os

app = Flask(__name__)
model = tf.keras.models.load_model("emotion_model.h5")

emotion_labels = ['angry', 'happy', 'neutral', 'sad', 'calm', 'fearful', 'Disgust', 'Surprized']

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        audio = request.files['audio']
        filepath = os.path.join('static', audio.filename)
        audio.save(filepath)

        features = extract_features(filepath)
        features = np.expand_dims(features, axis=0)
        prediction = model.predict(features)
        emotion = emotion_labels[np.argmax(prediction)]

        return render_template('index.html', prediction=emotion, filename=audio.filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

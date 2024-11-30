import os
import numpy as np
import tensorflow as tf
import pickle
import wave
import pyaudio
import librosa
import joblib
import cv2
from flask import Flask, request, jsonify, render_template, url_for
import speech_recognition as sr
from threading import Thread
from gtts import gTTS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['AUDIO_FOLDER'] = 'static/audio'

if not os.path.exists(app.config['AUDIO_FOLDER']):
    os.makedirs(app.config['AUDIO_FOLDER'])


def generate_audio():
    tts_depressed = gTTS("You show symptoms of depression. It is recommended to consult a mental health professional.", lang='en')
    tts_depressed.save(os.path.join(app.config['AUDIO_FOLDER'], "depressed.mp3"))

    tts_not_depressed = gTTS("You do not show symptoms of depression. Continue to monitor your mental health.", lang='en')
    tts_not_depressed.save(os.path.join(app.config['AUDIO_FOLDER'], "not_depressed.mp3"))

generate_audio()

# Load your models
print("Loading CNN model...")
cnn_model = tf.keras.models.load_model('my_cnn_model.h5')
print("CNN model loaded successfully.")

print("Loading SVM model...")
try:
    with open('my_svm.pkl', 'rb') as f:
        svm_model = joblib.load(f)
    print("SVM model loaded successfully..")
    if hasattr(svm_model, 'support_'):
        print("SVC model is fitted.")
    else:
        print("SVC model is NOT fitted.")
except Exception as e:
    print("Error loading SVM model:", e)


def process_audio(file_path):
    # Load audio file for model prediction
    print(f"Processing audio file: {file_path}")
    y, sr = librosa.load(file_path, sr=None)

    # Convert to mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # Convert to dB
    S_db = librosa.power_to_db(S, ref=np.max)

    # Resize the spectrogram to ensure it is 512x512
    # Note: S_db might not be equal to (512, 512) initially, you can pad or resize accordingly
    S_resized = cv2.resize(S_db, (512, 512))  # Resize to 512x512 with OpenCV (assuming input is 2D)

    # Now, convert this to a 3-channel image
    cnn_input = np.stack([S_resized] * 3, axis=-1)  # Shape now should be (512, 512, 3)

    # Ensure the shape needed for the CNN input
    cnn_input = np.expand_dims(cnn_input, axis=0)  # Shape: (1, 512, 512, 3)
    print(f"Input shape for CNN: {cnn_input.shape}")

    # Prepare input for SVM
    svm_input = S_db.flatten()  # You may need to adjust this to ensure it matches the training input
    # Ensure svm_input has the correct shape for SVM prediction
    print(f"Input shape for SVM before flattening: {S_db.shape}")  # Debug: shape before flattening
    if svm_input.shape[0] > 1000:
        # If there are more than 1000 features, truncate or extract relevant features
        svm_input = svm_input[:1000]  # Keep only the first 1000 features
    elif svm_input.shape[0] < 1000:
        # If there are less than 1000 features, we need to pad
        padding = np.zeros(1000 - svm_input.shape[0])
        svm_input = np.concatenate((svm_input, padding))  # Pad with zeros up to 1000

    # Reshape to (1, 1000) for SVC prediction
    # svm_input = svm_input.reshape(1, -1)  # Shape: (1, 1000)
    print(f"Input shape for SVM after reshaping: {svm_input.shape}")

    print("Checking if SVM model is fitted...")
    try:
        # Attempt to access an attribute to see if the model is fitted
        if hasattr(svm_model, 'support_'):
            print("SVM model is fitted.")
        else:
            print("SVM model is NOT fitted. Please check the fitting process.")
    except Exception as e:
        print("Error during SVM model state check:", e)
    
    # Make predictions
    try:
        print("Making predictions using CNN...")
        cnn_prediction = cnn_model.predict(cnn_input)
        print("CNN prediction made successfully.")
        cnn_prob = cnn_prediction[0][0]
        cnn_prob = np.clip(cnn_prob, 0.01, 0.99) 
        print(cnn_prediction)
        print(cnn_prob)
        
        print("Making predictions using SVM...")
        svm_prediction = svm_model.predict_proba([svm_input])
        print("SVM prediction made successfully.")
        svm_prob = svm_prediction[0][1] 
        print(svm_prediction)
    except Exception as ex:
        print("Error making predictions:", ex)

    return cnn_prob, svm_prob

def record_audio(filename, record_seconds=30):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    paudio = pyaudio.PyAudio()
    stream = paudio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print(".....Recording Started.....")
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * record_seconds)):
        data = stream.read(CHUNK)
        frames.append(data)

    print(".....Recording Ended.....")
    
    stream.stop_stream()
    stream.close()
    paudio.terminate()

    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(paudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

@app.route('/')
def index():
    background_url = url_for('static', filename='orig.jpg')
    return render_template('index.html', background_url=background_url)

@app.route('/depressed')
def depressed():
    audio_url = url_for('static', filename='audio/depressed.mp3')
    return render_template('depressed.html', audio_url=audio_url)


@app.route('/process_questionnaire', methods=['POST'])
def process_questionnaire():
    # Get responses from the form
    responses = {
        "q1": request.form.get('q1'),
        "q2": request.form.get('q2'),
        "q3": request.form.get('q3'),
        "q4": request.form.get('q4'),
        "q5": request.form.get('q5'),
        "q6": request.form.get('q6'),
        "q7": request.form.get('q7'),
        "q8": request.form.get('q8'),
        "q9": request.form.get('q9'),
        "q10": request.form.get('q10'),
    }

    # Initialize score
    score = 0

    # Score calculation based on answers
    response_scores = {
        "Not at all": 0,
        "Several days": 1,
        "More than half the days": 2,
        "Nearly every day": 3,
        "Very good": 0,
        "Good": 1,
        "Poor": 2,
        "Very poor": 3
    }

    for response in responses.values():
        score += response_scores.get(response, 0)

    # Generate evaluation based on score ranges
    if score >= 20:
        evaluation = (
            "Your responses indicate a high level of depressive symptoms. "
            "It’s recommended to seek help from a mental health professional. "
            "Please remember, seeking help is a positive and courageous step."
        )
    elif 10 <= score < 20:
        evaluation = (
            "Your responses suggest moderate depressive symptoms. "
            "You may benefit from support, whether it’s talking to close ones "
            "or seeking professional guidance. Consider reaching out if you feel ready."
        )
    else:
        evaluation = (
            "Your responses indicate a low level of depressive symptoms. "
            "Continue to monitor your well-being, and don't hesitate to reach out "
            "if you ever feel the need to talk to someone."
        )

    # Render the report template with the evaluation
    return render_template('report.html', evaluation=evaluation)


@app.route('/notdepressed')
def notdepressed():
    audio_url = url_for('static', filename='audio/not_depressed.mp3')
    return render_template('notdepressed.html', audio_url=audio_url)

@app.route('/record', methods=['POST'])
def record():
    filename = os.path.join(app.config['UPLOAD_FOLDER'], 'recording.wav')
    
    # Start audio recording in a separate thread
    record_thread = Thread(target=record_audio, args=(filename,))
    record_thread.start()
    
    # Return response immediately (you may want to handle this differently in real applications)
    return jsonify({"message": "Recording started", "status": "success"})

from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.eval() 
classifier = tf.keras.models.load_model('mybert.h5')  # Load your saved Keras classifier

def generate_embedding(text):
    # inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()  # Mean pooling
    return embedding

def predict_depression(text):
    text_embedding = generate_embedding([text])
    prediction = classifier.predict(text_embedding)
    label = (prediction > 0.5).astype(int)
    result = "Depressed" if label == 1 else "Not Depressed"
    print(f"Text: {text}\nPrediction: {result}\nConfidence: {prediction[0][0]:.2f}")
    return prediction[0][0]

@app.route('/upload', methods=['POST'])
def upload():
    # File is already recorded through the record endpoint, so we can process it directly
    wav_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recording.wav')

    # Speech Recognition
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            rec_text = recognizer.recognize_google(audio_data)
            print(f"Recognized Text: {rec_text}")
            # depression_result, confidence = predict_depression(rec_text)
        except sr.UnknownValueError:
            rec_text = "Audio not understood"
        except sr.RequestError as e:
            rec_text = f"Could not request results from Google Speech Recognition service; {e}"

    # Process audio for your models
    cnn_prob, svm_prob = process_audio(wav_file_path)
    bert_prob = predict_depression(rec_text)
    cnn_weight = 0.2
    svm_weight = 0.1
    bert_weight = 0.7

    final_prob_depressed = (cnn_prob * cnn_weight) + (svm_prob * svm_weight) + (bert_prob * bert_weight)
    # Make final prediction based on the weighted probability
    if final_prob_depressed > 0.5:
        final_prediction = 1  # Depressed
    else:
        final_prediction = 0 
    

    return jsonify({'prediction': final_prediction})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)

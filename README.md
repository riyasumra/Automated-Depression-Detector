# Automated-Depression-Detector
An Automated System to Detect Depression Using Speech


**Depression Detection System** is a machine learning project designed to analyze speech data and detect depression using advanced techniques like **Convolutional Neural Networks (CNN)**, **Support Vector Machines (SVM)**, and **DistilBERT** (a transformer-based NLP model). The system leverages deep learning and traditional machine learning to classify speech/text data as **"Depressed"** or **"Non-Depressed"**.



## 🧠 **Features**

- **CNN for Speech Analysis**: Converts audio into spectrograms and classifies them using Convolutional Neural Networks.
- **SVM for Text-Based Analysis**: Uses TF-IDF feature extraction and SVM for classification.
- **DistilBERT**: Fine-tuned for text-based classification using transformer-based deep learning.
- **Multi-Model Integration**: Combines models to achieve robust results across various datasets.
- **Web Interface**: Includes a Flask-based web app for live testing.


## 📊 **Datasets**

- **DAIC-WOZ Depression Dataset**  
  - Includes audio recordings with transcriptions annotated for depression detection.  
  - Download [here](https://dcapswoz.ict.usc.edu/).

- **RAVDESS & SAVEE (Optional)**  
  - Used for additional emotion classification experiments.  



## ⚙️ **Installation**

1. **Clone the repository**:
   ```bash
   git clone https://github.com/username/depression-detector.git
   cd depression-detector
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate       # For Linux/macOS
   venv\Scripts\activate          # For Windows
   ```

3. **Download datasets**:
   Dataset is not included due to restrictions and huge size of dataset. You can request access to DAIC-WOZ Dataset from their official website.



 ## 🚀 **Usage**

 **Training**

- Train the **CNN model**:
   ```bash
   Convolutional Neural Network.ipynb
  You can also use pretrained model directly- my_cnn_model.h5
   ```

- Train the **SVM model**:
   ```bash
   CNN-SVM.ipynb
  or You can also use pretrained model directly- my_svm.pkl
   ```

- Fine-tune **DistilBERT**:
   ```bash
   BERT.ipynb
  You can also use pretrained model directly- mybert.h5
   ```

## **Web Application**
To launch the web app for live predictions:
```bash
cd app
python app.py
```
Visit `http://127.0.0.1:5000` in your browser.



## 🛠️ **Technologies Used**

- **Programming Language**: Python
- **Deep Learning Frameworks**: TensorFlow, PyTorch
- **Machine Learning Libraries**: Scikit-learn, Hugging Face Transformers
- **Frontend**: HTML, CSS (Flask Templates)
- **Visualization**: Matplotlib, Seaborn
- **Audio Processing**: LibROSA
- **Natural Language Processing**: Hugging Face, NLTK



## 📈 **Results**
![Screenshot 2024-11-13 220933](https://github.com/user-attachments/assets/97a3d253-c303-4700-92b4-6f5cef1c0362)
![Screenshot 2024-11-13 220949](https://github.com/user-attachments/assets/4bb1abb2-5474-4217-a5e0-43b90a1b0d7c)
![Screenshot 2024-11-13 221033](https://github.com/user-attachments/assets/299d92c1-2aed-4e02-bf94-8e7e7384ba09)
![Screenshot 2024-11-13 220855](https://github.com/user-attachments/assets/a7c6a051-9202-44d6-ad1f-465abc70c3c6)
![Screenshot 2024-11-13 220737](https://github.com/user-attachments/assets/dc6ece23-2579-45bf-9647-d8938451480b)
![Screenshot 2024-11-13 220751](https://github.com/user-attachments/assets/831dfd68-b2d0-458b-b805-09b58e63e630)
![Screenshot 2024-11-13 220821](https://github.com/user-attachments/assets/21f4366f-a858-45f1-86e7-4619370a4d7a)
![Screenshot 2024-11-16 185426](https://github.com/user-attachments/assets/42f5203e-aede-4f61-a56a-5841f76a1f3d)



| Model        | Accuracy |
|--------------|----------|
| CNN          | 91.0%    |
| SVM          | 88.0%    |
| DistilBERT   | 93.5%    |



## 👩‍🔬 **Future Improvements**

- Integrating multimodal data (audio + text + facial emotions).
- Implementing attention mechanisms in CNNs.
- Expanding dataset with diverse speech and text samples.
- Deploying on cloud platforms for scalability.


## 🤝 **Contributing**

We welcome contributions!  
- Fork the repository.
- Create a feature branch.
- Submit a pull request with your changes.



## ✨ **Acknowledgements**

- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [LibROSA](https://librosa.org/)


import os
from flask import Flask, render_template, request, redirect, url_for
from forms import UploadForm
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the saved models
pneumonia_model = load_model('pneumonia_detection_model (1).h5')
lung_cancer_model = load_model('lung_cancer_model_vgg16.h5')
colon_cancer_model = load_model('colon_cancer_model_vgg16.h5')
brain_tumor_model = load_model('vgg16_brain_tumor_classifier.h5')

# Class names for lung cancer
lung_class_names = ['Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

# Class names for colon cancer
colon_class_names = ['Colon Adenocarcinoma', 'Colon Benign Tissue']

# Function to preprocess the image for pneumonia detection
def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  # Load the image
    img = img_to_array(img)  # Convert the image to numpy array
    img = np.expand_dims(img, axis=0)  # Add a batch dimension
    img /= 255.0  # Rescale the image
    return img

# Function to prepare image for brain tumor detection
def prepare_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Make it a batch of one
    img_array /= 255.0  # Rescale pixel values to [0, 1]
    return img_array

# Function to load and preprocess images for lung and colon cancer
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to predict the class of an image
def predict_image(model, img_path, class_names):
    img = load_and_preprocess_image(img_path)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

# Function to provide suggestions based on pneumonia prediction
def provide_suggestions(prediction_percentage):
    if prediction_percentage > 50:
        return (
            f"The image is predicted as Pneumonia with a confidence of {prediction_percentage:.2f}%.",
            [
                "Consult a healthcare professional immediately for proper diagnosis and treatment.",
                "Rest adequately and stay hydrated.",
                "Monitor symptoms and temperature regularly.",
                "Follow prescribed treatment plans including taking any prescribed medications, such as antibiotics.",
                "Practice good hygiene, including regular hand washing.",
                "Keep away from others as much as possible to prevent spreading the infection."
            ]
        )
    else:
        return ("The image is predicted as Normal.", [])

# Function to provide suggestions based on brain tumor prediction
def provide_brain_tumor_suggestions(predicted_class):
    suggestions = {
        'Glioma': [
            "Consult a healthcare professional for further evaluation and treatment.",
            "Follow prescribed treatment plans, which may include surgery, radiation, or chemotherapy."
        ],
        'Meningioma': [
            "Consult a healthcare professional for further evaluation and treatment.",
            "Surgery is often recommended, followed by monitoring or additional treatments as necessary."
        ],
        'No Tumor': [
            "The image is predicted as having no tumor. Continue with regular check-ups and a healthy lifestyle."
        ],
        'Pituitary Tumor': [
            "Consult a healthcare professional for further evaluation and treatment.",
            "Treatment options may include medication, surgery, or radiation therapy."
        ]
    }
    return suggestions.get(predicted_class, ["Consult a healthcare professional for further evaluation."])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/organ/<organ_name>/<disease>', methods=['GET', 'POST'])
def organ(organ_name, disease):
    form = UploadForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        if organ_name == 'lungs' and disease == 'pneumonia':
            # Preprocess the image for pneumonia detection
            preprocessed_image = preprocess_image(filename, target_size=(150, 150))
            
            # Make a prediction using the pneumonia model
            prediction = pneumonia_model.predict(preprocessed_image)
            
            # Convert the prediction to percentage
            prediction_percentage = prediction[0][0] * 100
            
            # Provide suggestions based on the prediction
            prediction_text, suggestions = provide_suggestions(prediction_percentage)
        
        elif organ_name == 'lungs' and disease == 'cancer':
            # Predict the class of the image using the lung cancer model
            predicted_class_name = predict_image(lung_cancer_model, filename, lung_class_names)
            
            # Provide prediction text
            prediction_text = f"The image is predicted as {predicted_class_name}."
            suggestions = ["Consult a healthcare professional for further evaluation and treatment."]
        
        elif organ_name == 'colon' and disease == 'cancer':
            # Predict the class of the image using the colon cancer model
            predicted_class_name = predict_image(colon_cancer_model, filename, colon_class_names)
            
            # Provide prediction text
            prediction_text = f"The image is predicted as {predicted_class_name}."
            suggestions = ["Consult a healthcare professional for further evaluation and treatment."]
        
        elif organ_name == 'brain' and disease == 'tumor':
            # Prepare the image for brain tumor detection
            prepared_image = prepare_image(filename)
            
            # Predict the class of the image using the brain tumor model
            predictions = brain_tumor_model.predict(prepared_image)
            predicted_class_index = np.argmax(predictions)
            class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
            predicted_class = class_labels[predicted_class_index]
            
            # Provide suggestions based on the prediction
            suggestions = provide_brain_tumor_suggestions(predicted_class)
            prediction_text = f"The image is predicted as {predicted_class}."
        
        else:
            prediction_text = "No model available for this organ and disease."
            suggestions = []
        
        return render_template('result.html', organ=organ_name, disease=disease, prediction=prediction_text, suggestions=suggestions)
    return render_template('organ.html', organ=organ_name, disease=disease, form=form)

if __name__ == '__main__':
    app.run(debug=True)

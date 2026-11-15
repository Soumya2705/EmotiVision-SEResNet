# Facial Emotion Recognition (FER)

A Deep Learning-based Facial Emotion Recognition project using a custom SEResNet architecture trained from scratch.

## Features

- Recognition of 7 facial emotion classes.
- Live webcam emotion detection.
- Photo capture and prediction.
- Video recording and emotion detection.
- Image upload and prediction.
- OpenCV-based face detection.
- Streamlit-based user interface.

## Emotion Classes

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Model Architecture

- Architecture: Custom SEResNet
- Framework: PyTorch
- Parameters: 1,994,263
- Input size: 48 x 48 pixels
- Input channels: 1 (Grayscale)
- Output classes: 7
- Selected experiment: EXP05 (Batch Size 64)
- Best epoch: 72

## Validation Performance

| Metric | Score |
|---|---:|
| Accuracy | 66.35% |
| Balanced Accuracy | 65.94% |
| Macro Precision | 66.65% |
| Macro Recall | 65.94% |
| Macro F1-score | 66.20% |
| Weighted F1-score | 66.18% |

These metrics represent validation performance.

## Project Structure

- app.py
- FER_Final_Model.pth
- Final_Model_Info.json
- Final_Project_Summary.json
- requirements.txt
- notebooks/
- results/
- figures/
- screenshots/

## Installation

### 1. Clone the repository

    git clone https://github.com/YOUR_USERNAME/FER-Emotion-Recognition.git

### 2. Open the project folder

    cd FER-Emotion-Recognition

### 3. Create a virtual environment

    python -m venv venv

### 4. Activate the environment (Windows)

    venv\Scripts\activate

### 5. Install dependencies

    pip install -r requirements.txt

### 6. Run the application

    streamlit run app.py

## Results and Visualizations

Training curves, validation confusion matrices, and experiment comparison figures are available in the figures directory.

Detailed validation reports are available in the results directory.

## Dataset

The dataset is not included in this repository.

Dataset source and licensing information should be verified before redistribution.

## Disclaimer

This project is developed for educational and research purposes. Facial emotion predictions are model estimates and should not be treated as definitive measurements of a person's emotional state or used for clinical diagnosis.

## Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Streamlit
- Streamlit-WebRTC
- Matplotlib
- Jupyter Notebook

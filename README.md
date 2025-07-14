# Document Image Classifier using CNN

This project focuses on classifying document images into predefined categories using a Convolutional Neural Network (CNN). The model is trained on the [Tobacco-3482 dataset](https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg), which contains scanned document images across 10 types.

## Dataset Structure

The dataset should be arranged in the following folder format for compatibility with Keras' `ImageDataGenerator`:

```bash
Tobacco_data/
├── ADVE/
├── Email/
├── Form/
├── Letter/
├── Memo/
├── News/
├── Note/
├── Report/
├── Resume/
└── Scientific/
```

Each subfolder contains images corresponding to a document class.

## Features

- CNN model built using TensorFlow/Keras
- Supports training and validation split using `ImageDataGenerator`
- Outputs evaluation metrics including:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix
- Easy to run on [Google Colab](https://colab.research.google.com)

## Installation

To install dependencies:

```bash
pip install -r requirements.txt
```
Or manually install core packages:
```bash
pip install tensorflow scikit-learn matplotlib
```
## Training

To train the model, simply run the notebook:
```bash
doc_classifier.ipynb
```
You can also upload the Tobacco_data/ folder via Google Drive or manually zip → upload → unzip in Colab.
## Evaluation

After training, the model is evaluated using the validation set, and classification metrics are printed:
```bash
from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=class_labels))
```
## Saving the Model

To save the trained model:
```bash
model.save("document_classifier_model.h5")
```
##Author
Shreya Kumari




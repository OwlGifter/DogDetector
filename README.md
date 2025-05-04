# 🐶 Dog Detector using CNN

This project is a complete image classification pipeline built to detect whether an image contains a **dog or not**, using a custom-trained **Convolutional Neural Network (CNN)**. The model is trained on images collected from the internet and organized into training and validation sets.

## 📁 Project Structure

- **`cnn.py`**  
  Builds, trains, loads, and evaluates a CNN model using TensorFlow/Keras. Supports model reuse to avoid retraining if a saved model already exists.

- **`datasetDownloader.py`**  
  Automatically collects dog and non-dog images using DuckDuckGo image search. Images are resized and filtered for consistency and quality.

- **`splitdata.py`**  
  Splits the downloaded dataset into training and validation sets (e.g., 80/20) in a format compatible with Keras' `flow_from_directory()`.

## 🧠 Features

- Custom CNN model with dropout regularization
- Dataset auto-downloaded, resized, and filtered
- Automatic model saving/loading (`.keras` format)
- Supports image prediction using `model.predict()`
- Easy-to-extend for more breeds or categories

## 🚀 Getting Started

1. Run `datasetDownloader.py` to download and preprocess image data.
2. Run `splitdata.py` to organize the dataset into train/val folders.
3. Run `cnn.py` to train or load the model and evaluate or predict.

## 🐾 Requirements

- Python 3.8+
- TensorFlow 2.x
- matplotlib
- duckduckgo-search

### Install Dependencies

```bash
pip install tensorflow matplotlib duckduckgo-search

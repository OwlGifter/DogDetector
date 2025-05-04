# 🐶 Dog Detector using CNN

This project is a complete image classification pipeline built to detect whether an image contains a **dog or not**, using a custom-trained **Convolutional Neural Network (CNN)**. The model is trained on images collected from the internet and organized into training and validation sets.

## 🧠 Overview

- Custom CNN model with dropout regularization
- Dataset large or small, auto-downloaded, resized, and filtered
- Automatic model saving/loading (`.keras` format)
- Supports image prediction using `model.predict()`
- Easy-to-extend for more breeds or categories

---

## 📁 Project Structure

### `cnn.py`
- Loads training and validation datasets
- Builds and trains a CNN model if not already saved
- Saves the model as `.keras`
- Evaluates model accuracy and supports single image prediction

### `smallDatasetDownloader.py`
- Downloads images using [DuckDuckGo image search](https://github.com/deedy5/duckduckgo_search)
- Saves resized 256×256 images for `dog` and `no_dog` categories
- Intended for small-scale dataset creation (up to a few hundred images)

### `largeDatasetDownloader.py`
- Downloads 18,000+ **no-dog** images using WordNet synsets from [ImageNet](https://image-net.org)
- Extracts and resizes images to 256×256
- Organizes them into a flat structure for later splitting

### `splitdata.py`
- Splits images from `bulk data` into `train` and `val` folders using an 80/20 ratio
- Supports both `dog` and `no_dog` image categories
- Automatically moves and organizes files for model training

## 🗂️ FileStructure
```
DogDetector/
├── dataset/
│ ├── bulk data/ # Raw images from downloads
| | ├── dog/ # Images containing dogs
| | └── no_dog/ # Images not containing dogs
│ ├── train/ # Training images split by category
| | ├── dog/ # Images containing dogs
| | └── no_dog/ # Images not containing dogs
│ └── val/ # Validation images split by category
| | ├── dog/ # Images containing dogs
| | └── no_dog/ # Images not containing dogs
├── dog_detector_model.keras
├── cnn.py
├── smallDatasetDownloader.py
├── largeDatasetDownloader.py
├── splitdata.py
```
---

# 🚀 Getting Started

## Install Dependencies

```bash
pip install tensorflow matplotlib duckduckgo-search pillow numpy
```
---

1. Run `smallDatasetDownloader.py` or `largeDatasetDownloader` to download and preprocess image data(large or small).
2. Run `splitdata.py` to organize the dataset into train/val folders.
3. Run `cnn.py` to train or load the model and evaluate or predict.

## Notes
- All images are resized to 256x256
- Model uses flow_from_directory, which supports .jpg, .jpeg, .png
- Model saved in Keras' native .keras format
- Compatible with any binary image classification task with minimal changes

## 🐾 Requirements
- Python 3.8+
- TensorFlow 2.x
- matplotlib
- duckduckgo-search
- Pillow
- Numpy

# Contact
Created by Zabdiel Hernandez for educational and research use


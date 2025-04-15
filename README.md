# San Diego Bird Classifier
## 530-Class Deep Learning Model for Bird Species Identification

### Overview
This project aims to classify 530 bird species native to the San Diego County area using a deep learning model based on TensorFlow and EfficientNetB0. The workflow includes:
- Web scraping bird images via SerpAPI (Google/Bing)
- Data cleaning, validation, and augmentation
- Transfer learning with class-weighted training to handle imbalance
- Model evaluation (Top-1: 77.16%, Top-5: 91.66%)

### Key Features
1. Data Pipeline
Image Scraping: Automated collection of 300 images per species using SerpAPI.

Data Validation:
- Filter non-bird images using pretrained EfficientNetB0.
- Convert formats to JPEG and remove corrupt files.
- Train/Validation Split: 80/20 split with directory structuring.

2. Model Architecture
Base Model: EfficientNetB0 pretrained on ImageNet.

Custom Layers:
- Data augmentation (flips, rotation, contrast, brightness).
- Global average pooling, dense layers (1024, 512 neurons), dropout.
- Class Weighting: Compensates for imbalanced species distribution.

3. Training
Mixed Precision: mixed_float16 for faster training.

Callbacks:
- Early stopping, model checkpointing, TensorBoard logging.
- Custom callback to save models every 5 epochs.

Optimizer: Adam with learning rate 1e-4.

4. Evaluation
- Top-1 Accuracy: 77.16%

- Top-5 Accuracy: 91.66%

Metrics tracked for validation performance and overfitting.

### Usage
  
### Dataset
Source: Images scraped from Google/Bing using species names.

Size: 232,389 training images | 58,331 validation images.

Preprocessing: Resized to 256x256, normalized, augmented.

### Results
Metric	Performance
Top-1 Acc	77.16%
Top-5 Acc	91.66%

### Future Improvements
Data Expansion: Include rare species and geographic variants.
Model Tuning: Experiment with EfficientNetV2 or Vision Transformers.
Deployment: Build a Flask API or mobile app for real-time classification.

### Acknowledgments
SerpAPI for image scraping.
TensorFlow Hub for pretrained EfficientNetB0.
San Diego birding communities for species documentation.

For questions or contributions, reach out to ashhal.s.usmani@gmail.com!

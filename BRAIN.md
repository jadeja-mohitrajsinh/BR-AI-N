# Brain Radiology Assisted Intelligence Neuronet.

A comprehensive machine learning project for detecting brain tumors in MRI images using two different approaches: Custom CNN and ResNet50 transfer learning.

## 🧠 Project Overview

This project implements and compares two deep learning models for brain tumor detection:

1. **Custom CNN**: A lightweight convolutional neural network built from scratch
2. **ResNet50 Transfer Learning**: Pre-trained ResNet50 model fine-tuned for medical imaging

### 🎯 Key Features

- ✅ Complete data preprocessing pipeline
- ✅ Two-phase training approach (head training + fine-tuning)
- ✅ Advanced data augmentation techniques
- ✅ Comprehensive model evaluation with multiple metrics
- ✅ Grad-CAM visualizations for model interpretability
- ✅ Clinical implications analysis
- ✅ Professional visualization suite

## 📊 Dataset

- **Source**: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- **Classes**: 2 (Tumor, No Tumor)
- **Image Size**: 224x224 pixels
- **Format**: RGB images
- **Split**: 70% Training, 15% Validation, 15% Test

## 🚀 Quick Start

### Prerequisites

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn pandas numpy kaggle
```

### Setup Instructions

1. **Clone or download the project**
2. **Place your Kaggle API credentials** (`kaggle.json`) in the project directory
3. **Run the notebook** `brain_tumor_detection.ipynb`

### Kaggle API Setup

1. Go to Kaggle Account Settings
2. Create New API Token
3. Download `kaggle.json`
4. Place it in the project directory

## 🏗️ Model Architectures

### Custom CNN Architecture

```
Input (224, 224, 3)
├── Conv2D(16) + BatchNorm + MaxPool + Dropout(0.25)
├── Conv2D(32) + BatchNorm + MaxPool + Dropout(0.25)  
├── Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25)
├── GlobalAvgPool2D
├── Dense(128) + Dropout(0.5)
├── Dense(64) + Dropout(0.3)
└── Dense(1, sigmoid)
```

**Parameters**: ~500K | **Training Time**: ~15 minutes

### ResNet50 Transfer Learning

```
ResNet50 (ImageNet pretrained)
├── Frozen base layers (Phase 1)
├── GlobalAvgPool2D
├── Dense(512) + BatchNorm + Dropout(0.4) [Branch 1]
├── Dense(512) + BatchNorm + Dropout(0.4) [Branch 2]
├── Add (Residual Connection)
├── Dense(256) + BatchNorm + Dropout(0.4)
├── Dense(128) + BatchNorm + Dropout(0.3)
└── Dense(1, sigmoid)
```

**Parameters**: ~25M | **Training Time**: ~45 minutes

## 📈 Training Strategy

### Two-Phase Training Approach

**Phase 1: Head Training**
- All base layers frozen
- Train classification head only
- 20 epochs without augmentation
- Learning rate: 3e-4

**Phase 2: Fine-tuning**
- Unfreeze top 30 layers
- Light data augmentation
- 15 epochs with augmentation
- Learning rate: 3e-5

### Data Augmentation

```python
# Enhanced augmentation pipeline
rotation_range=15,
width_shift_range=0.05,
height_shift_range=0.05,
horizontal_flip=True,
zoom_range=[0.95, 1.05],
brightness_range=[0.95, 1.05]
```

## 🎯 Results

### Performance Metrics

| Metric | Custom CNN | ResNet50 |
|--------|------------|----------|
| **Accuracy** | 75.2% | 78.6% |
| **Precision** | 0.742 | 0.801 |
| **Recall** | 0.768 | 0.774 |
| **F1-Score** | 0.755 | 0.787 |
| **ROC-AUC** | 0.812 | 0.845 |

### Model Comparison

**Custom CNN Strengths:**
- ✅ Lightweight (500K parameters)
- ✅ Fast training and prediction
- ✅ Good for resource-constrained environments
- ✅ Interpretable architecture

**ResNet50 Strengths:**
- ✅ Higher accuracy (78.6% vs 75.2%)
- ✅ Better feature extraction
- ✅ Pre-trained ImageNet knowledge
- ✅ More robust to variations

## 🔍 Model Interpretability

### Grad-CAM Visualizations

The project includes comprehensive Grad-CAM (Gradient-weighted Class Activation Mapping) analysis:

- **Heat maps** showing model focus areas
- **Overlay visualizations** combining original images with attention maps
- **Comparative analysis** between CNN and ResNet50 focus patterns

```python
# Generate Grad-CAM visualization
def generate_gradcam_heatmap(model, img_array, last_conv_layer_name):
    # ...existing code...
    return heatmap
```

## 📊 Comprehensive Analysis

### Clinical Implications

- **Sensitivity**: 77.4% (True Positive Rate)
- **Specificity**: 79.8% (True Negative Rate)
- **Clinical Use**: Preliminary screening tool
- **Recommendation**: Assists radiologists, doesn't replace human expertise

### Threshold Optimization

The project includes automatic threshold optimization:
- **Default threshold**: 0.5
- **Optimized threshold**: 0.47
- **Improvement**: +2.3% balanced accuracy

## 📁 Project Structure

```
brain_tumor_detection/
├── brain_tumor_detection.ipynb    # Main notebook
├── README.md                      # This file
├── kaggle.json                    # Kaggle API credentials
├── brain_tumor_dataset/           # Dataset directory
│   ├── yes/                       # Tumor images
│   └── no/                        # No tumor images
├── saved_models/                  # Trained models
│   ├── custom_cnn_fixed.keras
│   └── resnet50_transfer.keras
└── results/                       # Generated visualizations
    ├── sample_images.png
    ├── training_history.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── gradcam_heatmaps.png
```

## 🔧 Usage Examples

### Loading Pre-trained Model

```python
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model('saved_models/resnet50_transfer.keras')

# Make prediction on new image
prediction = model.predict(processed_image)
probability = prediction[0][0]

if probability > 0.5:
    print(f"Tumor detected (confidence: {probability:.3f})")
else:
    print(f"No tumor detected (confidence: {1-probability:.3f})")
```

### Batch Processing

```python
# Process multiple images
def predict_batch(model, image_paths):
    predictions = []
    for path in image_paths:
        img = load_and_preprocess_image(path)
        pred = model.predict(img)[0][0]
        predictions.append({
            'image': path,
            'probability': pred,
            'diagnosis': 'Tumor' if pred > 0.5 else 'No Tumor'
        })
    return predictions
```

## 🏥 Clinical Integration

### Deployment Considerations

1. **Accuracy Requirements**: Target >95% for clinical use
2. **PACS Integration**: Hospital picture archiving systems
3. **Real-time Processing**: <5 seconds per image
4. **Regulatory Compliance**: FDA approval required
5. **Continuous Monitoring**: Performance tracking in production

### Limitations

- ⚠️ Model should not replace radiologist diagnosis
- ⚠️ Requires diverse training data for generalization
- ⚠️ Regular retraining needed with new data
- ⚠️ Careful monitoring of false negatives essential

## 🔬 Technical Details

### Preprocessing Pipeline

```python
def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    image = cv2.resize(image, (224, 224))
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    
    # Apply model-specific preprocessing
    if model_type == "ResNet50":
        image = resnet_preprocess_input(image * 255.0)
    
    return np.expand_dims(image, axis=0)
```

### Class Weight Calculation

```python
# Handle class imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
```

## 📚 References

1. [Brain MRI Images Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
2. [ResNet Paper](https://arxiv.org/abs/1512.03385)
3. [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
4. [Transfer Learning in Medical Imaging](https://www.nature.com/articles/s41746-019-0099-x)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for research and educational purposes only. It is not intended for clinical diagnosis or treatment decisions. Always consult qualified medical professionals for medical advice.

## 🆘 Troubleshooting

### Common Issues

**1. Kaggle Dataset Download Fails**
```bash
# Solution: Check kaggle.json credentials
export KAGGLE_CONFIG_DIR=./
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
```

**2. Memory Issues**
```python
# Solution: Reduce batch size
BATCH_SIZE = 4  # Instead of 8
```

**3. Model Loading Errors**
```python
# Solution: Load without compilation
model = tf.keras.models.load_model('model.keras', compile=False)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the notebook comments for detailed explanations

---


**Made with ❤️ for advancing medical AI research**

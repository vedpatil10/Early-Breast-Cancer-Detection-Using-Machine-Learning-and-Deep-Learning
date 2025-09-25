# Early Breast Cancer Detection Using Machine Learning and Deep Learning

A comprehensive machine learning and deep learning solution for early breast cancer detection using mammographic images. This project implements multiple algorithms to classify breast masses as benign or malignant, achieving high accuracy rates for improved diagnostic support.

## Features

- Multi-Algorithm Approach: Implements 8 different ML/DL algorithms for comprehensive comparison
- High Accuracy: XGBoost achieves 98.5% training accuracy and 93.4% testing accuracy
- Advanced Feature Extraction: Extracts 30 statistical and texture-based features from mammographic images
- Image Processing: Comprehensive preprocessing pipeline with grayscale conversion, Otsu's thresholding, and contour detection
- Data Balancing: SMOTE implementation to handle class imbalance
- Performance Visualization: Confusion matrices, ROC curves, and accuracy comparison graphs
- Cell Detection: Optional cancer cell visualization and region highlighting
- Model Persistence: Save and load trained models for deployment

## Algorithms Implemented

### Machine Learning Models
- **Support Vector Machine (SVM)** - 89% accuracy
- **Decision Tree** - 86.7% accuracy  
- **Random Forest** - 93.1% accuracy
- **XGBoost** - 93.4% accuracy (Best performer)
- **K-Nearest Neighbors (KNN)** - 78.8% accuracy
- **Naive Bayes** - 82.8% accuracy
- **K-Means Clustering** - 76.9% accuracy

### Deep Learning Model
- **Convolutional Neural Network (CNN)** - 91.9% accuracy (using MobileNetV2)

### Dimensionality Reduction
- **Principal Component Analysis (PCA)** - Applied to SVM, KNN, K-Means, and Naive Bayes

## Dataset

- **Total Images**: 9,686 mammographic images
- **Benign Cases**: 8,061 images
- **Malignant Cases**: 1,625 images
- **Data Split**: 80% training, 20% testing
- **Source**: MIAS (Mammographic Image Analysis Society) dataset

## Tech Stack

### Programming Language
- Python 3.x

### Development Environment
- Google Colab (cloud-based with GPU support)

### Machine Learning Libraries
- **Scikit-Learn**: Classical ML algorithms and evaluation metrics
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning implementation
- **Imbalanced-learn**: SMOTE for handling class imbalance

### Image Processing & Visualization
- **OpenCV**: Image processing and computer vision
- **NumPy & Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Visualization and plotting
- **Scikit-image**: Advanced image processing features

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Required dependencies**
   ```bash
   pip install numpy pandas scikit-learn opencv-python matplotlib seaborn
   pip install tensorflow keras xgboost imbalanced-learn scikit-image
   pip install tqdm joblib
   ```

## Usage

### 1. Feature Extraction
```python
from feature_extraction import BreastTumorFeatureExtractor

extractor = BreastTumorFeatureExtractor()
features_df = extractor.extract_features_from_folder('path/to/images/', 'features.csv')
```

### 2. Training Models
```python
from cancer_classifier import CancerClassifier

classifier = CancerClassifier()
classifier.load_and_preprocess('features.csv')
classifier.train_model()
classifier.evaluate_model()
```

### 3. Making Predictions
```python
# Import feature extractor
classifier.import_feature_extractor('feature_extraction.py')

# Predict on new image
result = classifier.predict_from_image('mammogram.jpg', display_image=True)
print(f"Prediction: {result['cancer_type']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 4. Model Management
```python
# Save trained model
classifier.save_model('trained_model.pkl')

# Load existing model
classifier.load_model('trained_model.pkl')
```

## Feature Extraction

The system extracts 30 comprehensive features from each mammographic image:

### Mean Features (10)
- Mean Radius, Texture, Perimeter, Area
- Mean Smoothness, Compactness, Concavity
- Mean Concave Points, Symmetry, Fractal Dimension

### Error Features (10)
- Standard errors for all mean features
- Indicates measurement variability and reliability

### Worst Features (10)
- Maximum values for all mean features
- Represents most severe characteristics

## Image Processing Pipeline

1. **Grayscale Conversion**: RGB to single-channel grayscale
2. **Otsu's Thresholding**: Automatic threshold selection for segmentation
3. **Contour Detection**: Identify object boundaries
4. **Intensity Rescaling**: Enhance contrast for better feature extraction
5. **Masking**: Isolate regions of interest
6. **GLCM Analysis**: Gray-Level Co-occurrence Matrix for texture features

## Performance Metrics

### Model Comparison
| Algorithm | Training Accuracy | Testing Accuracy | AUC Score |
|-----------|------------------|------------------|-----------|
| XGBoost   | 98.5%           | 93.4%           | 0.96      |
| Random Forest | 99.6%       | 93.1%           | 0.95      |
| CNN       | 92.4%           | 91.9%           | 0.94      |
| SVM       | 91.5%              | 89.0%           | 0.92      |
| Decision Tree | 91.2%          | 86.7%           | 0.89      |
| Naive Bayes | 83.2%             | 82.8%           | 0.87      |
| KNN       | 81.2%               | 78.8%           | 0.84      |
| K-Means   | 77.5%               | 76.9%           | 0.81      |

### Evaluation Metrics
- **Confusion Matrix**: Detailed breakdown of predictions
- **ROC Curve**: Receiver Operating Characteristic analysis
- **Precision, Recall, F1-Score**: Comprehensive performance evaluation
- **Sensitivity & Specificity**: Medical diagnostic metrics

## Project Structure

```
breast-cancer-detection/
├── data/
│   ├── mammographic_images/
│   └── extracted_features.csv
├── src/
│   ├── feature_extraction.py    # Feature extraction utilities
│   ├── cancer_classifier.py     # Main classification system
│   ├── preprocessing.py         # Image preprocessing functions
│   └── visualization.py         # Plotting and visualization
├── models/
│   ├── xgboost_model.pkl
│   ├── cnn_model.h5
│   └── feature_scaler.pkl
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── results_analysis.ipynb
├── requirements.txt
└── README.md
```

## Medical Relevance

### Clinical Applications
- **Early Detection**: Improved survival rates through timely diagnosis
- **Diagnostic Support**: Assists radiologists in mammogram interpretation
- **Screening Programs**: Automated analysis for large-scale screening
- **Resource Optimization**: Reduces workload in resource-limited settings

### Social Impact
- **Healthcare Accessibility**: Democratizes breast cancer screening
- **Cost Reduction**: Minimizes expensive false positive investigations
- **Public Health**: Supports population-wide screening initiatives

## Results & Findings

- **Best Performing Model**: XGBoost with 93.4% test accuracy
- **Deep Learning Performance**: CNN achieved 91.9% accuracy with automatic feature learning
- **Ensemble Methods**: Random Forest and XGBoost showed superior performance
- **Class Imbalance Handling**: SMOTE successfully improved minority class detection
- **Feature Importance**: Texture and morphological features most discriminative

## Future Enhancements

- **Advanced Architectures**: U-Net and Mask R-CNN for precise tumor segmentation
- **Multi-Modal Integration**: Combine mammography with ultrasound and MRI
- **Real-Time Processing**: Optimize for clinical deployment
- **Explainable AI**: Implement LIME/SHAP for decision interpretability
- **Mobile Application**: Develop smartphone-based screening tool

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jagdale2024breast,
  title={Early Breast Cancer Detection Using Machine Learning and Deep Learning Algorithms},
  author={Jagdale, Manthan Vasant and Patil, Vedant Dilip and Tekale, Prachi Prakash},
  journal={K.J. Somaiya Institute of Technology},
  year={2024}
}
```


## Disclaimer

This tool is intended for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare providers for medical decisions.

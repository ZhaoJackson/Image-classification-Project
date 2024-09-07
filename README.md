# **AI Image Recognition and Classification Project**

This project focuses on image recognition and classification using deep learning (DL) and machine learning (ML) frameworks. We address the challenge of noisy labels in a large dataset and use advanced feature engineering, model selection, and hyperparameter tuning to enhance prediction accuracy.

## **Overview**
The goal of this project is to classify images from the CIFAR-10 dataset using various machine learning and deep learning techniques. The dataset contains both noisy and clean labels, with feature engineering and dimensionality reduction applied to improve the classification accuracy.

- **Clean Labels**: 10,000 images with high-quality labels.
- **Noisy Labels**: 40,000 images with low-quality labels.
- **Objective**: Enhance label prediction accuracy through advanced features and model selection.

## **Dataset**
The dataset is derived from the CIFAR-10 dataset, which consists of 50,000 labeled images across 10 different categories. The project makes use of both clean and noisy labels for training and evaluation.

- **Clean Labels**: Stored in `data/clean_labels.csv`.
- **Noisy Labels**: Stored in `data/noisy_labels.csv`.
- **Predicted Labels**: Predicted by the best model, stored in `data/predicted_labels.csv`.

### **Test Data**
The test data includes images and corresponding labels, which are stored in `data/test_data/test_labels.csv`.

## **Project Workflow**

1. **Feature Engineering**:
   - **Engineered 67 features** from CIFAR-10 images using RGB color histograms, grayscale, color statistics, Sobel operator, Local Binary Patterns (LBP), and Fast Fourier Transform (FFT).
   - Utilized **VGG16** to generate 4096 feature vectors per image for deeper feature extraction.
   
2. **Dimensionality Reduction**:
   - Applied **Principal Component Analysis (PCA)** and XGBoost for feature selection, reducing the dimensionality from 4096 to **400 features** while retaining important information.
   - Compared **PCA vs. feature importance** extracted via XGBoost to enhance accuracy from **81.3% to 86.75%**.

3. **Model Selection**:
   - Performed model selection using several machine learning and deep learning models, including:
     - Random Forest (RF)
     - XGBoost
     - Logistic Regression (LR)
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
     - Convolutional Neural Networks (CNN)
     - Recurrent Neural Networks (RNN)
   - **SVM** was identified as the best model, achieving a validation accuracy of **89.35%**.

4. **Hyperparameter Tuning**:
   - Applied **Random Search Cross Validation (CV)** and **Bayesian Optimization** for hyperparameter tuning, improving the SVM model's validation accuracy by **11.5%** from baseline.

## **Features and Feature Engineering**

- Engineered features include:
  - **RGB Histograms**
  - **Grayscale Conversion**
  - **Color Statistics**
  - **Sobel Edge Detection**
  - **Local Binary Patterns (LBP)**
  - **Fast Fourier Transform (FFT)**
  - **VGG16**-based feature extraction, generating 4096-dimensional vectors per image.

- After feature engineering, **PCA** and XGBoost were used to reduce the feature set to 400 features, optimizing for both dimensionality and performance.

## **Models and Techniques**

The project explored various models, including:
- **Support Vector Machine (SVM)**: Selected as the best-performing model, achieving an accuracy of **89.35%**.
- **XGBoost**: Used both for feature selection and classification, yielding competitive results.
- **Convolutional Neural Networks (CNN)**: Used for raw pixel data classification, but outperformed by SVM on this dataset.
- **Random Forest, KNN, Logistic Regression**: Explored but did not outperform SVM.

## **How to Run the Project**

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd image-classification-project
   ```

2. **Install Dependencies**:
   Install all necessary packages listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   Execute the project by running the `main.ipynb` notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

4. **Model Prediction**:
   To use the pre-trained model for predictions:
   ```python
   import pickle
   with open('best_model_xgboost.pkl', 'rb') as model_file:
       model = pickle.load(model_file)
   # Add your prediction code here using the test data
   ```

## **Dependencies**

This project relies on the following key libraries and tools (full list in `requirements.txt`):
- **XGBoost**: Model training and feature selection.
- **PCA**: Dimensionality reduction.
- **SVM**: Best-performing classification model.
- **TensorFlow/Keras**: For CNN model training and VGG16 feature extraction.
- **Pandas & NumPy**: Data manipulation and preprocessing.
- **Matplotlib**: Visualization of results.

## **File Structure**

```bash
image-classification-project/
|------main.ipynb                  # Main execution file
|------README.md                  # Project documentation
|------requirements.txt           # Dependencies
|------best_model3.pkl             # Baseline model
|------best_model_xgboost.pkl      # XGBoost model
|------data/
|--------test_data/
|------------test_labels.csv       # Test labels
|--------best_model_xgboost.pkl
|--------clean_labels.csv          # 10,000 clean labels
|--------noisy_labels.csv          # 40,000 noisy labels
|--------predicted_labels.csv      # Predicted labels from the best model
```

## **Results**

- **Baseline Accuracy**: 31.5% using simple RGB features.
- **Post Feature Engineering Accuracy**: 86% after introducing advanced features such as color statistics, LBP, and FFT.
- **Dimensionality Reduction**: PCA and XGBoost reduced features to 400, improving model accuracy to **86.75%**.
- **Best Model**: SVM with an accuracy of **89.35%** after hyperparameter tuning using Random Search CV and Bayesian Optimization.
# 🕵️‍♂️ Real and Fake Face Detection using EfficientNet and Ensemble Models

This project aims to **classify real and fake face images** using a combination of **EfficientNet, SVM, and K-Nearest Neighbors (KNN)** classifiers. Enhanced by **Error Level Analysis (ELA)** preprocessing, this approach leverages **deep learning** and **traditional machine learning** techniques for robust performance.

---

## 📂 Project Structure

- **`real_and_fake_face/`**: Contains subfolders:
  - `training_real/` - Real face images  
  - `training_fake/` - Fake face images  
- **Dataset**: [Kaggle - Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection/data)
- **`notebook.ipynb`**: Jupyter notebook for **data preprocessing, model training, evaluation, and exporting the model**.
- **`README.md`**: Guide on **setup, training, and inference**.

---

## 🛠️ Requirements

Before running the code, install the necessary dependencies:

```bash
pip install numpy pandas matplotlib opencv-python-headless pillow scikit-learn tensorflow torch torchvision tensorflow_model_optimization
```

---

## 🔍 Code Overview

1. **Data Loading**
    - Images are loaded and labeled from directories.

2. **Error Level Analysis (ELA)**
    - Each image undergoes ELA preprocessing to enhance distinguishing features between real and fake images.

3. **Model Preparation**
    - EfficientNet is trained and used in an ensemble.
    - SVM & KNN classifiers are trained on flattened images for ensemble predictions.

4. **Pruning & Quantization**
    - EfficientNet is pruned and quantized for optimized deployment.

5. **Ensemble Prediction**
    - Predictions from EfficientNet, SVM, and KNN are combined for final classification.

6. **Evaluation**
    - Accuracy, confusion matrix, and other metrics are computed.

---

## 🚀 How to Run

**Step 1: Data Preprocessing and ELA Transformation**

Run the cells in the notebook to load and preprocess the images using adaptive ELA.

**Step 2: Train the Models**

Execute the following command to train EfficientNet, SVM, and KNN models:

```python
history = efficientnet_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32
)
```

**Step 3: Ensemble Prediction and Evaluation**

To evaluate the ensemble model on test data, run:

```python
final_predictions = ensemble_predict(efficientnet_model, svm_clf, knn_clf, X_test, X_test_flat)
print(classification_report(y_test, final_predictions))
```

**Step 4: Model Optimization & Deployment**

EfficientNet is pruned and quantized before saving the final model:

```python
model_for_export.save('pruned_efficientnet_model.keras')
```

**Step 5: Inference Time Measurement**

To measure inference time, run:

```python
import time
start_time = time.time()
_ = efficientnet_model.predict(X_test)
end_time = time.time()
print(f'Total inference time: {end_time - start_time:.2f} seconds')
```

**Step 6: Visualization**

The notebook includes loss and accuracy plots to visualize training performance.

---

## 📊 Presentation Slides

For an overview of this project, view the slides:

📌 [Presentation Slides](Final_Slides.pdf)

---

## 🏆 Key Features

✔️ Deep Learning (EfficientNet) - High-performance CNN model

✔️ Machine Learning (SVM, KNN) - Traditional ML classifiers

✔️ Error Level Analysis (ELA) - Enhanced image preprocessing

✔️ Pruning & Quantization - Model optimization for deployment

✔️ Ensemble Model - Combining deep learning & ML for accuracy

---

## 📬 Contact  
For collaborations, inquiries, or project discussions, feel free to reach out!  

📌 **GitHub:** [https://github.com/kc-3]  
📌 **LinkedIn:** [https://www.linkedin.com/in/kranthi-chaithanya-thota/]  

---

**🚀 If you find this repository useful, give it a ⭐ and follow for more updates!**

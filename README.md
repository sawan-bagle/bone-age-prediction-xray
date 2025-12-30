# Bone Age Estimation from X-ray Images

This repository contains a deep learning–based machine learning project focused on estimating bone age from hand X-ray images. The task is formulated as a **regression problem**, where the model predicts skeletal age (in months) directly from medical images.

---

## Project Overview

Bone age assessment is an important indicator of skeletal maturity in pediatric healthcare. Traditional assessment methods are manual and subject to inter-observer variability.  
This project explores automated bone age estimation using convolutional neural networks and ensemble learning techniques.

---

## Dataset

- Publicly available bone age X-ray dataset
- Input: Hand X-ray images (`.png`)
- Target: Bone age (in months)
- Total samples: ~12,000 images

---

## Models Implemented

1. **Custom CNN**
   - Convolutional layers with max pooling
   - Tuned using Keras Tuner

2. **VGG16 (Transfer Learning)**
   - Pre-trained on ImageNet
   - Frozen convolutional base with custom regression head

3. **Weighted Ensemble**
   - Weighted average of CNN and VGG16 predictions

4. **Stacking Ensemble**
   - Linear Regression meta-model trained on validation predictions
   - Provided the best overall performance

---

## Evaluation Metrics

Models were evaluated using standard regression metrics:
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Results Summary

- Transfer learning significantly outperformed a custom CNN trained from scratch
- Stacking ensemble improved predictive performance over individual models
- Ensemble modeling proved effective for reducing prediction error

---

## My Contribution

This project was completed as part of a **team academic project**.  
My primary contributions included:

- Designing and implementing ensemble strategies (weighted and stacking)
- Building a stacking ensemble using a Linear Regression meta-model
- Model evaluation and comparative analysis
- Result interpretation and visualization

---

## Compute Note

Due to computational requirements, full model training was performed in a university lab environment.  
This repository focuses on **model architecture, ensemble methodology, and evaluation logic**.

---

## Technologies Used

- Python
- TensorFlow / Keras
- Keras Tuner
- Scikit-learn
- NumPy, Pandas, Matplotlib

---

## Disclaimer

This project is intended for academic and research purposes only and does **not** represent a medical diagnostic system.

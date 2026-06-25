# Sickle Cell Detection

A deep learning-based web application that detects Sickle Cell disease from blood smear images using a CNN model (MobileNet with transfer learning).

## 🚀 Features
- Upload blood smear images
- Predict Normal or Sickle Cell
- Shows probability score
- Decision explanation
- Simple medical dashboard using Streamlit

## 🧠 Model Details
- CNN Architecture: MobileNet (Transfer Learning)
- Framework: TensorFlow / Keras
- Input: Blood smear image
- Output: Binary classification

## 🛠️ Tech Stack
- Python
- Streamlit
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn

## 📁 Project Files
- app.py → main application
- mobilenet_weights.weights.h5 → trained model
- requirements.txt → dependencies

## 📊 Results
🔬 5-Fold Cross Validation Results

The model was evaluated using 5-Fold Cross Validation to ensure robustness and reduce bias.

Metric	Score
Accuracy	0.9032 (90.32%) ± 0.0224
Precision	0.9261
Recall	0.9454
F1-Score	0.9351
ROC-AUC	0.9534 ± 0.0178

📈 Test Set Classification Report
Class	Precision	Recall	F1-Score	Support
Normal (0)	0.79	1.00	0.88	22
Sickle (1)	1.00	0.91	0.95	64
Overall Performance:
Accuracy: 0.93 (93%)
Macro Avg F1-score: 0.92
Weighted Avg F1-score: 0.93

📉 Confusion Matrix
[[22  0]
 [ 6 58]]

📊 ROC-AUC Score
ROC-AUC: 0.9843

🧠 Interpretation of Results
The model performs very well in detecting Sickle Cell cases (Class 1) with high recall (0.91–1.00 range).
Normal class shows slightly lower precision due to class imbalance.
High ROC-AUC (0.98) indicates strong separability between classes.
5-Fold results show the model is stable and not heavily overfitting (low standard deviation).
🏆 Final Model Performance (Best Result)
Accuracy: 93%
ROC-AUC: 0.98
F1-Score: 0.93
This model demonstrates strong generalization ability and is suitable for assisting early-stage screening of Sickle Cell Disease using medical imaging.


---

## 🖼 Outputs

### Normal Cell Prediction
![Normal Cell](images/normal.png)

### Sickle Cell Prediction
![Sickle Cell](images/sickle.png)

---

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py


## 📦 Installation
```bash
pip install -r requirements.txt

---




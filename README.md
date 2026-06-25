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

## 📦 Installation
```bash
pip install -r requirements.txt

---

## 📊 Results
### Normal Case Output
![Normal Prediction](images/normal cell.png)
### Sickle Cell Prediction
![Sickle Cell Prediction](images/sickle cell.png)


---

## 🖼 Results

### Normal Cell Prediction
![Normal Cell](images/normal.png)

### Sickle Cell Prediction
![Sickle Cell](images/sickle.png)

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

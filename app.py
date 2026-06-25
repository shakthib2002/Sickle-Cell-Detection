import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():

    base_model = tf.keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_shape=(224,224,3)
    )

    base_model.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=output)
    model.load_weights("mobilenet_weights.weights.h5")

    return model

model = load_model()

# ---------------- TITLE ----------------
st.title("🩸 Sickle Cell Detection System")
st.write("AI-based detection of sickle cell disease from blood smear images")

st.markdown("---")

# ---------------- ABOUT ----------------
st.header("📘 About Sickle Cell Disease")
st.write("""
Sickle Cell Disease is a genetic blood disorder where red blood cells become rigid and sickle-shaped,
reducing oxygen flow in the body.
""")

# ---------------- CAUSES ----------------
st.subheader("🧬 Causes")
st.write("""
- Genetic mutation in hemoglobin gene  
- Inherited from parents  
- Autosomal recessive disorder  
""")

# ---------------- SYMPTOMS ----------------
st.subheader("⚠️ Symptoms")
st.write("""
- Fatigue  
- Pain episodes  
- Swelling in hands/feet  
- Frequent infections  
""")

# ---------------- CARE ----------------
st.subheader("🍎 Care & Prevention")
st.write("""
- Drink plenty of water  
- Healthy balanced diet  
- Regular checkups  
- Avoid extreme temperatures  
""")

st.markdown("---")

# ---------------- PATIENT DETAILS ----------------
st.header("👤 Patient Details")

name = st.text_input("Patient Name")
age = st.number_input("Age", 0, 120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.markdown("---")

# ---------------- IMAGE UPLOAD ----------------
st.header("🔬 Upload Blood Smear Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # ---------------- PREPROCESS ----------------
    img = image.resize((224,224))
    img = np.array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    # ---------------- PREDICTION ----------------
    pred = float(model.predict(img, verbose=0)[0][0])

    st.markdown("---")
    st.header("📊 Analysis Result")

    # ---------------- RESULT ----------------
    if pred >= 0.5:
        st.error(f"🩸 Sickle Cell Positive ({pred*100:.2f}%)")
    else:
        st.success(f"✅ Normal Blood Smear ({(1-pred)*100:.2f}%)")

    # ---------------- PROBABILITY BAR ----------------
    st.subheader("📈 Probability Score")

    st.write("Sickle Cell Probability")
    st.progress(pred)

    st.write(f"{pred*100:.2f}% chance of Sickle Cell")

    # ---------------- DECISION EXPLANATION ----------------
    st.subheader("🧠 Decision Explanation")

    if pred > 0.8:
        st.warning("High confidence: Strong signs of Sickle Cell detected.")
    elif pred > 0.5:
        st.info("Moderate risk: Some abnormal patterns detected.")
    else:
        st.success("Low risk: Blood smear appears normal.")

    # ---------------- PATIENT SUMMARY ----------------
    st.subheader("📄 Patient Summary")

    st.write(f"Name: {name}")
    st.write(f"Age: {age}")
    st.write(f"Gender: {gender}")

# ---------------- DISCLAIMER ----------------
st.markdown("---")
st.warning("""
⚠️ DISCLAIMER:  
This tool is for educational and research purposes only.  
It is not a substitute for professional medical diagnosis or treatment.
""")
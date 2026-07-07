import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

IMG_SIZE = 128

@st.cache_resource
def load_trained_model():
    return load_model("model_transfer.keras")

model = load_trained_model()

st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf image, and the model will predict the disease (or confirm it's healthy).")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict"):
        predictions = model.predict(img_array)
        predicted_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = predictions[0][predicted_idx] * 100

        readable_name = predicted_class.replace("___", " - ").replace("__", " ").replace("_", " ")

        if "healthy" in predicted_class.lower():
            st.success(f"✅ {readable_name} ({confidence:.1f}% confidence)")
        else:
            st.error(f"⚠️ {readable_name} ({confidence:.1f}% confidence)")

        st.subheader("Confidence across all classes")
        top3_idx = np.argsort(predictions[0])[-3:][::-1]
        for idx in top3_idx:
            name = CLASS_NAMES[idx].replace("___", " - ").replace("__", " ").replace("_", " ")
            st.write(f"{name}: {predictions[0][idx]*100:.1f}%")
            st.progress(float(predictions[0][idx]))
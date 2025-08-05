import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import io

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Brain Tumor Classifier", layout="centered")

# Load your trained model
@st.cache_resource
def load_brain_tumor_model():
    model = load_model("model.h5")  # update path if different
    return model

model = load_brain_tumor_model()

# Enhanced preprocessing function to handle shape mismatches
def preprocess_image(image: Image.Image):
    """
    Preprocess image to match model's expected input shape (128, 128, 3)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((128, 128))
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]
    elif img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Define Streamlit UI
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI scan image to detect if a tumor is present.")
st.info("📋 Supported formats: JPG, PNG, JPEG. The image will be automatically resized to 128x128 pixels for analysis.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.write(f"**Original image size:** {image.size}")
        st.write(f"**Image mode:** {image.mode}")
        st.image(image, caption="Uploaded MRI Image", use_column_width=True)
        
        if st.button("🔍 Analyze MRI Scan"):
            with st.spinner("⏳ Analyzing the MRI scan..."):
                try:
                    processed_image = preprocess_image(image)
                    st.write(f"**Processed image shape:** {processed_image.shape}")
                    
                    prediction = model.predict(processed_image)
                    predicted_class = np.argmax(prediction, axis=1)[0]
                    confidence = np.max(prediction) * 100
                    
                    class_names = ['No Tumor', 'Meningioma', 'Glioma', 'Pituitary']
                    result = class_names[predicted_class]
                    
                    st.success(f"🧠 **Prediction: {result}**")
                    st.write(f"**Confidence: {confidence:.2f}%**")
                    
                    st.subheader("📊 Prediction Probabilities")
                    prob_dict = {class_names[i]: prediction[0][i] for i in range(len(class_names))}
                    st.bar_chart(prob_dict)
                    
                    st.warning("⚠️ **Disclaimer:** This is an AI prediction tool for educational purposes only. Always consult with qualified medical professionals for proper diagnosis and treatment.")
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.write("Please check that your model file is compatible and accessible.")
    
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")
        st.write("Please make sure you've uploaded a valid image file.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This application uses a deep learning model to classify brain MRI scans into four categories:
    - **No Tumor**: Healthy brain tissue
    - **Meningioma**: Tumor of the meninges
    - **Glioma**: Tumor of glial cells
    - **Pituitary**: Pituitary gland tumor
    """)

    st.header("🔧 Technical Details")
    st.write("- **Input Size:** 128 × 128 pixels")
    st.write("- **Model:** VGG16-based CNN")
    st.write("- **Format:** RGB images")

    st.header("📝 Usage Tips")
    st.write("""
    - Upload clear MRI scan images
    - Supported formats: JPG, PNG, JPEG
    - Images are automatically preprocessed
    - Check confidence scores for reliability
    """)

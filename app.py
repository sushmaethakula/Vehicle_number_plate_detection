import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Paths
cascade_file = "haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(cascade_file)

st.title("Vehicle number Plate Detection🚗")
st.write("Upload an image, and we'll detect the number plate.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
min_area = 500

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image).copy()
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Detect plates
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # If no plates detected
    if len(plates) == 0:
        st.warning("No number plate detected.")
    else:
        for (x, y, w, h) in plates:
            area = w * h
            if area > min_area:
                cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img_array, "Number Plate", (x, y - 5),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

        st.image(img_array, caption="Result", use_container_width=True)

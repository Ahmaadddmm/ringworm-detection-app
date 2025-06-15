import streamlit as st
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import numpy as np
import io

# Load model lokal
model = YOLO("my_model.pt")

st.title("Ringworm Detection (Lokal Model - YOLO)")

uploaded_file = st.file_uploader("Upload gambar kulit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Gambar Asli", use_column_width=True)

    with st.spinner("Memproses gambar..."):
        # Jalankan prediksi
        results = model(image)

        # Ambil result pertama (karena YOLO bisa batch)
        result = results[0]

        if result.masks is None:
            st.write("Tidak ada objek terdeteksi.")
        else:
            draw_image = image.copy().convert("RGBA")
            draw = ImageDraw.Draw(draw_image, 'RGBA')

            for i, (cls_id, conf, mask) in enumerate(zip(result.boxes.cls, result.boxes.conf, result.masks.data)):
                class_name = model.names[int(cls_id)]
                st.markdown(f"**Deteksi #{i+1} - {class_name}**")
                st.write(f"Confidence: {conf:.2f}")

                # Mask to polygon (approximate)
                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_np).resize(image.size)
                mask_data = np.array(mask_img)

                # Cari polygon dari mask
                from skimage import measure
                contours = measure.find_contours(mask_data, 0.5)

                for contour in contours:
                    polygon = [(x[1], x[0]) for x in contour]
                    draw.polygon(polygon, fill=(255, 0, 0, 80), outline=(255, 0, 0, 180))

            st.image(draw_image, caption="Hasil Segmentasi (Model Lokal)", use_column_width=True)

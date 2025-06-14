import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

ROBOFLOW_API_KEY = "JhoVl7G0GZ41MBBBr0eK"
PROJECT_NAME = "ringworm-detection"
MODEL_VERSION = "2"
ROBOFLOW_URL = f"https://detect.roboflow.com/{PROJECT_NAME}/{MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}"

st.title("Ringworm Detection with Instance Segmentation (Polygon Overlay)")

uploaded_file = st.file_uploader("Upload gambar kulit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    st.image(image, caption="Gambar Asli", use_column_width=True)

    with st.spinner("Memproses gambar..."):
        response = requests.post(
            ROBOFLOW_URL,
            files={"file": image_bytes},
            data={"confidence": 20, "overlap": 30}
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])
            st.subheader("Prediksi:")

            if not predictions:
                st.write("Tidak ada objek terdeteksi.")
            else:
                image_with_polygon = image.copy()
                draw = ImageDraw.Draw(image_with_polygon, 'RGBA')

                for i, pred in enumerate(predictions):
                    st.markdown(f"**Deteksi #{i+1} - {pred['class']}**")
                    st.write(f"Confidence: {pred['confidence']:.2f}")

                    if "points" in pred:
                        polygon = []
                        for point in pred['points']:
                            # Scaling jika koordinat masih dalam rasio 0-1
                            x = point['x'] * image.width if point['x'] <= 1 else point['x']
                            y = point['y'] * image.height if point['y'] <= 1 else point['y']
                            polygon.append((x, y))
                        draw.polygon(polygon, fill=(255, 0, 0, 80), outline=(255, 0, 0, 180))
                    else:
                        st.warning("Tidak ada data 'points' untuk prediksi ini.")

                st.image(image_with_polygon, caption="Hasil Segmentasi (Polygon)", use_column_width=True)
        else:
            st.error(f"Terjadi kesalahan: {response.text}")

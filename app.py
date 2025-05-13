import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

# === Konfigurasi Roboflow ===
ROBOFLOW_API_KEY = "JhoVl7G0GZ41MBBBr0eK"
PROJECT_NAME = "ringworm-detection"
MODEL_VERSION = "2"
ROBOFLOW_URL = f"https://detect.roboflow.com/{PROJECT_NAME}/{MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}"

st.title("Ringworm Detection with Bounding Boxes")

uploaded_file = st.file_uploader("Upload gambar kulit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca file hanya sekali sebagai bytes
    image_bytes = uploaded_file.read()

    # Load image dari bytes (bukan langsung uploaded_file)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="Gambar Asli", use_column_width=True)

    with st.spinner("Memproses gambar..."):
        response = requests.post(
            ROBOFLOW_URL,
            files={"file": image_bytes},
            data={"confidence": 40, "overlap": 30}
        )

        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", [])

            st.subheader("Prediksi:")
            if not predictions:
                st.write("Tidak ada objek terdeteksi.")
            else:
                for pred in predictions:
                    st.write(f"- **Label**: {pred['class']}, Confidence: {pred['confidence']:.2f}")
                    st.write(f"  Box: x={pred['x']}, y={pred['y']}, w={pred['width']}, h={pred['height']}")

                # Gambar bounding box
                image_with_boxes = image.copy()
                draw = ImageDraw.Draw(image_with_boxes)
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()

                for pred in predictions:
                    x, y = pred["x"], pred["y"]
                    w, h = pred["width"], pred["height"]
                    x0, y0 = x - w / 2, y - h / 2
                    x1, y1 = x + w / 2, y + h / 2

                    draw.rectangle([x0, y0, x1, y1], outline="red", width=2)
                    draw.text((x0, y0 - 10), pred["class"], fill="red", font=font)

                st.image(image_with_boxes, caption="Hasil Deteksi", use_column_width=True)
        else:
            st.error(f"Terjadi kesalahan: {response.text}")

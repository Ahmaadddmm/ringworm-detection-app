import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import base64

# === Konfigurasi Roboflow ===
ROBOFLOW_API_KEY = "JhoVl7G0GZ41MBBBr0eK"
PROJECT_NAME = "ringworm-detection"
MODEL_VERSION = "2"
ROBOFLOW_URL = f"https://detect.roboflow.com/{PROJECT_NAME}/{MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}"

st.title("Ringworm Detection with Instance Segmentation")

uploaded_file = st.file_uploader("Upload gambar kulit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
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

            if not predictions:
                st.subheader("Prediksi:")
                st.write("Tidak ada objek terdeteksi.")
            else:
                # Salin gambar untuk overlay mask
                image_with_mask = image.copy()

                for i, pred in enumerate(predictions):
                    st.markdown(f"**Deteksi #{i+1} - {pred['class']}**")
                    st.write(f"Confidence: {pred['confidence']:.2f}")
                    st.write(f"Box: x={pred['x']}, y={pred['y']}, w={pred['width']}, h={pred['height']}")

                    # Ambil mask base64
                    if "mask" in pred:
                        mask_data = pred["mask"].split(",")[1]  # Hilangkan 'data:image/png;base64,'
                        mask_image = Image.open(io.BytesIO(base64.b64decode(mask_data))).convert("L")

                        # Buat RGBA mask merah transparan
                        red_mask = Image.new("RGBA", mask_image.size, (255, 0, 0, 100))
                        image_with_mask.paste(red_mask, (0, 0), mask_image)

                st.image(image_with_mask, caption="Hasil Segmentasi", use_column_width=True)
        else:
            st.error(f"Terjadi kesalahan: {response.text}")

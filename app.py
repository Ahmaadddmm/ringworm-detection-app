import streamlit as st
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import numpy as np
import io
import google.generativeai as genai
from skimage import measure

# === KONFIGURASI GEMINI API ===
genai.configure(api_key=st.secrets["AIzaSyDw_4Ae1E5sPM4av6xs_1Av42g-h9Jrs_Q"])

def tanya_gemini(pertanyaan):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat()
    response = chat.send_message(pertanyaan)
    return response.text

# === LOAD MODEL LOKAL YOLO ===
model = YOLO("my_model.pt")

st.title("🧪 Ringworm Detection & Konsultasi AI")

uploaded_file = st.file_uploader("📷 Upload gambar kulit...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    st.image(image, caption="🖼️ Gambar Asli", use_column_width=True)

    with st.spinner("🔍 Mendeteksi ringworm..."):
        results = model(image)
        result = results[0]

        if result.masks is None:
            st.warning("❌ Tidak ada objek terdeteksi.")
        else:
            draw_image = image.copy().convert("RGBA")
            draw = ImageDraw.Draw(draw_image, 'RGBA')

            for i, (cls_id, conf, mask) in enumerate(zip(result.boxes.cls, result.boxes.conf, result.masks.data)):
                class_name = model.names[int(cls_id)]
                st.markdown(f"**🦠 Deteksi #{i+1} - {class_name}**")
                st.write(f"Confidence: {conf:.2f}")

                mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                mask_img = Image.fromarray(mask_np).resize(image.size)
                mask_data = np.array(mask_img)

                contours = measure.find_contours(mask_data, 0.5)
                for contour in contours:
                    polygon = [(x[1], x[0]) for x in contour]
                    draw.polygon(polygon, fill=(255, 0, 0, 80), outline=(255, 0, 0, 180))

            st.image(draw_image, caption="✅ Hasil Deteksi (Model Lokal)", use_column_width=True)

    # === FORM CHATBOT ===
    st.markdown("---")
    st.markdown("### 💬 Konsultasi dengan AI tentang Ringworm")

    with st.form("form_gemini"):
        pertanyaan = st.text_input("📝 Masukkan pertanyaan (contoh: cara merawat ringworm?)")
        submitted = st.form_submit_button("Tanya AI")

        if submitted and pertanyaan:
            with st.spinner("🤖 Menjawab..."):
                jawaban = tanya_gemini(pertanyaan)
                st.markdown("#### 💡 Jawaban dari AI:")
                st.write(jawaban)

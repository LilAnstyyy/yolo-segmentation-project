import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# -------------------------------
# 🧠 Загрузка обученной модели
# -------------------------------
MODEL_PATH = "runs/segment/my_yolo11_seg_new/weights/best.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(
    page_title="YOLOv8 Сегментация",
    page_icon="🤖",
    layout="centered"
)

st.title("Сегментация объектов с YOLOv8")
st.markdown(
    """
    Загрузите изображение — модель выделит найденные объекты.  
    *(Используется модель: `runs/segment/my_yolo11_seg_new/weights/best.pt`)*
    """
)

uploaded = st.file_uploader("📂 Загрузите изображение", type=["jpg", "jpeg", "png"])


if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Исходное изображение", use_column_width=True)

    with st.spinner("🧩 Обрабатываем изображение..."):
        results = model.predict(np.array(image))
        annotated_image = results[0].plot()  # рисует боксы и маски

    st.image(annotated_image, caption="Результат сегментации", use_column_width=True)
    st.success("Готово!")

st.markdown(
    """
    ---
    *Разработано с помощью Streamlit + YOLOv8 (Ultralytics)*  
    """
)

import json
import os
from datetime import datetime
from cv2 import VideoWriter_fourcc, VideoWriter, resize

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from re import compile
from ultralytics import YOLO


class ImgToVid:
    def __init__(self, img_dir, name, extension, fps, size):
        self.img_dir = img_dir
        self.name = name
        self.extension = extension
        self.fps = float(fps)
        self.size = tuple(size)

    def sort_files(self, value):
        num = compile(r'(\d+)')
        chunks = num.split(value)
        chunks[1::2] = map(int, chunks[1::2])
        return chunks

    def img_to_video(self):
        codec = VideoWriter_fourcc(*'XVID')  # Используем кодек XVID для AVI
        video_name = self.name + "." + self.extension
        writer = VideoWriter(video_name, codec, self.fps, self.size)

        if not writer.isOpened():
            return

        try:
            files = [file for file in os.listdir(self.img_dir)
                     if os.path.isfile(os.path.join(self.img_dir, file))]
        except Exception as e:
            return

        files.sort(key=self.sort_files)

        if len(files) == 0:
            return

        for i in range(len(files)):
            filename = os.path.join(self.img_dir, files[i])
            img = cv2.imread(filename)
            img = resize(img, self.size)
            writer.write(img)

        writer.release()

st.set_page_config(page_title="Мониторинг коров", layout="centered")
st.title("🐄 Мониторинг коров на ферме")

model = YOLO("weights.pt")
HISTORY_FILE = "history.json"
PROCESSED_FRAMES_DIR = "processed_frames"
os.makedirs(PROCESSED_FRAMES_DIR, exist_ok=True)

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png"])
uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.empty()
    image = Image.open(uploaded_file)
    st.image(image, caption="Исходное изображение")

    if st.button("🔍 Обработать изображение"):
        results = model(image, imgsz=640, conf=0.25)
        results[0].save(filename="result.jpg")

        st.image("result.jpg", caption="Обнаруженные коровы")

        cow_count = len(results[0].boxes)
        st.success(f"✅ Обнаружено коров: {cow_count}")

        # Сохраняем в историю
        record = {
            "filename": uploaded_file.name,
            "is_video": False,
            "frames_processed": None,
            "cow_count": cow_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            history = []

        history.append(record)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

if uploaded_video is not None:
    st.empty()
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)
    st.write(f"⏱ Видео длительностью {duration} сек ({fps} кадров/сек)")
    st.video(temp_video_path)

    if st.button("🔍 Обработать видео"):

        frame_interval = fps
        frame_idx = 0
        processed_frames = []
        total_cows = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)

                results = model(image_pil, imgsz=640, conf=0.25)
                cow_count = len(results[0].boxes)
                total_cows += cow_count

                annotated = results[0].plot()
                frame_output_path = os.path.join(PROCESSED_FRAMES_DIR, f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_output_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                processed_frames.append(frame_output_path)

            frame_idx += 1

        cap.release()
        st.success(f"✅ Обработано {len(processed_frames)} кадров. Обнаружено коров в совокупности: {total_cows}")

        sample_frame = cv2.imread(processed_frames[0])
        height, width, _ = sample_frame.shape

        writer = ImgToVid(PROCESSED_FRAMES_DIR, "processed_slideshow", "avi",
                          1, (width, height))
        writer.img_to_video()
        with open('processed_slideshow.avi', "rb") as f:
            st.download_button(
                label="📥 Скачать обработанное видео",
                data=f,
                file_name='result_video.avi',
                mime="video/avi"
            )

        record = {
            "filename": uploaded_video.name,
            "is_video": False,
            "frames_processed": len(processed_frames),
            "cow_count": total_cows,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            history = []
        history.append(record)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)

        os.remove('temp_video.mp4')

# Показ истории и экспорт
st.markdown("### 📊 История запросов")
if os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, "r") as f:
        history = json.load(f)
    df = pd.DataFrame(history)
    st.dataframe(df)

    import io
    @st.cache_data
    def convert_df(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()

    excel_data = convert_df(df)
    st.download_button("📥 Скачать отчёт в Excel", data=excel_data, file_name="cow_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("Пока нет данных.")

import streamlit as st
from ultralytics import YOLO
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# 1. โหลดโมเดล
model = YOLO("best.pt")

st.title("YOLO Live Webcam Detection")

class VideoProcessor:
    def recv(self, frame):
        # แปลงเฟรมจาก WebRTC เป็นรูปแบบที่ OpenCV/YOLO ใช้ได้
        img = frame.to_ndarray(format="bgr24")

        # 2. ให้ YOLO ทำการตรวจสอบ (ปรับ iou และ conf ตามที่คุณต้องการ)
        results = model(img, conf=0.4, iou=0.3)
        
        # 3. วาดผลลัพธ์ลงบนภาพ
        annotated_frame = results[0].plot()

        # 4. นับจำนวนวัตถุและแสดงบนจอ (แถมให้!)
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Count: {count}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 5. ส่วนแสดงผลบนหน้าเว็บ
webrtc_streamer(key="yolo-detection", video_processor_factory=VideoProcessor)

st.write("หมายเหตุ: การรันผ่านเว็บอาจมีความหน่วง (Lag) ขึ้นอยู่กับความเร็วอินเทอร์เน็ตและสเปคของ Server ฟรีครับ")
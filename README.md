# Smartbin - Smart Waste Classification System
## โครงการถังขยะอัจฉริยะด้วย YOLO Object Detection

**Smartbin** เป็นระบบจำแนกขยะอัตโนมัติแบบเรียลไทม์โดยใช้เทคโนโลยี YOLO และ Computer Vision สามารถตรวจจับและระบุประเภทของขยะผ่านกล้องเว็บแคมแบบสดๆ

### วัตถุประสงค์
- พัฒนาระบบจำแนกประเภทขยะอัตโนมัติด้วย AI แบบเรียลไทม์
- ศึกษาและประยุกต์ใช้เทคโนโลยี YOLO Object Detection
- สร้างเว็บแอปพลิเคชันที่ใช้งานง่ายและเข้าถึงได้จากทุกที่

### ฟีเจอร์หลัก
- **Real-time Detection**: ตรวจจับขยะแบบเรียลไทม์ผ่านกล้องเว็บแคม
- **Web-based Interface**: รันผ่านเว็บเบราว์เซอร์ด้วย Streamlit
- **Object Counting**: นับจำนวนวัตถุที่ตรวจพบในแต่ละเฟรม

### โครงสร้างโฟลเดอร์
```
Smartbin/
│
├── app.py                  # Streamlit web application
├── best.pt                 # โมเดล YOLOv8 ที่ฝึกแล้ว (20MB)
├── requirements.txt        # Python dependencies
├── packages.txt           # System packages สำหรับ deployment
└── README.md              # คู่มือการใช้งาน (ไฟล์นี้)
```

### เทคโนโลยีที่ใช้
- **YOLOv26** (Ultralytics): Object Detection model สำหรับตรวจจับวัตถุแบบเรียลไทม์
- **Streamlit**: Framework สำหรับสร้าง Web Application แบบ Python
- **Streamlit-WebRTC**: ส่วนขยายสำหรับการสตรีมวิดีโอจากกล้องเว็บแคม
- **OpenCV**: ประมวลผลภาพและวิดีโอ
- **PyAV**: การจัดการวิดีโอสตรีม

### การทำงานของระบบ

```
┌─────────────────┐
│   Webcam Input  │
│   (Live Video)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Video Frame Capture    │
│  (BGR24 Format)         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│   YOLO v26 Detection     │
│   - conf=0.4            │
│   - iou=0.3             │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Result Processing      │
│  - Draw Bounding Boxes  │
│  - Add Labels           │
│  - Count Objects        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Display on Web         │
│  (Annotated Frame)      │
└─────────────────────────┘
```
### สถาปัตยกรรมระบบ

```
┌────────────────────────────────────────┐
│         User Interface (Browser)       │
│              Streamlit UI              │
└──────────────┬─────────────────────────┘
               │
               │ WebRTC Stream
               │
┌──────────────▼─────────────────────────┐
│       Streamlit-WebRTC Layer           │
│   - Video Frame Capture                │
│   - Frame Format Conversion            │
└──────────────┬─────────────────────────┘
               │
               │ BGR24 Frames
               │
┌──────────────▼─────────────────────────┐
│         Video Processor                │
│   - Frame Processing                   │
│   - YOLO Inference                     │
│   - Result Annotation                  │
└──────────────┬─────────────────────────┘
               │
               │ Annotated Frames
               │
┌──────────────▼─────────────────────────┐
│      YOLOv8 Model (best.pt)            │
│   - Object Detection                   │
│   - Bounding Box Prediction            │
│   - Class Classification               │
└────────────────────────────────────────┘
```
### วิธีติดตั้งและใช้งาน

#### 1. Clone โครงการ
```bash
git clone https://github.com/Nawattakorn/Smartbin.git
cd Smartbin
```

#### 2. ติดตั้ง Dependencies

**สร้าง Virtual Environment** (แนะนำ):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**ติดตั้ง Python Packages**:
```bash
pip install -r requirements.txt
```

**สำหรับ Linux** (ติดตั้ง System Packages):
```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

#### 3. รันแอปพลิเคชัน
```bash
streamlit run app.py
```

เปิดเว็บเบราว์เซอร์ที่ `http://localhost:8501`

### การใช้งาน

1. **อนุญาตการเข้าถึงกล้อง**: เมื่อเปิดเว็บครั้งแรก จะมีการขออนุญาตเข้าถึงกล้องเว็บแคม
2. **กดปุ่ม START**: เริ่มการตรวจจับแบบเรียลไทม์
3. **วางขยะหน้ากล้อง**: ระบบจะแสดง:
   - กรอบสี่เหลี่ยมรอบวัตถุที่ตรวจพบ
   - ป้ายชื่อประเภทขยะ
   - ค่า Confidence Score
   - จำนวนวัตถุทั้งหมดที่ตรวจพบ
4. **กด STOP**: หยุดการตรวจจับ

### พารามิเตอร์ที่ปรับแต่งได้

ในไฟล์ `app.py` สามารถปรับค่าต่างๆ ได้:

```python
# Confidence Threshold (0.0-1.0)
# ค่าสูง = ต้องการความแน่ใจมากกว่าถึงจะแสดงผล
conf=0.4  # ค่าปัจจุบัน 40%

# IOU Threshold (0.0-1.0)
# ใช้ในการกรองกล่องที่ซ้อนทับกัน
iou=0.3   # ค่าปัจจุบัน 30%
```

### โมเดล YOLO ที่ใช้

- **ไฟล์**: `best.pt` (ขนาด ~20MB)
- **สถาปัตยกรรม**: YOLOv26 (Ultralytics)
- **ประเภทที่ตรวจจับได้**: ขึ้นอยู่กับการฝึกโมเดล (เช่น พลาสติก, กระดาษ, แก้ว, โลหะ, ฯลฯ)

### การพัฒนาต่อยอด

- [ ] เพิ่มประเภทขยะที่ตรวจจับได้มากขึ้น
- [ ] ระบบบันทึกสถิติการคัดแยกขยะ
- [ ] เชื่อมต่อกับฐานข้อมูลเพื่อเก็บข้อมูลการใช้งาน
- [ ] เพิ่มเสียงแจ้งเตือนเมื่อตรวจพบขยะ
- [ ] ระบบให้คะแนนการคัดแยกขยะ
- [ ] Mobile App (React Native/Flutter)
- [ ] ระบบ Multi-Camera Support
- [ ] Dashboard แสดงสถิติการใช้งาน
- [ ] API สำหรับเชื่อมต่อกับระบบอื่น
- [ ] ระบบ Auto-bin (เปิดฝาถังขยะอัตโนมัติ)

### ปัญหาที่อาจพบและแนวทางแก้ไข

| ปัญหา | สาเหตุ | แนวทางแก้ไข |
|-------|--------|--------------|
| กล้องไม่ทำงาน | Browser ไม่อนุญาตการเข้าถึง | ตรวจสอบการตั้งค่าความเป็นส่วนตัวของเบราว์เซอร์ |
| Lag หรือค้าง | Internet ช้าหรือ CPU ไม่แรงพอ | ลดความละเอียดกล้อง หรือใช้ GPU |
| ตรวจจับผิด | Confidence threshold ต่ำเกินไป | เพิ่มค่า conf เป็น 0.5-0.6 |
| Import Error | ไลบรารีไม่ครบ | ติดตั้ง requirements.txt ใหม่ |
| Model ไม่โหลด | ไฟล์ best.pt ไม่อยู่ | ตรวจสอบว่ามีไฟล์ best.pt ในโฟลเดอร์เดียวกับ app.py |

### Dependencies ที่ใช้

```
ultralytics          # YOLOv8 framework
streamlit            # Web application framework
streamlit-webrtc     # WebRTC component for Streamlit
av                   # Video/audio processing
opencv-python-headless  # Computer vision library (no GUI)
```

**System Packages** (Linux):
```
libgl1              # OpenGL library
libglib2.0-0        # GLib library
```

### การปรับแต่งโมเดล

หากต้องการฝึกโมเดลใหม่:

1. **เตรียมชุดข้อมูล** (Dataset):
   - รูปภาพของขยะแต่ละประเภท
   - ไฟล์ annotation (YOLO format)

2. **ฝึกโมเดล**:
```python
from ultralytics import YOLO

# โหลดโมเดลพื้นฐาน
model = YOLO('yolov8n.pt')

# ฝึกโมเดล
results = model.train(
    data='waste_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

3. **ทดสอบโมเดล**:
```python
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
```

4. **นำไปใช้**: คัดลอก `best.pt` มาแทนที่ไฟล์เดิม


### License
โครงการนี้พัฒนาขึ้นเพื่อการศึกษาในรายวิชา Computer Vision

### ผู้พัฒนา
- **นวัตกร** - [Nawattakorn](https://github.com/Nawattakorn)
- **อัทธเมศร์** - [Autthamet](https://github.com/b3y0und)
  
---

**หมายเหตุ**: 
- แนะนำให้รันบนเครื่องที่มี GPU สำหรับประสิทธิภาพที่ดีที่สุด
- การรันผ่าน Streamlit Cloud (ฟรี) อาจมี lag ขึ้นอยู่กับโหลดของ server
- หากต้องการความเร็วสูงสุด ให้ลด resolution ของกล้องหรือใช้โมเดลที่เล็กกว่า (yolov8n.pt)
- ระบบต้องการอินเทอร์เน็ตสำหรับการโหลดไลบรารีครั้งแรก

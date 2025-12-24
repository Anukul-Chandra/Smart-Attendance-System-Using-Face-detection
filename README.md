# Smart Attendance System Using Face Recognition

A real-time smart attendance system that automatically records attendance using face recognition and synchronizes data with Google Sheets. This project eliminates proxy attendance and manual errors found in traditional attendance systems.

---

## 📌 Problem Statement
Traditional attendance methods (paper-based or manual sign-in) are prone to proxy attendance, data manipulation, and inefficiency. This project addresses these issues by using computer vision and machine learning to verify identity through facial recognition and automatically mark attendance.

---

## 🎯 Objectives
- Automate attendance using face recognition
- Prevent duplicate attendance on the same day
- Store attendance securely in CSV format
- Sync attendance data to Google Sheets
- Provide a smooth user experience with automatic camera control

---

## 🧠 System Workflow
1. User enrollment with personal details and face images  
2. Face embeddings generation using a pretrained model  
3. Real-time face detection through webcam  
4. Face recognition using embedding comparison  
5. Automatic attendance marking  
6. Cloud synchronization with Google Sheets  

---

## 🛠️ Technologies Used
- Python  
- OpenCV  
- DeepFace (FaceNet)  
- NumPy  
- Google Sheets API  
- Google Drive API  
- Git & GitHub  

---

## 📂 Project Structure
```
Smart Attendance System/
├── enroll.py
├── embeddings.py
├── recognize_attendance.py
├── google_sheet.py
├── face_db/            # ignored
├── embeddings/         # ignored
├── attendance.csv      # ignored
├── users.csv           # ignored
├── .gitignore
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Anukul-Chandra/Smart-Attendance-System-Using-Face-detection.git
cd Smart-Attendance-System-Using-Face-detection
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv sas
source sas/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install opencv-python deepface numpy gspread google-auth
```

### 4️⃣ Google API Setup
- Enable Google Sheets API and Google Drive API  
- Create a Service Account  
- Download credentials.json (do NOT push to GitHub)  
- Share the Google Sheet with the service account (Editor)

### 5️⃣ Enroll User
```bash
python enroll.py
```

### 6️⃣ Generate Face Embeddings
```bash
python embeddings.py
```

### 7️⃣ Run Attendance System
```bash
python recognize_attendance.py
```

---

## ✅ Features
- Real-time face recognition  
- Automatic camera close after recognition  
- Duplicate attendance prevention per day  
- Google Sheets cloud synchronization  

---

## 🔐 Security Notes
- credentials.json is excluded via .gitignore  
- Face images and embeddings are not pushed to GitHub  

---

## 🔮 Future Improvements
- Anti-spoofing (photo/video attack prevention)  
- Multi-user continuous attendance mode  
- Web dashboard (Flask/FastAPI)  
- Reports & analytics  

---

## 👤 Author
Anukul Chandra  
BSc in Computer Science & Engineering  
Machine Learning & Computer Vision Enthusiast  

---

## 📜 License
This project is developed for educational and demonstration purposes.

import cv2
import os
import time
import csv

print("=== User Enrollment ===")

# -------- USER INFORMATION --------
name = input("Enter Name: ").strip()
age = input("Enter Age: ").strip()
phone = input("Enter Phone: ").strip()
position = input("Enter Position: ").strip()

# -------- SAVE USER INFO --------
csv_file = "users.csv"
file_exists = os.path.isfile(csv_file)

with open(csv_file, mode="a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["name", "age", "phone", "position"])
    writer.writerow([name, age, phone, position])

print("\n✔ User info saved")
input("Press ENTER to Verify Face...")

# -------- FACE DIRECTORY --------
user_dir = os.path.join("face_db", name.lower())
os.makedirs(user_dir, exist_ok=True)

# -------- FACE DETECTOR --------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------- CAMERA --------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot access webcam")
    exit()

# -------- CAPTURE SETTINGS --------
instructions = [
    "Look straight at the camera",
    "Turn your face slightly LEFT",
    "Turn your face slightly RIGHT",
    "Look slightly UP or DOWN"
]

img_count = 0
max_images = len(instructions)
last_capture = time.time()

print("\n📸 Face Verification Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80)
    )

    # Show instruction on screen
    cv2.putText(
        frame,
        instructions[img_count],
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Capture one image per instruction (1 sec gap)
        if time.time() - last_capture >= 1:
            face_img = frame[y:y+h, x:x+w]
            img_path = os.path.join(user_dir, f"img{img_count+1}.jpg")
            cv2.imwrite(img_path, face_img)
            print(f"✅ Saved: {img_path}")

            img_count += 1
            last_capture = time.time()
            time.sleep(0.5)

            if img_count >= max_images:
                print("🎉 Face enrollment completed!")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow("STEP 2 - Verify Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

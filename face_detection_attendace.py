import cv2
import numpy as np
import face_recognition
import urllib.request
import os
from datetime import datetime
import pandas as pd

# 🔹 ESP32-CAM URL
URL = "http://192.168.84.218/capture"

# 🔹 Load known images for face recognition
path = r"B:\Smart Attendance System (SAS)\image_folder"
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    file_path = r"B:\Smart Attendance System (SAS)\Attendance.xlsx"

    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(file_path, index=False, engine='openpyxl')

    df = pd.read_excel(file_path, engine='openpyxl')

    if "Name" not in df.columns or "Date" not in df.columns or "Time" not in df.columns:
        print("Error: Excel file does not have the correct column names. Creating a new file...")
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(file_path, index=False, engine='openpyxl')

    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%I:%M:%S %p')  # 12-hour format with AM/PM

    if not ((df["Name"] == name) & (df["Date"] == dateString)).any():
        new_entry = pd.DataFrame({"Name": [name], "Date": [dateString], "Time": [timeString]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_excel(file_path, index=False, engine='openpyxl')

    print(f"Attendance marked for {name} on {dateString} at {timeString}")

# 🔹 Encode faces from image folder
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# 🔥 Start ESP32-CAM Face Recognition
while True:
    try:
        # 🔹 Get frame from ESP32-CAM
        img_resp = urllib.request.urlopen(URL)
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)

        # 🔹 Resize & Convert for face recognition
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # 🔹 Detect faces in the frame
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                print(f"Recognized: {name}")

                # 🔹 Draw rectangle around face
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # 🔹 Mark attendance
                markAttendance(name)

        cv2.imshow("ESP32-CAM", img)

    except Exception as e:
        print("Failed to fetch frame from ESP32-CAM. Check the connection!", e)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

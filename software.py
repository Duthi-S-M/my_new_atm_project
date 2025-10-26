import cv2
import os
import numpy as np
import random
import csv
import face_recognition
from tkinter import *
from PIL import Image, ImageTk

# -------------------- PATHS --------------------
face_cascade_path = 'face_detection_model/deploy.prototxt'
face_model_path = 'face_detection_model/res10_300x300_ssd_iter_140000.caffemodel'
dataset_path = 'dataset/'
bank_csv = 'bank_details.csv'

# Load face detection model
face_detector = cv2.dnn.readNetFromCaffe(face_cascade_path, face_model_path)

# Ensure dataset directory exists
os.makedirs(dataset_path, exist_ok=True)

# -------------------- GUI --------------------
root = Tk()
root.title("ATM System")

# -------------------- FACE FUNCTIONS --------------------
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
    return faces

def capture_face(account_no):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_face(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = frame[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(dataset_path, f"{account_no}.jpg"), face_img)
            cv2.imshow("Capturing Face - Press q to exit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def verify_face(account_no):
    # Load registered face
    registered_face_path = os.path.join(dataset_path, f"{account_no}.jpg")
    if not os.path.exists(registered_face_path):
        return False

    registered_image = face_recognition.load_image_file(registered_face_path)
    registered_encoding = face_recognition.face_encodings(registered_image)
    if not registered_encoding:
        return False
    registered_encoding = registered_encoding[0]

    # Capture from webcam
    cap = cv2.VideoCapture(0)
    face_matched = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for encoding in face_encodings:
            matches = face_recognition.compare_faces([registered_encoding], encoding)
            if True in matches:
                face_matched = True
                cv2.putText(frame, "Face Verified", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face Not Matched", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Verification - Press q to confirm", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return face_matched


# -------------------- HELPER FUNCTIONS --------------------
def generate_unique_account():
    while True:
        account_no = str(random.randint(10000000, 99999999))
        if os.path.exists(bank_csv):
            with open(bank_csv, 'r') as f:
                if any(account_no == line.strip().split(',')[0] for line in f):
                    continue
        return account_no

def generate_unique_pin():
    return str(random.randint(1000, 9999))

# -------------------- REGISTRATION --------------------
def register_user():
    name = name_entry.get()
    phone = phone_entry.get()
    email = email_entry.get()
    balance = balance_entry.get()

    if not (name and phone and email and balance):
        status_label.config(text="Please fill all fields")
        return

    account_no = generate_unique_account()
    pin = generate_unique_pin()

    # Save to CSV
    file_exists = os.path.exists(bank_csv)
    with open(bank_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['AccountNo','PIN','Name','Phone','Email','Balance'])
        writer.writerow([account_no, pin, name, phone, email, balance])

    # Capture face
    capture_face(account_no)

    status_label.config(text=f"Registered! Account No: {account_no}, PIN: {pin}")

# -------------------- LOGIN --------------------
def login_user():
    account_no = account_entry.get()
    pin = pin_entry.get()

    if not (account_no and pin):
        login_status_label.config(text="Enter both Account No and PIN")
        return

    if not os.path.exists(bank_csv):
        login_status_label.config(text="No users registered yet")
        return

    user_found = False
    user_name = ""
    with open(bank_csv, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if row[0] == account_no and row[1] == pin:
                user_found = True
                user_name = row[2]
                break

    if not user_found:
        login_status_label.config(text="Invalid Account No or PIN")
        return

    # Face verification
    login_status_label.config(text="Please verify your face using webcam")
    root.update()
    if verify_face(account_no):
        login_status_label.config(text=f"Login Successful! Welcome {user_name}")
        show_atm_options(account_no)
    else:
        login_status_label.config(text="Face verification failed! Access Denied")

# -------------------- ATM FUNCTIONS --------------------
def show_atm_options(account_no):
    atm_window = Toplevel(root)
    atm_window.title("ATM Operations")

    # Balance display
    balance_lbl = Label(atm_window, text="")
    balance_lbl.pack()

    # Status label
    atm_status_lbl = Label(atm_window, text="")
    atm_status_lbl.pack(pady=5)

    def update_balance():
        with open(bank_csv, 'r') as f:
            reader = list(csv.reader(f))
        for row in reader:
            if row[0] == account_no:
                balance_lbl.config(text=f"Balance: ₹{row[5]}")
                break

    def deposit_amount():
        amt = deposit_entry.get()
        if not amt.isdigit():
            atm_status_lbl.config(text="Enter valid amount")
            return
        amt = int(amt)
        rows = list(csv.reader(open(bank_csv)))
        for row in rows:
            if row[0] == account_no:
                row[5] = str(int(row[5]) + amt)
        with open(bank_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        atm_status_lbl.config(text=f"Deposited ₹{amt}")
        update_balance()

    def withdraw_amount():
        amt = withdraw_entry.get()
        if not amt.isdigit():
            atm_status_lbl.config(text="Enter valid amount")
            return
        amt = int(amt)
        rows = list(csv.reader(open(bank_csv)))
        for row in rows:
            if row[0] == account_no:
                if int(row[5]) < amt:
                    atm_status_lbl.config(text="Insufficient Balance")
                    return
                row[5] = str(int(row[5]) - amt)
        with open(bank_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        atm_status_lbl.config(text=f"Withdrawn ₹{amt}")
        update_balance()

    def logout():
        atm_window.destroy()  # Close the ATM window and return to login
        login_status_label.config(text="You have logged out successfully.")

    # Deposit
    Label(atm_window, text="Deposit Amount:").pack()
    deposit_entry = Entry(atm_window)
    deposit_entry.pack()
    Button(atm_window, text="Deposit", command=deposit_amount).pack(pady=5)

    # Withdraw
    Label(atm_window, text="Withdraw Amount:").pack()
    withdraw_entry = Entry(atm_window)
    withdraw_entry.pack()
    Button(atm_window, text="Withdraw", command=withdraw_amount).pack(pady=5)

    # Exit button
    Button(atm_window, text="Logout / Exit", command=logout, bg="red", fg="white").pack(pady=10)

    update_balance()

# -------------------- GUI ELEMENTS --------------------
Label(root, text="--- Registration ---").pack(pady=5)
Label(root, text="Name:").pack()
name_entry = Entry(root)
name_entry.pack()
Label(root, text="Phone:").pack()
phone_entry = Entry(root)
phone_entry.pack()
Label(root, text="Email:").pack()
email_entry = Entry(root)
email_entry.pack()
Label(root, text="Initial Balance:").pack()
balance_entry = Entry(root)
balance_entry.pack()
Button(root, text="Register", command=register_user).pack(pady=5)

status_label = Label(root, text="")
status_label.pack(pady=5)

Label(root, text="--- Login ---").pack(pady=10)
Label(root, text="Account Number:").pack()
account_entry = Entry(root)
account_entry.pack()
Label(root, text="PIN:").pack()
pin_entry = Entry(root, show="*")
pin_entry.pack()
Button(root, text="Login", command=login_user).pack(pady=5)

login_status_label = Label(root, text="")
login_status_label.pack(pady=5)

root.mainloop()

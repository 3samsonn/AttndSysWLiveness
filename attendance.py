import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
from PIL import Image, ImageTk
from datetime import datetime, timedelta
import os

class AttendanceGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x800")
        self.window.configure(bg="#1e272e")

        # --- Data & Logic Setup ---
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_PATH = os.path.join(self.BASE_DIR, "dataset")
        self.TRAINER_PATH = os.path.join(self.BASE_DIR, "trainer")

        for p in [self.DATASET_PATH, self.TRAINER_PATH]:
            if not os.path.exists(p): os.makedirs(p)

        self.load_model()
        self.cap = cv2.VideoCapture(0)

        # --- State Variables ---
        self.current_reg_name = ""
        self.identified_name = "None"
        self.last_seen_name = "None"
        self.is_locked = False         
        self.lock_counter = 0          
        self.LOCK_THRESHOLD = 15       
        
        self.verification_active = False 
        self.auto_reg_active = False
        self.blink_count = 0
        self.eye_closed = False
        self.last_log_time = {} 

        # --- Registration Specifics ---
        self.is_counting_down = False
        self.reg_countdown = 0
        self.images_captured_in_session = 0
        self.last_countdown_time = datetime.now()

        # --- UI LAYOUT ---
        self.header = tk.Label(window, text="AUTOMATIC BIOMETRIC SYSTEM", font=("Segoe UI", 24, "bold"), bg="#1e272e", fg="#ffffff")
        self.header.pack(pady=10)

        self.main_container = tk.Frame(window, bg="#1e272e")
        self.main_container.pack(expand=True, fill=tk.BOTH, padx=20)

        self.cam_frame = tk.Frame(self.main_container, bg="#2f3542", bd=2)
        self.cam_frame.pack(side=tk.LEFT, padx=10)
        self.canvas = tk.Label(self.cam_frame, bg="black")
        self.canvas.pack()

        self.ctrl_panel = tk.Frame(self.main_container, bg="#1e272e", width=450)
        self.ctrl_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.ctrl_panel.pack_propagate(False) 

        self.user_display = tk.Label(self.ctrl_panel, text="STATUS: Idle", font=("Segoe UI", 16, "bold"), 
                                     bg="#1e272e", fg="#718093", width=25, anchor="w")
        self.user_display.pack(pady=10)

        self.btn_verify = tk.Button(self.ctrl_panel, text="ACTIVATE SENSOR", command=self.activate_verification,
                                   bg="#f39c12", fg="white", font=("Segoe UI", 12, "bold"), width=25)
        self.btn_verify.pack(pady=5)

        tk.Label(self.ctrl_panel, text="--- ADMIN CONTROLS ---", font=("Segoe UI", 10), bg="#1e272e", fg="#718093").pack(pady=(20,0))
        self.btn_reg = tk.Button(self.ctrl_panel, text="AUTO REGISTRATION", command=self.start_auto_registration,
                                 bg="#3498db", fg="white", font=("Segoe UI", 11, "bold"), width=25)
        self.btn_reg.pack(pady=5)

        self.btn_train = tk.Button(self.ctrl_panel, text="TRAIN DATABASE", command=self.train_model,
                                   bg="#9b59b6", fg="white", font=("Segoe UI", 11), width=25)
        self.btn_train.pack(pady=5)

        self.tree = ttk.Treeview(self.ctrl_panel, columns=('name', 'time', 'status'), show='headings', height=12)
        self.tree.heading('name', text='Name'); self.tree.heading('time', text='Time'); self.tree.heading('status', text='Action')
        self.tree.column('name', width=100); self.tree.column('time', width=120); self.tree.column('status', width=70)
        self.tree.pack(pady=20, fill=tk.X)

        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.window.mainloop()

    def load_model(self):
        model_p, label_p = os.path.join(self.TRAINER_PATH, "trainer.yml"), os.path.join(self.TRAINER_PATH, "labels.npy")
        if os.path.exists(model_p) and os.path.exists(label_p):
            self.recognizer.read(model_p)
            self.label_map = np.load(label_p, allow_pickle=True).item()
        else: self.label_map = {}

    def activate_verification(self):
        self.is_locked = False
        self.lock_counter = 0
        self.blink_count = 0
        self.verification_active = True
        self.btn_verify.config(state=tk.DISABLED, text="SENSOR ACTIVE")

    def show_auto_closing_msg(self, message, timeout):
        """Creates a non-blocking popup that closes after specified ms."""
        popup = tk.Toplevel(self.window)
        popup.title("Auto Log Notification")
        popup.geometry("400x180")
        popup.configure(bg="#2f3640")
        
        # Center popup
        x = self.window.winfo_x() + (self.window.winfo_width() // 2) - 200
        y = self.window.winfo_y() + (self.window.winfo_height() // 2) - 90
        popup.geometry(f"+{x}+{y}")

        tk.Label(popup, text=message, font=("Segoe UI", 14, "bold"), bg="#2f3640", fg="white", pady=30).pack()
        
        # Progress bar to show time remaining
        progress = ttk.Progressbar(popup, length=300, mode='determinate')
        progress.pack(pady=5)
        progress['value'] = 100
        
        popup.after(timeout, popup.destroy)
        tk.Button(popup, text="CLOSE", command=popup.destroy, bg="#e84118", fg="white", width=15).pack(pady=10)

    def log_attendance_auto(self, name):
        now = datetime.now()
        if name in self.last_log_time:
            if now < self.last_log_time[name] + timedelta(minutes=1):
                self.reset_after_log()
                return 

        status = "IN"
        if os.path.exists("attendance.csv"):
            with open("attendance.csv", "r") as f:
                lines = f.readlines()
                for line in reversed(lines):
                    parts = line.strip().split(",")
                    if len(parts) >= 3 and parts[0] == name:
                        status = "OUT" if parts[2] == "IN" else "IN"
                        break

        with open("attendance.csv", "a") as f:
            f.write(f"{name},{now.strftime('%Y-%m-%d %H:%M:%S')},{status}\n")
        
        self.tree.insert('', 0, values=(name, now.strftime("%H:%M:%S"), status))
        self.last_log_time[name] = now
        
        log_msg = f"Goodbye! {status} logged for {name}" if status == "OUT" else f"Welcome! {status} logged for {name}"
        
        self.reset_after_log()
        # Trigger the 5-second auto-closing popup
        self.show_auto_closing_msg(log_msg, 5000)

    def reset_after_log(self):
        self.is_locked = False
        self.blink_count = 0
        self.verification_active = False
        self.btn_verify.config(state=tk.NORMAL, text="ACTIVATE SENSOR")
        self.user_display.config(text="STATUS: Idle", fg="#718093")

    def start_auto_registration(self):
        name = simpledialog.askstring("Input", "Enter Full Name:")
        if name:
            self.current_reg_name = name.replace(" ", "_")
            self.user_path = os.path.join(self.DATASET_PATH, self.current_reg_name)
            if not os.path.exists(self.user_path): os.makedirs(self.user_path)
            
            self.reg_countdown = 5
            self.images_captured_in_session = 0
            self.is_counting_down = True
            self.auto_reg_active = True
            self.last_countdown_time = datetime.now()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            h_img, w_img, _ = frame.shape
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # --- REGISTRATION ---
            if self.auto_reg_active:
                zw, zh = 300, 300
                x1, y1 = (w_img - zw)//2, (h_img - zh)//2
                cv2.rectangle(frame, (x1, y1), (x1+zw, y1+zh), (255, 255, 255), 2)

                if self.is_counting_down:
                    if (datetime.now() - self.last_countdown_time).seconds >= 1:
                        self.reg_countdown -= 1
                        self.last_countdown_time = datetime.now()
                    cv2.putText(frame, f"STARTING IN: {self.reg_countdown}", (x1+10, y1-20), 1, 2, (0, 255, 255), 2)
                    if self.reg_countdown <= 0: self.is_counting_down = False
                else:
                    for (x, y, w, h) in faces:
                        if x > x1-20 and y > y1-20 and (x+w) < x1+zw+20 and (y+h) < y1+zh+20:
                            if self.images_captured_in_session < 100:
                                self.images_captured_in_session += 1
                                # UNIQUE NAME TIMESTAMP
                                t_stamp = datetime.now().strftime('%H%M%S_%f')
                                i_name = f"{self.current_reg_name}_{t_stamp}.jpg"
                                cv2.imwrite(os.path.join(self.user_path, i_name), gray[y:y+h, x:x+w])
                                cv2.putText(frame, f"SAVING: {self.images_captured_in_session}/100", (x1, y1-20), 1, 1.5, (0, 255, 0), 2)
                            else:
                                self.auto_reg_active = False
                                messagebox.showinfo("Success", f"Capture for {self.current_reg_name} Complete!")

            # --- VERIFICATION ---
            elif self.verification_active:
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    if not self.is_locked:
                        if self.label_map:
                            lbl, conf = self.recognizer.predict(roi_gray)
                            if conf < 75:
                                name = self.label_map.get(lbl, "Unknown")
                                if name == self.last_seen_name: self.lock_counter += 1
                                else: self.last_seen_name, self.lock_counter = name, 0
                                if self.lock_counter >= self.LOCK_THRESHOLD:
                                    self.is_locked, self.identified_name = True, name
                            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 2)
                    else:
                        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 15)
                        if len(eyes) == 0:
                            if not self.eye_closed: self.eye_closed = True
                        else:
                            if self.eye_closed:
                                self.blink_count += 1
                                self.eye_closed = False
                        
                        if self.blink_count >= 2:
                            # Trigger log
                            self.window.after(100, lambda: self.log_attendance_auto(self.identified_name))
                            self.blink_count = 0 

                        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 165, 0), 2)
                        cv2.putText(frame, f"BLINK TO LOG: {self.blink_count}/2", (x, y-10), 1, 0.7, (255, 165, 0), 2)

                self.user_display.config(text=f"IDENTIFIED: {self.identified_name}", fg="#f1c40f")

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.imgtk = imgtk; self.canvas.configure(image=imgtk)

        self.window.after(10, self.update_frame)

    def train_model(self):
        faces, ids, label_to_id = [], [], {}
        folders = [f for f in os.listdir(self.DATASET_PATH) if os.path.isdir(os.path.join(self.DATASET_PATH, f))]
        for current_id, folder in enumerate(folders):
            f_path = os.path.join(self.DATASET_PATH, folder)
            label_to_id[current_id] = folder
            for img in os.listdir(f_path):
                pil_img = Image.open(os.path.join(f_path, img)).convert('L')
                faces.append(np.array(pil_img, 'uint8')); ids.append(current_id)
        if faces:
            self.recognizer.train(faces, np.array(ids))
            self.recognizer.save(os.path.join(self.TRAINER_PATH, "trainer.yml"))
            np.save(os.path.join(self.TRAINER_PATH, "labels.npy"), label_to_id)
            self.load_model()
            messagebox.showinfo("Success", "Training Complete!")

    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    AttendanceGUI(tk.Tk(), "Zero-Touch Attendance V2") 
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
        self.window.geometry("1250x850")
        
        # --- OLFU COLORS ---
        self.OLFU_GREEN = "#006837"
        self.OLFU_GOLD = "#f1c40f"
        self.OLFU_LIGHT_GREEN = "#27ae60"
        self.BG_WHITE = "#f5f6fa"
        self.TEXT_DARK = "#2f3640"

        self.window.configure(bg=self.BG_WHITE)

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

        # --- UI STYLING ---
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background=self.OLFU_GREEN, foreground="white")
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 10))
        style.map("Treeview", background=[('selected', self.OLFU_LIGHT_GREEN)])

        # --- TOP HEADER ---
        self.header = tk.Frame(window, bg=self.OLFU_GREEN, height=80)
        self.header.pack(side=tk.TOP, fill=tk.X)
        
        self.header_label = tk.Label(self.header, text="OUR LADY OF FATIMA UNIVERSITY", 
                                     font=("Segoe UI", 22, "bold"), bg=self.OLFU_GREEN, fg="white")
        self.header_label.pack(pady=10)
        self.sub_label = tk.Label(self.header, text="OLFU Employee Attendance System", 
                                  font=("Segoe UI", 10), bg=self.OLFU_GREEN, fg=self.OLFU_GOLD)
        self.sub_label.pack()

        # --- MAIN CONTAINER ---
        self.main_container = tk.Frame(window, bg=self.BG_WHITE)
        self.main_container.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

        # --- LEFT SIDE: CAMERA VIEW ---
        self.left_frame = tk.Frame(self.main_container, bg="white", bd=0)
        self.left_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        self.cam_title = tk.Label(self.left_frame, text="LIVE MONITORING", font=("Segoe UI", 12, "bold"), 
                                  bg="white", fg=self.OLFU_GREEN)
        self.cam_title.pack(anchor="w", pady=(0, 10))

        self.canvas = tk.Label(self.left_frame, bg="black", bd=5, relief=tk.FLAT)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- RIGHT SIDE: SIDEBAR ---
        self.sidebar = tk.Frame(self.main_container, bg="white", width=400)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        self.sidebar.pack_propagate(False)

        # Status Card
        self.status_card = tk.Frame(self.sidebar, bg=self.OLFU_GREEN, padx=15, pady=15)
        self.status_card.pack(fill=tk.X, padx=10, pady=10)
        
        self.user_display = tk.Label(self.status_card, text="STATUS: Ready", font=("Segoe UI", 14, "bold"), 
                                     bg=self.OLFU_GREEN, fg="white")
        self.user_display.pack()

        # Control Buttons
        self.btn_verify = tk.Button(self.sidebar, text="START SCANNING", command=self.activate_verification,
                                   bg=self.OLFU_LIGHT_GREEN, fg="white", font=("Segoe UI", 12, "bold"), 
                                   relief=tk.FLAT, height=2, cursor="hand2")
        self.btn_verify.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(self.sidebar, text="ADMINISTRATOR TOOLS", font=("Segoe UI", 9, "bold"), 
                 bg="white", fg="#95a5a6").pack(pady=(20, 5))

        self.btn_reg = tk.Button(self.sidebar, text="Register New User", command=self.start_auto_registration,
                                 bg=self.TEXT_DARK, fg="white", font=("Segoe UI", 10), relief=tk.FLAT)
        self.btn_reg.pack(fill=tk.X, padx=20, pady=5)

        self.btn_train = tk.Button(self.sidebar, text="Sync & Train Database", command=self.train_model,
                                   bg=self.OLFU_GOLD, fg=self.TEXT_DARK, font=("Segoe UI", 10, "bold"), relief=tk.FLAT)
        self.btn_train.pack(fill=tk.X, padx=20, pady=5)

        # Log Table
        self.log_label = tk.Label(self.sidebar, text="RECENT LOGS", font=("Segoe UI", 10, "bold"), 
                                  bg="white", fg=self.OLFU_GREEN)
        self.log_label.pack(anchor="w", padx=20, pady=(20, 5))
        
        self.tree = ttk.Treeview(self.sidebar, columns=('name', 'time', 'status'), show='headings', height=10)
        self.tree.heading('name', text='Employee Name'); self.tree.heading('time', text='Time'); self.tree.heading('status', text='Log')
        self.tree.column('name', width=120); self.tree.column('time', width=100); self.tree.column('status', width=60)
        self.tree.pack(padx=20, fill=tk.X)

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
        self.btn_verify.config(state=tk.DISABLED, text="SCANNER ACTIVE", bg="#bdc3c7")
        self.user_display.config(text="STATUS: Scanning Faces...", fg="white")

    def show_auto_closing_msg(self, message, timeout):
        popup = tk.Toplevel(self.window)
        popup.title("OLFU Security Alert")
        popup.geometry("400x180")
        popup.configure(bg=self.OLFU_GREEN)
        x = self.window.winfo_x() + (self.window.winfo_width() // 2) - 200
        y = self.window.winfo_y() + (self.window.winfo_height() // 2) - 90
        popup.geometry(f"+{x}+{y}")
        tk.Label(popup, text=message, font=("Segoe UI", 14, "bold"), bg=self.OLFU_GREEN, fg="white", pady=30, justify=tk.CENTER).pack()
        progress = ttk.Progressbar(popup, length=300, mode='determinate')
        progress.pack(pady=5)
        progress['value'] = 100
        popup.after(timeout, popup.destroy)

    def log_attendance_auto(self, name):
        if not name or name == "None":
            self.reset_after_log()
            return
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
        display_name = name.replace("_", " ")
        log_msg = f"WELCOME BACK!\n{display_name}" if status == "IN" else f"STAY SAFE!\n{display_name}"
        self.reset_after_log()
        self.show_auto_closing_msg(log_msg, 3000)

    def reset_after_log(self):
        self.is_locked = False
        self.blink_count = 0
        self.eye_closed = False
        self.identified_name = "None"
        self.last_seen_name = "None"
        self.lock_counter = 0
        self.user_display.config(text="STATUS: Scanning for Next User", fg="white")

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

            if self.auto_reg_active:
                zw, zh = 300, 300
                x1, y1 = (w_img - zw)//2, (h_img - zh)//2
                cv2.rectangle(frame, (x1, y1), (x1+zw, y1+zh), (255, 255, 255), 2)
                instructions = {0: "LOOK CENTER", 1: "TILT CHIN UP", 2: "TILT CHIN DOWN", 3: "TURN LEFT", 4: "TURN RIGHT", 5: "GLASSES OFF"}
                current_stage = self.images_captured_in_session // 20
                if self.is_counting_down:
                    if (datetime.now() - self.last_countdown_time).seconds >= 1:
                        self.reg_countdown -= 1
                        self.last_countdown_time = datetime.now()
                    cv2.putText(frame, f"STAGE: {instructions.get(current_stage)}", (x1, y1-40), 1, 1.5, (0, 255, 255), 2)
                    cv2.putText(frame, f"STARTING IN: {self.reg_countdown}", (x1, y1-15), 1, 1.5, (255, 255, 255), 2)
                    if self.reg_countdown <= 0: self.is_counting_down = False
                else:
                    for (x, y, w, h) in faces:
                        if x > x1-20 and y > y1-20 and (x+w) < x1+zw+20 and (y+h) < y1+zh+20:
                            if self.images_captured_in_session < 120:
                                self.images_captured_in_session += 1
                                t_stamp = datetime.now().strftime('%H%M%S_%f')
                                i_name = f"{self.current_reg_name}_{t_stamp}.jpg"
                                cv2.imwrite(os.path.join(self.user_path, i_name), gray[y:y+h, x:x+w])
                                cv2.putText(frame, f"CAPTURING: {instructions.get(current_stage)}", (x1, y1-15), 1, 1.2, (0, 255, 0), 2)
                                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                                if self.images_captured_in_session % 20 == 0 and self.images_captured_in_session < 120:
                                    self.is_counting_down = True
                                    self.reg_countdown = 3 
                            else:
                                self.auto_reg_active = False
                                messagebox.showinfo("OLFU Systems", "Profile Created Successfully!")

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
                            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 210, 255), 2)
                    else:
                        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 15)
                        if len(eyes) == 0:
                            if not self.eye_closed: self.eye_closed = True
                        else:
                            if self.eye_closed:
                                self.blink_count += 1
                                self.eye_closed = False
                        if self.blink_count >= 2:
                            current_name = self.identified_name
                            self.window.after(100, lambda: self.log_attendance_auto(current_name))
                            self.blink_count = 0 
                        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"BLINK TO LOG: {self.blink_count}/2", (x, y-10), 1, 1, (0, 255, 0), 2)
                
                display_text = f"DETECTED: {self.identified_name.replace('_', ' ')}" if self.is_locked else "SCANNING..."
                self.user_display.config(text=display_text)

            imgtk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.imgtk = imgtk; self.canvas.configure(image=imgtk)

        self.window.after(10, self.update_frame)

    def train_model(self):
        """Modified to ensure ALL folders in the dataset are included every time."""
        faces, ids, label_to_id = [], [], {}
        
        # Get all subdirectories (folders) in the dataset folder
        folders = [f for f in os.listdir(self.DATASET_PATH) if os.path.isdir(os.path.join(self.DATASET_PATH, f))]
        
        if not folders:
            messagebox.showwarning("OLFU Systems", "No datasets found to train!")
            return

        for current_id, folder in enumerate(folders):
            f_path = os.path.join(self.DATASET_PATH, folder)
            label_to_id[current_id] = folder  # Map the folder name to an ID
            
            # Get all images in the folder
            for img in os.listdir(f_path):
                if img.endswith(".jpg") or img.endswith(".png"):
                    try:
                        pil_img = Image.open(os.path.join(f_path, img)).convert('L') # Convert to Grayscale
                        img_np = np.array(pil_img, 'uint8')
                        faces.append(img_np)
                        ids.append(current_id)
                    except Exception as e:
                        print(f"Error loading image {img}: {e}")

        if faces:
            # Train the LBPH Recognizer with ALL faces and ALL IDs
            self.recognizer.train(faces, np.array(ids))
            
            # Save the trained model and the labels
            self.recognizer.save(os.path.join(self.TRAINER_PATH, "trainer.yml"))
            np.save(os.path.join(self.TRAINER_PATH, "labels.npy"), label_to_id)
            
            # Reload the model into memory
            self.load_model()
            
            messagebox.showinfo("OLFU Systems", f"Successfully trained {len(folders)} profile(s)!")
        else:
            messagebox.showerror("OLFU Systems", "No valid images found in the datasets.")

    def on_closing(self):
        if self.cap.isOpened(): self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    AttendanceGUI(tk.Tk(), "OLFU Attendance System")
import cv2
import os
import time
import numpy as np
from datetime import datetime
import json
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import threading
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm  # Progress bar for data splitting

class ModernDataCollectionGUI:
    def __init__(self, root=None):
        """Khởi tạo GUI thu thập dữ liệu"""
        if root is None:
            self.root = tk.Tk()
        else:
            self.root = root
        self.setup_window()
        
        # Camera variables
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.capture_thread = None
        
        # Data collection variables
        self.current_class = None
        self.is_auto_recording = False
        self.image_count = {"no_phone": 0, "using_phone": 0}
        self.session_stats = {
            "start_time": None,
            "total_images": 0,
            "images_per_class": {"no_phone": 0, "using_phone": 0}
        }
        self.last_auto_capture = 0
        
        # Configuration
        self.config = {
            "image_size": (640, 480),
            "save_size": (224, 224),
            "fps": 30,
            "quality": 95,
            "auto_capture_interval": 0.5,
            "camera_id": 0
        }
        
        self.setup_directories()
        self.load_existing_counts()
        self.create_interface()
        self.update_stats_display()
        
    def setup_window(self):
        """Thiết lập cửa sổ chính"""
        self.root.title("Thu Thập Dữ Liệu - Nhận Diện Điện Thoại")
        self.root.geometry("1000x600")
        self.root.configure(bg='#2c3e50')
        self.root.resizable(True, True)
        
        # Icon (nếu có)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass
            
    def setup_directories(self):
        """Tạo cấu trúc thư mục"""
        directories = [
            "data/no_phone",
            "data/using_phone"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def load_existing_counts(self):
        """Đếm số ảnh hiện có"""
        for class_name in ["no_phone", "using_phone"]:
            path = f"data/{class_name}"
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                self.image_count[class_name] = count
            
    def create_interface(self):
        """Tạo giao diện chính"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left side - Camera view
        self.create_camera_section(main_frame)
        
        # Right side - Control panel
        self.create_control_panel(main_frame)
        
    def create_camera_section(self, parent):
        """Tạo phần hiển thị camera bên trái"""
        camera_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=2)
        camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera title
        title_label = tk.Label(camera_frame, text="📷 CAMERA VIEW", 
                              font=('Arial', 14, 'bold'),
                              bg='#34495e', fg='#ecf0f1')
        title_label.pack(pady=10)
        
        # Camera display
        self.camera_label = tk.Label(camera_frame, bg='#2c3e50', 
                                    text="Camera chưa được khởi động\nNhấn 'Bắt Đầu Camera' để bắt đầu",
                                    font=('Arial', 12),
                                    fg='#bdc3c7')
        self.camera_label.pack(expand=True, padx=10, pady=10)
        
        # Status bar under camera
        self.status_frame = tk.Frame(camera_frame, bg='#34495e')
        self.status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.status_label = tk.Label(self.status_frame, text="Trạng thái: Chờ khởi động",
                                    font=('Arial', 10, 'bold'),
                                    bg='#34495e', fg='#f39c12')
        self.status_label.pack(side=tk.LEFT)
        
        self.mode_label = tk.Label(self.status_frame, text="",
                                  font=('Arial', 10, 'bold'),
                                  bg='#34495e', fg='#e74c3c')
        self.mode_label.pack(side=tk.RIGHT)
        
    def create_control_panel(self, parent):
        """Tạo bảng điều khiển bên phải"""
        control_frame = tk.Frame(parent, bg='#34495e', relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))
        control_frame.configure(width=350)  # Fixed width
        
        # Title
        title_label = tk.Label(control_frame, text="🎛️ BẢNG ĐIỀU KHIỂN", 
                              font=('Arial', 14, 'bold'),
                              bg='#34495e', fg='#ecf0f1')
        title_label.pack(pady=15)
        
        # Camera controls
        self.create_camera_controls(control_frame)
        
        # Mode selection
        self.create_mode_selection(control_frame)
        
        # Capture controls
        self.create_capture_controls(control_frame)
        
        # Statistics
        self.create_statistics_section(control_frame)
        
        # Data management
        self.create_data_management(control_frame)
        
        # Instructions
        self.create_instructions(control_frame)
        
    def create_camera_controls(self, parent):
        """Tạo điều khiển camera"""
        frame = tk.LabelFrame(parent, text="📹 Điều Khiển Camera", 
                             font=('Arial', 10, 'bold'),
                             bg='#34495e', fg='#ecf0f1')
        frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Camera ID
        id_frame = tk.Frame(frame, bg='#34495e')
        id_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(id_frame, text="Camera ID:", bg='#34495e', fg='#bdc3c7',
                font=('Arial', 9)).pack(side=tk.LEFT)
        
        self.camera_id_var = tk.IntVar(value=0)
        camera_id_spin = tk.Spinbox(id_frame, from_=0, to=10, width=5,
                                   textvariable=self.camera_id_var,
                                   font=('Arial', 9))
        camera_id_spin.pack(side=tk.RIGHT)
        
        # Auto capture interval
        interval_frame = tk.Frame(frame, bg='#34495e')
        interval_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(interval_frame, text="Khoảng cách (s):", bg='#34495e', fg='#bdc3c7',
                font=('Arial', 9)).pack(side=tk.LEFT)
        
        self.interval_var = tk.DoubleVar(value=0.5)
        interval_spin = tk.Spinbox(interval_frame, from_=0.1, to=5.0, increment=0.1,
                                  width=8, textvariable=self.interval_var,
                                  font=('Arial', 9))
        interval_spin.pack(side=tk.RIGHT)
        
        # Camera button
        self.camera_btn = tk.Button(frame, text="▶️ Bắt Đầu Camera",
                                   command=self.toggle_camera,
                                   bg='#27ae60', fg='white',
                                   font=('Arial', 10, 'bold'),
                                   relief=tk.RAISED, bd=2)
        self.camera_btn.pack(fill=tk.X, padx=10, pady=10)
        
    def create_mode_selection(self, parent):
        """Tạo lựa chọn chế độ"""
        frame = tk.LabelFrame(parent, text="📱 Chế Độ Thu Thập", 
                             font=('Arial', 10, 'bold'),
                             bg='#34495e', fg='#ecf0f1')
        frame.pack(fill=tk.X, padx=15, pady=10)
        
        self.mode_var = tk.StringVar()
        
        # No phone mode
        no_phone_btn = tk.Radiobutton(frame, text="🚫 Không dùng điện thoại",
                                     variable=self.mode_var, value="no_phone",
                                     command=self.set_mode,
                                     bg='#34495e', fg='#2ecc71',
                                     selectcolor='#27ae60',
                                     font=('Arial', 10, 'bold'))
        no_phone_btn.pack(anchor=tk.W, padx=10, pady=5)
        
        # Using phone mode
        using_phone_btn = tk.Radiobutton(frame, text="📱 Đang dùng điện thoại",
                                        variable=self.mode_var, value="using_phone",
                                        command=self.set_mode,
                                        bg='#34495e', fg='#e74c3c',
                                        selectcolor='#c0392b',
                                        font=('Arial', 10, 'bold'))
        using_phone_btn.pack(anchor=tk.W, padx=10, pady=5)
        
    def create_capture_controls(self, parent):
        """Tạo điều khiển chụp ảnh"""
        frame = tk.LabelFrame(parent, text="📸 Điều Khiển Chụp", 
                             font=('Arial', 10, 'bold'),
                             bg='#34495e', fg='#ecf0f1')
        frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Manual capture
        manual_btn = tk.Button(frame, text="📸 Chụp Thủ Công",
                              command=self.manual_capture,
                              bg='#3498db', fg='white',
                              font=('Arial', 10, 'bold'))
        manual_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Auto capture toggle
        self.auto_btn = tk.Button(frame, text="🔴 Bắt Đầu Tự Động",
                                 command=self.toggle_auto_capture,
                                 bg='#e74c3c', fg='white',
                                 font=('Arial', 10, 'bold'))
        self.auto_btn.pack(fill=tk.X, padx=10, pady=5)
        
    def create_statistics_section(self, parent):
        """Tạo phần thống kê"""
        frame = tk.LabelFrame(parent, text="📊 Thống Kê", 
                             font=('Arial', 10, 'bold'),
                             bg='#34495e', fg='#ecf0f1')
        frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Current counts
        self.stats_labels = {}
        
        # No phone count
        no_phone_frame = tk.Frame(frame, bg='#34495e')
        no_phone_frame.pack(fill=tk.X, padx=10, pady=3)
        
        tk.Label(no_phone_frame, text="🚫 Không dùng ĐT:", 
                bg='#34495e', fg='#2ecc71', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        self.stats_labels['no_phone'] = tk.Label(no_phone_frame, text="0", 
                                                bg='#34495e', fg='#ecf0f1', 
                                                font=('Arial', 9, 'bold'))
        self.stats_labels['no_phone'].pack(side=tk.RIGHT)
        
        # Using phone count
        using_phone_frame = tk.Frame(frame, bg='#34495e')
        using_phone_frame.pack(fill=tk.X, padx=10, pady=3)
        
        tk.Label(using_phone_frame, text="📱 Đang dùng ĐT:", 
                bg='#34495e', fg='#e74c3c', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        self.stats_labels['using_phone'] = tk.Label(using_phone_frame, text="0", 
                                                   bg='#34495e', fg='#ecf0f1', 
                                                   font=('Arial', 9, 'bold'))
        self.stats_labels['using_phone'].pack(side=tk.RIGHT)
        
        # Session stats
        session_frame = tk.Frame(frame, bg='#34495e')
        session_frame.pack(fill=tk.X, padx=10, pady=3)
        
        tk.Label(session_frame, text="🎯 Phiên này:", 
                bg='#34495e', fg='#f39c12', font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        
        self.stats_labels['session'] = tk.Label(session_frame, text="0", 
                                               bg='#34495e', fg='#ecf0f1', 
                                               font=('Arial', 9, 'bold'))
        self.stats_labels['session'].pack(side=tk.RIGHT)
        
    def create_data_management(self, parent):
        """Tạo quản lý dữ liệu"""
        frame = tk.LabelFrame(parent, text="💾 Quản Lý Dữ Liệu", 
                             font=('Arial', 10, 'bold'),
                             bg='#34495e', fg='#ecf0f1')
        frame.pack(fill=tk.X, padx=15, pady=10)
        
        # Split data
        split_btn = tk.Button(frame, text="📂 Phân Chia Dữ liệu YOLO",
                             command=self.split_data_to_yolo,
                             bg='#9b59b6', fg='white',
                             font=('Arial', 10, 'bold'))
        split_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Reset session
        reset_btn = tk.Button(frame, text="🔄 Reset Thống Kê",
                             command=self.reset_session,
                             bg='#e67e22', fg='white',
                             font=('Arial', 10, 'bold'))
        reset_btn.pack(fill=tk.X, padx=10, pady=5)
        
    def create_instructions(self, parent):
        """Tạo hướng dẫn"""
        frame = tk.LabelFrame(parent, text="ℹ️ Hướng Dẫn", 
                             font=('Arial', 10, 'bold'),
                             bg='#34495e', fg='#ecf0f1')
        frame.pack(fill='both', expand=True, padx=15, pady=10)
        
        instructions = [
            "1. Chọn camera ID và bắt đầu camera",
            "2. Chọn chế độ thu thập dữ liệu",
            "3. Sử dụng chụp thủ công hoặc tự động",
            "4. Theo dõi thống kê realtime",
            "5. Phân chia dữ liệu khi hoàn thành"
        ]
        
        for i, instruction in enumerate(instructions):
            tk.Label(frame, text=instruction, 
                    bg='#34495e', fg='#bdc3c7',
                    font=('Arial', 8),
                    justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=2)
            
    def toggle_camera(self):
        """Bật/tắt camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
            
    def start_camera(self):
        """Khởi động camera"""
        try:
            camera_id = self.camera_id_var.get()
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                raise Exception(f"Không thể mở camera ID: {camera_id}")
                
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["image_size"][0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["image_size"][1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config["fps"])
            
            self.is_running = True
            self.camera_btn.configure(text="⏹️ Dừng Camera", bg='#e74c3c')
            self.status_label.configure(text="Trạng thái: Đang chạy", fg='#2ecc71')
            
            # Start session
            if not self.session_stats["start_time"]:
                self.session_stats["start_time"] = datetime.now()
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
            
        except Exception as e:
            messagebox.showerror("Lỗi Khởi Động", f"Không thể khởi động camera:\n{str(e)}")

            
    def stop_camera(self):
        """Dừng camera"""
        self.is_running = False
        self.is_auto_recording = False
        
        if self.camera:
            self.camera.release()
            self.camera = None
            
        self.camera_btn.configure(text="▶️ Bắt Đầu Camera", bg='#27ae60')
        self.auto_btn.configure(text="🔴 Bắt Đầu Tự Động", bg='#e74c3c')
        self.status_label.configure(text="Trạng thái: Đã dừng", fg='#e74c3c')
        self.mode_label.configure(text="")
        
        # Clear camera display
        self.camera_label.configure(image="", text="Camera đã dừng")
        
    def capture_loop(self):
        """Vòng lặp capture camera"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                continue
                
            # Flip frame
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            # Auto capture
            if (self.is_auto_recording and self.current_class and 
                time.time() - self.last_auto_capture >= self.interval_var.get()):
                self.save_image(frame)
                self.last_auto_capture = time.time()
            
            # Convert to display format
            self.display_frame(frame)
            
            time.sleep(1/30)  # ~30 FPS
            
    def display_frame(self, frame):
        """Hiển thị frame lên GUI"""
        if not self.is_running:
            return
            
        # Add overlay information
        overlay_frame = self.add_overlay(frame.copy())
        
        # Convert to PIL format
        rgb_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize to fit display
        display_size = (600, 450)  # Adjust as needed
        pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Update label
        self.camera_label.configure(image=photo, text="")
        self.camera_label.image = photo  # Keep reference
        
    def add_overlay(self, frame):
        """Thêm thông tin overlay lên frame"""
        height, width = frame.shape[:2]
        
        # Add recording indicator
        if self.is_auto_recording:
            cv2.circle(frame, (width-30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (width-60, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add mode indicator
        if self.current_class:
            mode_text = "DANG DUNG DIEN THOAI" if self.current_class == "using_phone" else "KHONG DUNG DIEN THOAI"
            color = (0, 0, 255) if self.current_class == "using_phone" else (0, 255, 0)
            
            cv2.rectangle(frame, (10, height-40), (width-10, height-10), color, 2)
            cv2.putText(frame, mode_text, (20, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
        
    def set_mode(self):
        """Thiết lập chế độ thu thập"""
        self.current_class = self.mode_var.get()
        
        if self.current_class == "no_phone":
            self.mode_label.configure(text="🚫 Không dùng ĐT", fg='#2ecc71')
        elif self.current_class == "using_phone":
            self.mode_label.configure(text="📱 Đang dùng ĐT", fg='#e74c3c')
            
    def manual_capture(self):
        """Chụp ảnh thủ công"""
        if not self.current_frame is None and self.current_class:
            self.save_image(self.current_frame)
        elif not self.current_class:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn chế độ thu thập trước!")
        else:
            messagebox.showwarning("Cảnh báo", "Camera chưa sẵn sàng!")
            
    def toggle_auto_capture(self):
        """Bật/tắt chụp tự động"""
        if not self.current_class:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn chế độ thu thập trước!")
            return
            
        self.is_auto_recording = not self.is_auto_recording
        
        if self.is_auto_recording:
            self.auto_btn.configure(text="⏹️ Dừng Tự Động", bg='#2ecc71')
            self.last_auto_capture = time.time()
        else:
            self.auto_btn.configure(text="🔴 Bắt Đầu Tự Động", bg='#e74c3c')
            
    def save_image(self, frame):
        """Lưu ảnh"""
        if not self.current_class:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.current_class}_{timestamp}_{self.image_count[self.current_class]:06d}.jpg"
        filepath = f"data/{self.current_class}/{filename}"
        
        # Resize image before saving
        img_resized = cv2.resize(frame, self.config["save_size"])
        
        # Save image
        cv2.imwrite(filepath, img_resized, [cv2.IMWRITE_JPEG_QUALITY, self.config["quality"]])
        
        # Update counts
        self.image_count[self.current_class] += 1
        self.session_stats["total_images"] += 1
        self.session_stats["images_per_class"][self.current_class] += 1
        
        # Update display
        self.update_stats_display()
        
        print(f"📸 Đã lưu: {filename}")
        
    def count_images_on_disk(self, class_name):
        """Đếm số ảnh thực tế trên ổ đĩa cho class_name"""
        path = f"data/{class_name}"
        if os.path.exists(path):
            return len([f for f in os.listdir(path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        return 0

    def update_stats_display(self):
        """Cập nhật hiển thị thống kê từ ổ đĩa"""
        no_phone_count = self.count_images_on_disk('no_phone')
        using_phone_count = self.count_images_on_disk('using_phone')
        self.stats_labels['no_phone'].configure(text=str(no_phone_count))
        self.stats_labels['using_phone'].configure(text=str(using_phone_count))
        self.stats_labels['session'].configure(text=str(self.session_stats['total_images']))

    def verify_dataset_structure(self):
        """Kiểm tra cấu trúc dữ liệu YOLO"""
        print("="*50)
        print("Kiểm tra Cấu trúc Dữ liệu")
        print("="*50)
        
        required_dirs = [
            "data/train/images",
            "data/train/labels", 
            "data/val/images",
            "data/val/labels"
        ]
        
        total_train_images = 0
        total_train_labels = 0
        total_val_images = 0
        total_val_labels = 0
        
        for dir_path in required_dirs:
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                file_count = len(files)
                print(f"✓ {dir_path}: {file_count} files")
                
                # Count files for verification
                if "train/images" in dir_path:
                    total_train_images = file_count
                elif "train/labels" in dir_path:
                    total_train_labels = file_count
                elif "val/images" in dir_path:
                    total_val_images = file_count
                elif "val/labels" in dir_path:
                    total_val_labels = file_count
            else:
                print(f"✗ {dir_path}: Missing!")
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✓ Created {dir_path}")
                
        # Verify image-label correspondence
        print(f"\nKiểm tra tính nhất quán dữ liệu:")
        print(f"Train - Images: {total_train_images}, Labels: {total_train_labels} {'✓' if total_train_images == total_train_labels else '✗'}")
        print(f"Val - Images: {total_val_images}, Labels: {total_val_labels} {'✓' if total_val_images == total_val_labels else '✗'}")
        
        # Check for potential issues and show message box
        message = ""
        if total_train_images == 0:
            message += "⚠️ Không tìm thấy ảnh train!\n"
        if total_val_images == 0:
            message += "⚠️ Không tìm thấy ảnh validation!\n"
        
        if message:
            messagebox.showwarning("Cảnh báo", message)
        elif total_train_images == total_train_labels and total_val_images == total_val_labels:
            messagebox.showinfo("Thành công", "Cấu trúc dữ liệu hợp lệ!\n" +
                              f"Train: {total_train_images} ảnh\n" +
                              f"Validation: {total_val_images} ảnh")
        else:
            messagebox.showerror("Lỗi", "Số lượng ảnh và nhãn không khớp!\n" +
                               f"Train: {total_train_images} ảnh, {total_train_labels} nhãn\n" +
                               f"Validation: {total_val_images} ảnh, {total_val_labels} nhãn")
        
        print("="*50)
        
    def split_data_to_yolo(self):
        """Phân chia dữ liệu thành định dạng YOLO với progress bar"""
        print("="*50)
        print("Bắt đầu phân chia dữ liệu YOLO...")
        print("="*50)
        
        # Kiểm tra dữ liệu gốc
        no_phone_path = "data/no_phone"
        using_phone_path = "data/using_phone"
        
        if not os.path.exists(no_phone_path) or not os.path.exists(using_phone_path):
            messagebox.showerror("Lỗi", "Không tìm thấy thư mục dữ liệu gốc!")
            return
            
        # Đếm số ảnh
        no_phone_images = [f for f in os.listdir(no_phone_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        using_phone_images = [f for f in os.listdir(using_phone_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(no_phone_images) == 0 or len(using_phone_images) == 0:
            messagebox.showerror("Lỗi", "Cần có ít nhất 1 ảnh cho mỗi class!")
            return
            
        total_images = len(no_phone_images) + len(using_phone_images)
        print(f"Tổng số ảnh: {total_images}")
        print(f"- Không dùng điện thoại: {len(no_phone_images)}")
        print(f"- Đang dùng điện thoại: {len(using_phone_images)}")
        
        # Tạo cấu trúc thư mục YOLO
        yolo_dirs = [
            "data/train/images",
            "data/train/labels",
            "data/val/images", 
            "data/val/labels"
        ]
        
        for dir_path in yolo_dirs:
            os.makedirs(dir_path, exist_ok=True)
            # Xóa file cũ
            for file in os.listdir(dir_path):
                os.remove(os.path.join(dir_path, file))
        
        # Tỷ lệ phân chia (80% train, 20% val)
        train_ratio = 0.8
        
        # Tổng số file cần xử lý (mỗi ảnh tạo 1 ảnh + 1 label)
        total_operations = total_images * 2
        
        # Tạo progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Đang phân chia dữ liệu...")
        progress_window.geometry("400x150")
        progress_window.configure(bg='#2c3e50')
        progress_window.resizable(False, False)
        
        # Center the progress window
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        tk.Label(progress_window, text="Đang phân chia dữ liệu YOLO...", 
                font=('Arial', 12, 'bold'), bg='#2c3e50', fg='#ecf0f1').pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(progress_window, variable=progress_var, 
                                      maximum=100, length=350)
        progress_bar.pack(pady=10)
        
        status_label = tk.Label(progress_window, text="Chuẩn bị...", 
                               font=('Arial', 10), bg='#2c3e50', fg='#bdc3c7')
        status_label.pack(pady=5)
        
        # Update progress function
        def update_progress(current, total, message):
            progress = (current / total) * 100
            progress_var.set(progress)
            status_label.configure(text=message)
            progress_window.update()
        
        def process_class_data(images, class_name, class_id):
            """Xử lý dữ liệu cho một class"""
            np.random.shuffle(images)  # Trộn ngẫu nhiên
            
            split_index = int(len(images) * train_ratio)
            train_images = images[:split_index]
            val_images = images[split_index:]
            
            processed = 0
            
            # Xử lý train images
            for img_name in train_images:
                # Copy image
                src_path = os.path.join(f"data/{class_name}", img_name)
                dst_path = os.path.join("data/train/images", img_name)
                shutil.copy2(src_path, dst_path)
                
                # Tạo label file
                label_name = os.path.splitext(img_name)[0] + ".txt"
                label_path = os.path.join("data/train/labels", label_name)
                
                # Tạo bounding box giả (full image)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
                processed += 2  # 1 image + 1 label
                update_progress(processed, total_operations, 
                              f"Xử lý train {class_name}: {len(train_images)} ảnh")
            
            # Xử lý validation images
            for img_name in val_images:
                # Copy image
                src_path = os.path.join(f"data/{class_name}", img_name)
                dst_path = os.path.join("data/val/images", img_name)
                shutil.copy2(src_path, dst_path)
                
                # Tạo label file
                label_name = os.path.splitext(img_name)[0] + ".txt"
                label_path = os.path.join("data/val/labels", label_name)
                
                # Tạo bounding box giả (full image)
                with open(label_path, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                
                processed += 2  # 1 image + 1 label
                update_progress(processed, total_operations,
                              f"Xử lý val {class_name}: {len(val_images)} ảnh")
            
            return len(train_images), len(val_images), processed
        
        try:
            total_processed = 0
            
            # Xử lý class "no_phone" (class_id = 0)
            train_no_phone, val_no_phone, processed = process_class_data(
                no_phone_images, "no_phone", 0)
            total_processed += processed
            
            # Xử lý class "using_phone" (class_id = 1)  
            train_using_phone, val_using_phone, processed = process_class_data(
                using_phone_images, "using_phone", 1)
            total_processed += processed
            
            # Tạo file cấu hình YOLO
            self.create_yolo_config(train_no_phone + train_using_phone,
                                  val_no_phone + val_using_phone)
            
            # Hoàn thành
            update_progress(total_operations, total_operations, "Hoàn thành!")
            
            # Hiển thị kết quả
            result_message = f"""✅ Phân chia dữ liệu thành công!

📊 Thống kê:
• Train: {train_no_phone + train_using_phone} ảnh
  - Không dùng ĐT: {train_no_phone}
  - Đang dùng ĐT: {train_using_phone}

• Validation: {val_no_phone + val_using_phone} ảnh  
  - Không dùng ĐT: {val_no_phone}
  - Đang dùng ĐT: {val_using_phone}

📁 Cấu trúc đã tạo:
• data/train/images/ & labels/
• data/val/images/ & labels/
• data/dataset.yaml (cấu hình YOLO)"""
            
            print("\n" + "="*50)
            print("KẾT QUẢ PHÂN CHIA DỮ LIỆU")
            print("="*50)
            print(f"✅ Train: {train_no_phone + train_using_phone} ảnh")
            print(f"✅ Validation: {val_no_phone + val_using_phone} ảnh")
            print(f"✅ Tổng cộng: {total_images} ảnh được xử lý")
            print("="*50)
            
            # Đóng progress window trước khi hiện messagebox
            progress_window.destroy()
            
            messagebox.showinfo("Thành công", result_message)
            
        except Exception as e:
            progress_window.destroy()
            error_msg = f"Lỗi khi phân chia dữ liệu:\n{str(e)}"
            print(f"❌ {error_msg}")
            messagebox.showerror("Lỗi", error_msg)
    
    def create_yolo_config(self, train_count, val_count):
        """Tạo file cấu hình YOLO"""
        config = {
            'path': './data',  # Đường dẫn dataset root
            'train': 'train/images',  # Đường dẫn train images (relative to path)
            'val': 'val/images',      # Đường dẫn val images (relative to path)
            'test': '',               # Đường dẫn test images (optional)
            
            'nc': 2,  # Số lượng classes
            'names': ['no_phone', 'using_phone']  # Tên các classes
        }
        
        # Lưu file YAML
        config_path = "data/dataset.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Đã tạo file cấu hình: {config_path}")
        
        # Tạo thêm file README
        readme_content = f"""# Phone Usage Detection Dataset

## Thông tin Dataset
- **Tổng số ảnh**: {train_count + val_count}
- **Classes**: 2 (no_phone, using_phone)
- **Train**: {train_count} ảnh
- **Validation**: {val_count} ảnh
- **Ngày tạo**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Cấu trúc thư mục
```
data/
├── train/
│   ├── images/     # Ảnh training
│   └── labels/     # Label files (.txt)
├── val/
│   ├── images/     # Ảnh validation  
│   └── labels/     # Label files (.txt)
├── dataset.yaml    # Cấu hình YOLO
└── README.md       # File này

no_phone/           # Dữ liệu gốc - không dùng điện thoại
using_phone/        # Dữ liệu gốc - đang dùng điện thoại
```

## Sử dụng với YOLO
```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # hoặc yolov8s.pt, yolov8m.pt, etc.

# Train
model.train(data='data/dataset.yaml', epochs=100, imgsz=640)
```

## Classes
- **0**: no_phone - Không sử dụng điện thoại
- **1**: using_phone - Đang sử dụng điện thoại

## Ghi chú
- Tất cả bounding box được thiết lập full image (0.5 0.5 1.0 1.0)
- Định dạng label: class_id center_x center_y width height (normalized)
- Dataset được tạo tự động bởi Data Collection GUI
"""
        
        readme_path = "data/README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print(f"✅ Đã tạo file README: {readme_path}")
            
    def reset_session(self):
        """Reset thống kê phiên làm việc"""
        if messagebox.askyesno("Xác nhận", "Bạn có chắc muốn reset thống kê phiên làm việc?"):
            self.session_stats = {
                "start_time": datetime.now() if self.is_running else None,
                "total_images": 0,
                "images_per_class": {"no_phone": 0, "using_phone": 0}
            }
            self.update_stats_display()
            messagebox.showinfo("Thành công", "Đã reset thống kê phiên làm việc!")
            
    def on_closing(self):
        """Xử lý khi đóng ứng dụng"""
        if self.is_running:
            if messagebox.askokcancel("Thoát", "Camera đang chạy. Bạn có muốn thoát?"):
                self.stop_camera()
                self.root.quit()
        else:
            self.root.quit()

def main():
    """Hàm chính"""
    print("="*60)
    print("🚀 KHỞI ĐỘNG HỆ THỐNG THU THẬP DỮ LIỆU")
    print("📱 Nhận diện sử dụng điện thoại")
    print("="*60)
    
    root = tk.Tk()
    app = ModernDataCollectionGUI(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    print("✅ Giao diện đã sẵn sàng!")
    print("💡 Hướng dẫn sử dụng:")
    print("   1. Chọn Camera ID và khởi động camera")
    print("   2. Chọn chế độ thu thập (có/không điện thoại)")
    print("   3. Sử dụng chụp thủ công hoặc tự động")
    print("   4. Theo dõi tiến độ qua thống kê")
    print("   5. Phân chia dữ liệu YOLO khi hoàn thành")
    print("="*60)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n🛑 Dừng chương trình bằng Ctrl+C")
        if app.is_running:
            app.stop_camera()
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {e}")
    finally:
        print("👋 Tạm biệt!")

if __name__ == "__main__":
    main()
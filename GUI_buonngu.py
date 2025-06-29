import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import threading
import queue
import cv2
from PIL import Image, ImageTk
import os
import json
import numpy as np
from datetime import datetime
import time

# Import thư viện cần thiết để tạo class trực tiếp
import mediapipe as mp
import yaml
import shutil
from sklearn.model_selection import train_test_split
import random

class ThuThapDuLieuYOLO:
    def __init__(self):
        # Khởi tạo MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Các chỉ số landmark cho mắt
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Cấu hình
        self.NGUONG_BUON_NGU = 0.25
        self.kich_thuoc_anh = 640
        self.classes = ['alert', 'drowsy']
        self.enable_augmentation = True
        
        # Thư mục lưu trữ
        self.thu_muc_raw = "raw_data"
        self.thu_muc_dataset = "yolo_dataset"
        
        # Đếm số mẫu
        self.so_mau_alert = 0
        self.so_mau_drowsy = 0
        
        # Lịch sử EAR
        self.history_ear = []
        
        # Tạo thư mục
        self.tao_thu_muc()
        self.tao_file_cau_hinh()
        
    def tao_thu_muc(self):
        """Tạo cấu trúc thư mục"""
        # Raw data
        os.makedirs(os.path.join(self.thu_muc_raw, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.thu_muc_raw, "labels"), exist_ok=True)
        
        # YOLO dataset
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.thu_muc_dataset, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.thu_muc_dataset, "labels", split), exist_ok=True)
    
    def tao_file_cau_hinh(self):
        """Tạo file cấu hình YOLO"""
        config = {
            'train': f'./yolo_dataset/images/train',
            'val': f'./yolo_dataset/images/val',
            'test': f'./yolo_dataset/images/test',
            'nc': 2,
            'names': self.classes
        }
        
        with open(os.path.join(self.thu_muc_dataset, "data.yaml"), 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def tinh_ear(self, landmarks, mat_trai=True):
        """Tính Eye Aspect Ratio"""
        if mat_trai:
            eye_landmarks = self.LEFT_EYE_LANDMARKS
        else:
            eye_landmarks = self.RIGHT_EYE_LANDMARKS
        
        # Lấy tọa độ các điểm mắt
        points = []
        for idx in eye_landmarks:
            x = landmarks.landmark[idx].x
            y = landmarks.landmark[idx].y
            points.append([x, y])
        
        points = np.array(points)
        
        # Tính EAR đơn giản
        # Chiều cao mắt / chiều rộng mắt
        height1 = np.linalg.norm(points[1] - points[5])  
        height2 = np.linalg.norm(points[2] - points[4])  
        width = np.linalg.norm(points[0] - points[3])    
        
        if width == 0:
            return 0.0
        
        ear = (height1 + height2) / (2.0 * width)
        return ear
    
    def danh_gia_buon_ngu_v2(self, face_landmarks):
        """Đánh giá tình trạng buồn ngủ phiên bản 2"""
        try:
            # Tính EAR cho cả hai mắt
            ear_left = self.tinh_ear(face_landmarks, mat_trai=True)
            ear_right = self.tinh_ear(face_landmarks, mat_trai=False)
            ear_avg = (ear_left + ear_right) / 2.0
            
            # Lưu vào lịch sử
            self.history_ear.append(ear_avg)
            if len(self.history_ear) > 30:  # Giữ 30 frame gần nhất
                self.history_ear.pop(0)
            
            # Đánh giá trạng thái
            is_drowsy = ear_avg < self.NGUONG_BUON_NGU
            confidence = abs(ear_avg - self.NGUONG_BUON_NGU) / self.NGUONG_BUON_NGU
            confidence = min(confidence, 1.0)
            
            return ear_avg, is_drowsy, confidence
            
        except Exception as e:
            print(f"Lỗi khi đánh giá buồn ngủ: {e}")
            return 0.0, False, 0.0
    
    def tao_bbox_mat(self, face_landmarks, width, height):
        """Tạo bounding box cho khuôn mặt"""
        x_coords = []
        y_coords = []
        
        for landmark in face_landmarks.landmark:
            x_coords.append(landmark.x * width)
            y_coords.append(landmark.y * height)
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Mở rộng bbox một chút
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(width, x_max + padding)
        y_max = min(height, y_max + padding)
        
        # Chuyển về format YOLO (normalized)
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Normalize
        center_x /= width
        center_y /= height
        bbox_width /= width
        bbox_height /= height
        
        return center_x, center_y, bbox_width, bbox_height
    
    def luu_mau_yolo_v2(self, frame, face_landmarks, loai_mau=None):
        """Lưu mẫu YOLO với auto classification"""
        try:
            if loai_mau is None:
                # Auto classify
                ear_avg, is_drowsy, confidence = self.danh_gia_buon_ngu_v2(face_landmarks)
                if confidence < 0.3:  # Confidence thấp, bỏ qua
                    return False
                loai_mau = 1 if is_drowsy else 0
            
            # Tạo tên file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            class_name = self.classes[loai_mau]
            ten_file = f"{class_name}_{timestamp}"
            
            # Đường dẫn file
            duong_dan_anh = os.path.join(self.thu_muc_raw, "images", f"{ten_file}.jpg")
            duong_dan_label = os.path.join(self.thu_muc_raw, "labels", f"{ten_file}.txt")
            
            # Resize ảnh
            frame_resized = cv2.resize(frame, (self.kich_thuoc_anh, self.kich_thuoc_anh))
            
            # Lưu ảnh
            if not cv2.imwrite(duong_dan_anh, frame_resized):
                return False
            
            # Tạo bounding box
            h, w = frame_resized.shape[:2]
            center_x, center_y, bbox_width, bbox_height = self.tao_bbox_mat(face_landmarks, w, h)
            
            # Lưu label
            with open(duong_dan_label, 'w') as f:
                f.write(f"{loai_mau} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            
            # Cập nhật đếm
            if loai_mau == 0:
                self.so_mau_alert += 1
            else:
                self.so_mau_drowsy += 1
            
            return True
            
        except Exception as e:
            print(f"Lỗi khi lưu mẫu: {e}")
            return False
    
    def chia_du_lieu_train_val_test(self, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Chia dữ liệu thành train/val/test"""
        try:
            # Lấy danh sách file ảnh
            image_dir = os.path.join(self.thu_muc_raw, "images")
            if not os.path.exists(image_dir):
                return False
            
            image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
            
            if len(image_files) == 0:
                print("Không có dữ liệu để chia!")
                return False
            
            # Chia theo class để đảm bảo balance
            alert_files = [f for f in image_files if f.startswith('alert_')]
            drowsy_files = [f for f in image_files if f.startswith('drowsy_')]
            
            # Chia từng class
            def split_files(files, train_r, val_r, test_r):
                if len(files) == 0:
                    return [], [], []
                
                train_files, temp_files = train_test_split(files, test_size=(val_r + test_r), random_state=42)
                
                if len(temp_files) >= 2:
                    val_files, test_files = train_test_split(temp_files, test_size=test_r/(val_r + test_r), random_state=42)
                else:
                    val_files = temp_files
                    test_files = []
                
                return train_files, val_files, test_files
            
            alert_train, alert_val, alert_test = split_files(alert_files, train_ratio, val_ratio, test_ratio)
            drowsy_train, drowsy_val, drowsy_test = split_files(drowsy_files, train_ratio, val_ratio, test_ratio)
            
            # Kết hợp và copy files
            splits = {
                'train': alert_train + drowsy_train,
                'val': alert_val + drowsy_val,
                'test': alert_test + drowsy_test
            }
            
            for split_name, files in splits.items():
                for file_name in files:
                    # Copy ảnh
                    src_img = os.path.join(self.thu_muc_raw, "images", file_name)
                    dst_img = os.path.join(self.thu_muc_dataset, "images", split_name, file_name)
                    shutil.copy2(src_img, dst_img)
                    
                    # Copy label
                    label_name = file_name.replace('.jpg', '.txt')
                    src_label = os.path.join(self.thu_muc_raw, "labels", label_name)
                    dst_label = os.path.join(self.thu_muc_dataset, "labels", split_name, label_name)
                    
                    if os.path.exists(src_label):
                        shutil.copy2(src_label, dst_label)
            
            print(f"Đã chia dữ liệu: Train({len(splits['train'])}), Val({len(splits['val'])}), Test({len(splits['test'])})")
            return True
            
        except Exception as e:
            print(f"Lỗi khi chia dữ liệu: {e}")
            return False
    
    def kiem_tra_chat_luong_dataset(self):
        """Kiểm tra chất lượng dataset"""
        try:
            issues = []
            
            # Kiểm tra số lượng mẫu
            raw_img_dir = os.path.join(self.thu_muc_raw, "images")
            if os.path.exists(raw_img_dir):
                alert_count = len([f for f in os.listdir(raw_img_dir) if f.startswith('alert_')])
                drowsy_count = len([f for f in os.listdir(raw_img_dir) if f.startswith('drowsy_')])
                
                if alert_count < 10:
                    issues.append(f"Ít mẫu Alert: {alert_count}")
                if drowsy_count < 10:
                    issues.append(f"Ít mẫu Drowsy: {drowsy_count}")
                
                if alert_count > 0 and drowsy_count > 0:
                    ratio = alert_count / drowsy_count
                    if ratio > 3 or ratio < 0.33:
                        issues.append(f"Mất cân bằng class: {ratio:.2f}")
            
            # Kiểm tra file label bị thiếu
            for img_file in os.listdir(raw_img_dir):
                if img_file.endswith('.jpg'):
                    label_file = img_file.replace('.jpg', '.txt')
                    label_path = os.path.join(self.thu_muc_raw, "labels", label_file)
                    if not os.path.exists(label_path):
                        issues.append(f"Thiếu label: {img_file}")
            
            if issues:
                print("Các vấn đề phát hiện:")
                for issue in issues:
                    print(f"  - {issue}")
                return False
            else:
                print("Dataset có chất lượng tốt!")
                return True
                
        except Exception as e:
            print(f"Lỗi khi kiểm tra: {e}")
            return False
    
    def xoa_tat_ca_du_lieu(self):
        """Xóa tất cả dữ liệu"""
        try:
            # Xóa raw data
            if os.path.exists(self.thu_muc_raw):
                shutil.rmtree(self.thu_muc_raw)
            
            # Xóa dataset
            if os.path.exists(self.thu_muc_dataset):
                shutil.rmtree(self.thu_muc_dataset)
            
            # Tạo lại thư mục
            self.tao_thu_muc()
            self.tao_file_cau_hinh()
            
            # Reset counters
            self.so_mau_alert = 0
            self.so_mau_drowsy = 0
            self.history_ear = []
            
            print("Đã xóa tất cả dữ liệu!")
            
        except Exception as e:
            print(f"Lỗi khi xóa dữ liệu: {e}")
    
    def luu_metadata(self):
        """Lưu metadata"""
        metadata = {
            'threshold': self.NGUONG_BUON_NGU,
            'image_size': self.kich_thuoc_anh,
            'classes': self.classes,
            'alert_count': self.so_mau_alert,
            'drowsy_count': self.so_mau_drowsy,
            'created': datetime.now().isoformat()
        }
        
        try:
            with open(os.path.join(self.thu_muc_raw, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Lỗi lưu metadata: {e}")


class YOLODataCollectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Drowsiness Data Collector - GUI Interface")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Khởi tạo collector
        self.collector = ThuThapDuLieuYOLO()
        
        # Camera variables
        self.cap = None
        self.camera_active = False
        self.auto_save = False
        self.frame_count = 0
        self.save_interval = 30
        self.last_save_time = 0
        
        # GUI variables
        self.current_frame = None
        self.camera_frame = None
        self.stats_text = None
        
        # Threading variables
        self.camera_thread = None
        self.update_queue = queue.Queue()
        
        # Setup GUI
        self.setup_gui()
        self.setup_styles()
        self.update_stats_display()
        
        # Start GUI update loop
        self.root.after(100, self.process_queue)
    
    def setup_styles(self):
        """Thiết lập style cho GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), 
                       background='#2b2b2b', foreground='#ffffff')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), 
                       background='#2b2b2b', foreground='#00ff00')
        style.configure('Info.TLabel', font=('Arial', 10), 
                       background='#2b2b2b', foreground='#ffffff')
        style.configure('Success.TButton', font=('Arial', 10, 'bold'))
        style.configure('Warning.TButton', font=('Arial', 10, 'bold'))
        style.configure('Danger.TButton', font=('Arial', 10, 'bold'))
    
    def setup_gui(self):
        """Thiết lập giao diện chính"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="YOLO Drowsiness Data Collector", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right panel - Camera and stats
        self.setup_display_panel(main_frame)
    
    def setup_control_panel(self, parent):
        """Thiết lập panel điều khiển"""
        control_frame = ttk.LabelFrame(parent, text="Điều khiển", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Camera controls
        cam_frame = ttk.LabelFrame(control_frame, text="Camera", padding="5")
        cam_frame.pack(fill='x', pady=(0, 10))
        
        self.start_cam_btn = ttk.Button(cam_frame, text="Bắt đầu Camera", 
                                       command=self.toggle_camera, style='Success.TButton')
        self.start_cam_btn.pack(fill='x', pady=2)
        
        self.auto_save_var = tk.BooleanVar()
        auto_save_check = ttk.Checkbutton(cam_frame, text="Tự động lưu", 
                                         variable=self.auto_save_var, 
                                         command=self.toggle_auto_save)
        auto_save_check.pack(anchor='w', pady=2)
        
        # Manual save controls
        save_frame = ttk.LabelFrame(control_frame, text="Lưu mẫu thủ công", padding="5")
        save_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(save_frame, text="Lưu Alert (Tỉnh táo)", 
                  command=self.save_alert_manual).pack(fill='x', pady=2)
        ttk.Button(save_frame, text="Lưu Drowsy (Buồn ngủ)", 
                  command=self.save_drowsy_manual).pack(fill='x', pady=2)
        
        # Dataset management
        dataset_frame = ttk.LabelFrame(control_frame, text="Quản lý Dataset", padding="5")
        dataset_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Button(dataset_frame, text="Chia dữ liệu Train/Val/Test", 
                  command=self.split_dataset).pack(fill='x', pady=2)
        ttk.Button(dataset_frame, text="Kiểm tra chất lượng", 
                  command=self.check_quality).pack(fill='x', pady=2)
        ttk.Button(dataset_frame, text="Xóa tất cả dữ liệu", 
                  command=self.clear_all_data, style='Danger.TButton').pack(fill='x', pady=2)
        
        # Configuration
        config_frame = ttk.LabelFrame(control_frame, text="Cấu hình", padding="5")
        config_frame.pack(fill='x', pady=(0, 10))
        
        # EAR threshold
        ear_frame = ttk.Frame(config_frame)
        ear_frame.pack(fill='x', pady=2)
        ttk.Label(ear_frame, text="Ngưỡng EAR:").pack(side='left')
        self.ear_var = tk.DoubleVar(value=self.collector.NGUONG_BUON_NGU)
        ear_scale = ttk.Scale(ear_frame, from_=0.1, to=0.4, variable=self.ear_var, 
                             orient='horizontal', command=self.update_ear_threshold)
        ear_scale.pack(side='right', fill='x', padx=(5, 0))
        
        # Save interval
        interval_frame = ttk.Frame(config_frame)
        interval_frame.pack(fill='x', pady=2)
        ttk.Label(interval_frame, text="Khoảng lưu (frames):").pack(side='left')
        self.interval_var = tk.IntVar(value=self.save_interval)
        interval_scale = ttk.Scale(interval_frame, from_=10, to=60, variable=self.interval_var, 
                                  orient='horizontal', command=self.update_save_interval)
        interval_scale.pack(side='right', fill='x', padx=(5, 0))
        
        # Data augmentation
        self.aug_var = tk.BooleanVar(value=self.collector.enable_augmentation)
        aug_check = ttk.Checkbutton(config_frame, text="Data Augmentation", 
                                   variable=self.aug_var, command=self.toggle_augmentation)
        aug_check.pack(anchor='w', pady=2)
        
        # Reset button
        ttk.Button(config_frame, text="Reset EAR History", 
                  command=self.reset_ear_history).pack(fill='x', pady=2)
    
    def setup_display_panel(self, parent):
        """Thiết lập panel hiển thị"""
        display_frame = ttk.Frame(parent)
        display_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        display_frame.columnconfigure(0, weight=1)
        display_frame.rowconfigure(0, weight=1)
        display_frame.rowconfigure(1, weight=1)
        
        # Camera display
        cam_frame = ttk.LabelFrame(display_frame, text="Camera View", padding="5")
        cam_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 5))
        cam_frame.columnconfigure(0, weight=1)
        cam_frame.rowconfigure(0, weight=1)
        
        self.camera_frame = ttk.Label(cam_frame, text="Camera chưa khởi động", 
                                     anchor='center', background='black', foreground='white')
        self.camera_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Stats display
        stats_frame = ttk.LabelFrame(display_frame, text="Thống kê", padding="5")
        stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=15, 
                                                   font=('Consolas', 9), 
                                                   bg='#1e1e1e', fg='#ffffff')
        self.stats_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Update stats button
        ttk.Button(stats_frame, text="Cập nhật thống kê", 
                  command=self.update_stats_display).grid(row=1, column=0, pady=(5, 0))
    
    def toggle_camera(self):
        """Bật/tắt camera"""
        if not self.camera_active:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Khởi động camera"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera!")
                return
            
            # Thiết lập camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_active = True
            self.frame_count = 0
            self.last_save_time = 0
            
            # Bắt đầu thread camera
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.start_cam_btn.config(text="Dừng Camera")
            self.log_message("Camera đã khởi động")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động camera: {e}")
    
    def stop_camera(self):
        """Dừng camera"""
        self.camera_active = False
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        self.start_cam_btn.config(text="Bắt đầu Camera")
        self.camera_frame.config(image='', text="Camera đã dừng")
        self.log_message("Camera đã dừng")
    
    def camera_loop(self):
        """Vòng lặp xử lý camera"""
        while self.camera_active:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Xử lý MediaPipe
                results = self.collector.face_mesh.process(rgb_frame)
                
                current_ear = 0.0
                is_drowsy = False
                confidence = 0.0
                
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Tính EAR và phân loại
                        current_ear, is_drowsy, confidence = self.collector.danh_gia_buon_ngu_v2(face_landmarks)
                        
                        # Auto save
                        if self.auto_save and self.frame_count % self.save_interval == 0:
                            current_time = cv2.getTickCount()
                            if (current_time - self.last_save_time) / cv2.getTickFrequency() > 1.0:
                                if confidence > 0.3:
                                    if self.collector.luu_mau_yolo_v2(frame, face_landmarks):
                                        self.update_queue.put(('log', f"Auto saved - EAR: {current_ear:.3f}"))
                                        self.update_queue.put(('stats', None))
                                    self.last_save_time = current_time
                
                # Vẽ thông tin lên frame
                self.draw_info_on_frame(frame, current_ear, is_drowsy, confidence)
                
                # Chuyển đổi frame để hiển thị
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                # Resize để fit trong GUI
                display_size = (640, 480)
                frame_pil = frame_pil.resize(display_size, Image.Resampling.LANCZOS)
                
                frame_tk = ImageTk.PhotoImage(frame_pil)
                self.update_queue.put(('frame', frame_tk))
                
                # Lưu frame hiện tại để sử dụng cho save manual
                self.current_frame = frame
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.update_queue.put(('log', f"Lỗi camera: {e}"))
                break
    
    def draw_info_on_frame(self, frame, ear, is_drowsy, confidence):
        """Vẽ thông tin lên frame"""
        h, w = frame.shape[:2]
        
        # Background cho text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Thông tin
        texts = [
            f"EAR: {ear:.4f}",
            f"Status: {'DROWSY' if is_drowsy else 'ALERT'}",
            f"Confidence: {confidence:.3f}",
            f"Auto Save: {'ON' if self.auto_save else 'OFF'}",
            f"Samples: A:{self.collector.so_mau_alert} D:{self.collector.so_mau_drowsy}",
            f"Frame: {self.frame_count}"
        ]
        
        for i, text in enumerate(texts):
            color = (0, 0, 255) if i == 1 and is_drowsy else (255, 255, 255)
            cv2.putText(frame, text, (15, 30 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def process_queue(self):
        """Xử lý queue cập nhật GUI"""
        try:
            while True:
                item = self.update_queue.get_nowait()
                msg_type, data = item
                
                if msg_type == 'frame':
                    self.camera_frame.config(image=data, text='')
                    self.camera_frame.image = data  # Keep reference
                elif msg_type == 'log':
                    self.log_message(data)
                elif msg_type == 'stats':
                    self.update_stats_display()
                    
        except queue.Empty:
            pass
        
        self.root.after(100, self.process_queue)
    
    def save_alert_manual(self):
        """Lưu mẫu Alert thủ công"""
        if not self.camera_active or self.current_frame is None:
            messagebox.showwarning("Cảnh báo", "Camera chưa hoạt động!")
            return
        
        try:
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            results = self.collector.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    if self.collector.luu_mau_yolo_v2(self.current_frame, face_landmarks, loai_mau=0):
                        self.log_message(f"✓ Saved Alert sample - Total: {self.collector.so_mau_alert}")
                        self.update_stats_display()
                    else:
                        self.log_message("✗ Failed to save Alert sample")
            else:
                pass 
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi lưu mẫu Alert: {e}")
    
    def save_drowsy_manual(self):
        """Lưu mẫu Drowsy thủ công"""
        if not self.camera_active or self.current_frame is None:
            messagebox.showwarning("Cảnh báo", "Camera chưa hoạt động!")
            return
        
        try:
            rgb_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            results = self.collector.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    if self.collector.luu_mau_yolo_v2(self.current_frame, face_landmarks, loai_mau=1):
                        self.log_message(f"✓ Saved Drowsy sample - Total: {self.collector.so_mau_drowsy}")
                        self.update_stats_display()
                    else:
                        self.log_message("✗ Failed to save Drowsy sample")
            else:
                pass
                
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi lưu mẫu Drowsy: {e}")
    
    def toggle_auto_save(self):
        """Bật/tắt auto save"""
        self.auto_save = self.auto_save_var.get()
        self.log_message(f"Auto save: {'ON' if self.auto_save else 'OFF'}")
    
    def update_ear_threshold(self, value):
        """Cập nhật ngưỡng EAR"""
        self.collector.NGUONG_BUON_NGU = float(value)
        self.collector.luu_metadata()
    
    def update_save_interval(self, value):
        """Cập nhật khoảng thời gian lưu"""
        self.save_interval = int(float(value))
    
    def toggle_augmentation(self):
        """Bật/tắt data augmentation"""
        self.collector.enable_augmentation = self.aug_var.get()
        self.log_message(f"Data augmentation: {'ON' if self.collector.enable_augmentation else 'OFF'}")
    
    def reset_ear_history(self):
        """Reset lịch sử EAR"""
        self.collector.history_ear = []
        self.log_message("Đã reset lịch sử EAR")
    
    def split_dataset(self):
        """Chia dataset"""
        try:
            if self.collector.chia_du_lieu_train_val_test():
                messagebox.showinfo("Thành công", "Đã chia dữ liệu thành công!")
                self.update_stats_display()
            else:
                messagebox.showerror("Lỗi", "Không thể chia dữ liệu!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi chia dữ liệu: {e}")
    
    def check_quality(self):
        """Kiểm tra chất lượng dataset"""
        try:
            is_valid = self.collector.kiem_tra_chat_luong_dataset()
            if is_valid:
                messagebox.showinfo("Kết quả", "Dataset có chất lượng tốt!")
            else:
                messagebox.showwarning("Cảnh báo", "Dataset có một số vấn đề. Xem log để biết chi tiết.")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi kiểm tra: {e}")
    
    def clear_all_data(self):
        """Xóa tất cả dữ liệu"""
        result = messagebox.askyesno("Xác nhận", 
                                   "Bạn có chắc chắn muốn xóa tất cả dữ liệu?\n"
                                   "Hành động này không thể hoàn tác!")
        if result:
            try:
                self.collector.xoa_tat_ca_du_lieu()
                self.log_message("Đã xóa tất cả dữ liệu!")
                self.update_stats_display()
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi xóa dữ liệu: {e}")
    
    def update_stats_display(self):
        """Cập nhật hiển thị thống kê"""
        try:
            stats_info = self.get_detailed_stats()
            
            if self.stats_text:
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, stats_info)
        except Exception as e:
            if self.stats_text:
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, f"Lỗi khi cập nhật thống kê: {e}")
    
    def get_detailed_stats(self):
        """Lấy thống kê chi tiết"""
        stats = []
        stats.append("=" * 50)
        stats.append("THỐNG KÊ DATASET")
        stats.append("=" * 50)
        
        # Raw data stats
        raw_img_dir = os.path.join(self.collector.thu_muc_raw, "images")
        if os.path.exists(raw_img_dir):
            raw_files = [f for f in os.listdir(raw_img_dir) if f.endswith('.jpg')]
            alert_raw = len([f for f in raw_files if f.startswith('alert_')])
            drowsy_raw = len([f for f in raw_files if f.startswith('drowsy_')])
            
            stats.append(f"\nRAW DATA:")
            stats.append(f"  Alert: {alert_raw}")
            stats.append(f"  Drowsy: {drowsy_raw}")
            stats.append(f"  Total: {len(raw_files)}")
            if drowsy_raw > 0:
                stats.append(f"  Balance ratio: {alert_raw/drowsy_raw:.2f}")
        
        # Train/Val/Test stats
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.collector.thu_muc_dataset, "images", split)
            if os.path.exists(split_dir):
                files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
                alert_count = len([f for f in files if f.startswith('alert_')])
                drowsy_count = len([f for f in files if f.startswith('drowsy_')])
                
                stats.append(f"\n{split.upper()}:")
                stats.append(f"  Alert: {alert_count}")
                stats.append(f"  Drowsy: {drowsy_count}")
                stats.append(f"  Total: {len(files)}")
                if drowsy_count > 0:
                    stats.append(f"  Balance: {alert_count/drowsy_count:.2f}")
        
        # Configuration
        stats.append(f"\nCẤU HÌNH:")
        stats.append(f"  EAR Threshold: {self.collector.NGUONG_BUON_NGU:.3f}")
        stats.append(f"  Image Size: {self.collector.kich_thuoc_anh}x{self.collector.kich_thuoc_anh}")
        stats.append(f"  Auto Save Interval: {self.save_interval} frames")
        stats.append(f"  Augmentation: {'Enabled' if self.collector.enable_augmentation else 'Disabled'}")
        stats.append(f"  Classes: {', '.join(self.collector.classes)}")
        
        # Camera status
        stats.append(f"\nCAMERA STATUS:")
        stats.append(f"  Active: {'Yes' if self.camera_active else 'No'}")
        stats.append(f"  Auto Save: {'On' if self.auto_save else 'Off'}")
        stats.append(f"  Frames Processed: {self.frame_count}")
        
        stats.append("\n" + "=" * 50)
        
        return "\n".join(stats)
    
    def log_message(self, message):
        """Ghi log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {message}"
        print(log_text)  # Also print to console
    
    def on_closing(self):
        """Xử lý khi đóng ứng dụng"""
        if self.camera_active:
            self.stop_camera()
        
        self.root.destroy()

def main():
    """Hàm main"""
    try:
        root = tk.Tk()
        app = YOLODataCollectorGUI(root)
        
        # Xử lý sự kiện đóng cửa sổ
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Bắt đầu GUI
        root.mainloop()
        
    except Exception as e:
        print(f"Lỗi khởi động ứng dụng: {e}")
        messagebox.showerror("Lỗi", f"Không thể khởi động ứng dụng: {e}")

if __name__ == "__main__":
    main()
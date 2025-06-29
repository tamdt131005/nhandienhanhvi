import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
import os
from PIL import Image, ImageTk
from datetime import datetime
import pygame



PhatHienBuonNgu = None
WebcamNhanDienDayAnToan = None
PhoneDetector = None

try:
    from buonngu_runv2 import PhatHienBuonNgu
except ImportError as e:
    print(f"Warning: Could not import PhatHienBuonNgu from buonngu_run: {e}")
    PhatHienBuonNgu = None

try:
    from run_dayantoanYOLO import WebcamNhanDienDayAnToan
except ImportError as e:
    print(f"Warning: Could not import WebcamNhanDienDayAnToan from run_dayantoanYOLO: {e}")
    WebcamNhanDienDayAnToan = None

# SỬA: Import PhoneDetector trực tiếp
try:
    from run_dienthoai import PhoneDetector
except ImportError as e:
    print(f"Warning: Could not import PhoneDetector from run_dienthoai: {e}")
    PhoneDetector = None

class DriverMonitorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống giám sát hành vi tài xế")
        
        # Thiết lập kích thước cửa sổ
        self.setup_window_geometry()
        
        # Khởi tạo các model
        self.init_models()
        
        # Biến điều khiển
        self.cap = None
        self.running = False
        self.current_frame = None
        self.is_fullscreen = False
        
        # Biến theo dõi trạng thái
        self.alert_count = {'buon_ngu': 0, 'day_an_toan': 0, 'dien_thoai': 0}
        self.last_alert_time = {'buon_ngu': 0, 'day_an_toan': 0, 'dien_thoai': 0}
        self.phone_detection_counter = 0
        self.phone_detection_threshold = 5
        
        # Initialize pygame mixer once
        self.init_audio()
        
        # Tạo giao diện
        self.create_interface()
        
    def setup_window_geometry(self):
        """Setup window size and position"""
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = int(sw * 0.8)
            h = int(sh * 0.8)
            x = (sw - w) // 2
            y = int((sh - h) / 2.9)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception as e:
            print(f"Error setting window geometry: {e}")
            self.root.geometry("1200x800")
    
    def init_audio(self):
        """Initialize pygame mixer for audio alerts"""
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.audio_available = True
            print("✓ Audio system initialized")
        except Exception as e:
            print(f"✗ Audio initialization failed: {e}")
            self.audio_available = False
        
    def init_models(self):
        """Khởi tạo các model phát hiện"""
        print("Đang khởi tạo các model...")
        
        # Khởi tạo giá trị mặc định
        self.buonngu_available = False
        self.dayantoan_available = False
        self.dienthoai_available = False

        # Model buồn ngủ
        try:
            if PhatHienBuonNgu:
                self.phat_hien_buon_ngu = PhatHienBuonNgu()
                self.buonngu_available = True
                print("✓ Model buồn ngủ đã sẵn sàng")
            else:
                print("✗ PhatHienBuonNgu class not available")
        except Exception as e:
            print(f"✗ Lỗi khởi tạo model buồn ngủ: {e}")
        
        # Model dây an toàn
        try:
            if WebcamNhanDienDayAnToan:
                self.day_an_toan = WebcamNhanDienDayAnToan()
                self.dayantoan_available = True
                print("✓ Model dây an toàn đã sẵn sàng")
            else:
                print("✗ WebcamNhanDienDayAnToan class not available")
        except Exception as e:
            print(f"✗ Lỗi khởi tạo model dây an toàn: {e}")
        
        # Model điện thoại
        try:
            if PhoneDetector:
                model_path = 'runs/detect/phone_detection4/weights/best.pt'
                confidence_threshold = 0.86
                
                # Check if model file exists
                if os.path.exists(model_path):
                    self.dien_thoai = PhoneDetector(model_path=model_path, confidence=confidence_threshold)
                    self.dienthoai_available = True
                    print("✓ Model điện thoại đã sẵn sàng")
                else:
                    print(f"✗ Model file not found: {model_path}")
            else:
                print("✗ PhoneDetector class not available")
        except Exception as e:
            print(f"✗ Lỗi khởi tạo model điện thoại: {e}")
        
    def create_interface(self):
        """Tạo giao diện chính"""
        # Menu bar
        self.create_menu_bar()
        
        # Container chính
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video frame (bên trái, chiếm phần lớn)
        self.video_frame = tk.Frame(main_container, bg="black")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = tk.Label(self.video_frame, text="Camera chưa khởi động", 
                                   bg="black", fg="white", font=("Arial", 16))
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Control panel (bên phải)
        self.create_control_panel(main_container)
    
    def create_menu_bar(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menu Tùy chọn
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tùy chọn", menu=file_menu)
        file_menu.add_command(label="Toàn màn hình", command=self.toggle_fullscreen)
        file_menu.add_command(label="Chụp ảnh", command=self.capture_screenshot)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát", command=self.on_closing)
        
        # Training menu - only show if training modules exist
        training_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Training", menu=training_menu)
        training_menu.add_command(label="Training Buồn Ngủ", command=self.train_buon_ngu)
        training_menu.add_command(label="Training Dây An Toàn", command=self.train_day_an_toan)
        training_menu.add_command(label="Training Điện Thoại", command=self.train_dien_thoai)

        # Data collection menu
        collect_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Thu thập dữ liệu", menu=collect_menu)
        collect_menu.add_command(label="Quay Dữ Liệu Buồn Ngủ", command=self.quay_buon_ngu)
        collect_menu.add_command(label="Quay Dữ Liệu Dây An Toàn", command=self.quay_day_an_toan)
        collect_menu.add_command(label="Quay Dữ Liệu Điện Thoại", command=self.quay_dien_thoai)
        
    def create_control_panel(self, parent):
        """Tạo panel điều khiển"""
        control_panel = tk.Frame(parent, width=380, bg="#f0f0f0")
        control_panel.pack(side=tk.RIGHT, fill=tk.Y)
        control_panel.pack_propagate(False)
        
        # Title
        title_label = tk.Label(control_panel, text="GIÁM SÁT HÀNH VI MẤT AN TOÀN", 
                              font=("Arial", 16, "bold"), bg="#f0f0f0")
        title_label.pack(pady=10)
        
        # Camera controls
        self.create_camera_controls(control_panel)
        
        # Status panel
        self.create_status_panel(control_panel)
        
        # Results panel
        self.create_results_panel(control_panel)
        
        # Detail panel
        self.create_detail_panel(control_panel)
        
        # Alert log
        self.create_alert_log(control_panel)
        
        # Statistics
        self.create_statistics_panel(control_panel)
    
    def create_camera_controls(self, parent):
        """Create camera control buttons"""
        camera_frame = tk.LabelFrame(parent, text="Điều khiển Camera", 
                                   font=("Arial", 10, "bold"))
        camera_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_frame = tk.Frame(camera_frame)
        btn_frame.pack(pady=10)
        
        self.start_btn = tk.Button(btn_frame, text="▶ Bật Camera", 
                                  command=self.start_camera, bg="#4CAF50", fg="white",
                                  font=("Arial", 10, "bold"), width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = tk.Button(btn_frame, text="⏹ Tắt Camera", 
                                 command=self.stop_camera, bg="#f44336", fg="white",
                                 font=("Arial", 10, "bold"), width=12)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
    
    def create_status_panel(self, parent):
        """Create model status panel"""
        status_frame = tk.LabelFrame(parent, text="Trạng thái hệ thống", 
                                   font=("Arial", 10, "bold"))
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_status_frame = tk.Frame(status_frame)
        self.model_status_frame.pack(fill=tk.X, pady=5)
        
        models = [
            ("Buồn ngủ", self.buonngu_available),
            ("Dây an toàn", self.dayantoan_available),
            ("Điện thoại", self.dienthoai_available)
        ]
        
        for i, (name, available) in enumerate(models):
            status_color = "#4CAF50" if available else "#f44336"
            status_text = "●"
            
            tk.Label(self.model_status_frame, text=f"{status_text} {name}", 
                    fg=status_color, font=("Arial", 9, "bold")).pack(anchor=tk.W)
    
    def create_results_panel(self, parent):
        """Create detection results panel"""
        results_frame = tk.LabelFrame(parent, text="Kết quả phát hiện", 
                                    font=("Arial", 10, "bold"))
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_buonngu = tk.Label(results_frame, text="🟢 Buồn ngủ: Bình thường", 
                                     font=("Arial", 11), anchor=tk.W)
        self.status_buonngu.pack(fill=tk.X, pady=2)
        
        self.status_dayantoan = tk.Label(results_frame, text="🟢 Dây an toàn: Có", 
                                       font=("Arial", 11), anchor=tk.W)
        self.status_dayantoan.pack(fill=tk.X, pady=2)
        
        self.status_dienthoai = tk.Label(results_frame, text="🟢 Điện thoại: Không sử dụng", 
                                       font=("Arial", 11), anchor=tk.W)
        self.status_dienthoai.pack(fill=tk.X, pady=2)
    
    def create_detail_panel(self, parent):
        """Create detailed information panel"""
        detail_frame = tk.LabelFrame(parent, text="Thông tin chi tiết", 
                                   font=("Arial", 10, "bold"))
        detail_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.ear_label = tk.Label(detail_frame, text="EAR: ---", 
                                font=("Arial", 9), anchor=tk.W)
        self.ear_label.pack(fill=tk.X)
        
        self.confidence_label = tk.Label(detail_frame, text="Độ tin cậy: ---", 
                                       font=("Arial", 9), anchor=tk.W)
        self.confidence_label.pack(fill=tk.X)
        
        self.fps_label = tk.Label(detail_frame, text="FPS: ---", 
                                font=("Arial", 9), anchor=tk.W)
        self.fps_label.pack(fill=tk.X)
        
        self.resolution_label = tk.Label(detail_frame, text="Độ phân giải: ---", 
                                      font=("Arial", 9), anchor=tk.W)
        self.resolution_label.pack(fill=tk.X)
        
        # Removed brightness_label completely
    
    def create_alert_log(self, parent):
        """Create alert log panel"""
        log_frame = tk.LabelFrame(parent, text="Nhật ký cảnh báo", 
                                font=("Arial", 10, "bold"))
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Text widget với scrollbar
        text_frame = tk.Frame(log_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.alert_text = tk.Text(text_frame, height=6, font=("Arial", 9)) 
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.alert_text.yview)
        self.alert_text.configure(yscrollcommand=scrollbar.set)
        
        self.alert_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_statistics_panel(self, parent):
        """Create statistics panel"""
        stats_frame = tk.Frame(parent)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(stats_frame, text="Thống kê cảnh báo:", 
               font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        self.stats_label = tk.Label(stats_frame, text="Buồn ngủ: 0 | Dây AT: 0 | Điện thoại: 0", 
                                  font=("Arial", 8), fg="gray")
        self.stats_label.pack(anchor=tk.W)
        
    def start_camera(self):
        """Khởi động camera"""
        try:
            # Try different camera indices
            for camera_index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    print(f"Camera found at index {camera_index}")
                    break
                self.cap.release()
            else:
                raise Exception("Không thể tìm thấy camera")
            
            # Thiết lập độ phân giải với error handling
            try:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception as e:
                print(f"Warning: Could not set camera properties: {e}")
            
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # Bắt đầu thread xử lý video
            self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
            self.video_thread.start()
            
            self.add_alert_log("Hệ thống đã bắt đầu giám sát")
            messagebox.showinfo("Thông báo", "Camera đã khởi động thành công!")
            
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể khởi động camera: {str(e)}")
            self.reset_camera_state()
    
    def reset_camera_state(self):
        """Reset camera state after error"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
    def stop_camera(self):
        """Dừng camera"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Reset phone detection counter when stopping camera
        self.phone_detection_counter = 0
        
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        self.video_label.config(image="", text="Camera đã dừng")
        self.add_alert_log("Hệ thống đã dừng giám sát")
        messagebox.showinfo("Thông báo", "Camera đã được dừng!")
    
    def video_loop(self):
        """Vòng lặp xử lý video với error handling"""
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Cannot read frame from camera")
                    break
                
                # Lật frame
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Xử lý các phát hiện
                self.process_detections(frame)
                
                # Hiển thị frame
                self.display_frame(frame)
                
                # Tính FPS
                fps_counter += 1
                if fps_counter % 30 == 0:  # Cập nhật FPS mỗi 30 frame
                    current_time = time.time()
                    if current_time > fps_start_time:
                        fps = 30 / (current_time - fps_start_time)
                        self.root.after(0, lambda: self.fps_label.config(text=f"FPS: {fps:.1f}"))
                        fps_start_time = current_time
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in video loop: {e}")
                break
                
        # Cleanup when loop ends
        self.root.after(0, self.reset_camera_state)
    
    def process_detections(self, frame):
        """Xử lý các phát hiện trên frame với error handling"""
        detection_results = {
            'buon_ngu': False,
            'day_an_toan': True,
            'dien_thoai': False,
            'ear': 0.0,
            'confidence': 0.0,
            'buon_ngu_conf': 0.0,
            'seatbelt_conf': 0.0,
            'phone_conf': 0.0,
            'phone_frame_count': 0
        }

        processed_frame = frame.copy()

        try:
            # Nhận diện điện thoại
            if self.dienthoai_available and hasattr(self, 'dien_thoai'):
                try:
                    result = self.dien_thoai.nhan_dien_dien_thoai(processed_frame)
                    if len(result) == 4:
                        co_dien_thoai_final, phone_conf, phone_results, phone_frame_count = result
                        detection_results['dien_thoai'] = co_dien_thoai_final
                        detection_results['phone_conf'] = phone_conf
                        detection_results['phone_frame_count'] = phone_frame_count

                        if phone_results:
                            processed_frame = self.dien_thoai.draw_detections(processed_frame, phone_results)

                        cv2.putText(processed_frame, f"Phone Counter: {phone_frame_count}/{self.phone_detection_threshold}",
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                except Exception as e:
                    print(f"Error in phone detection: {e}")

            # Phát hiện buồn ngủ
            if self.buonngu_available and hasattr(self, 'phat_hien_buon_ngu'):
                try:
                    drowsiness_results = self.phat_hien_buon_ngu.xu_ly_frame(processed_frame)
                    if drowsiness_results:
                        fusion_state = drowsiness_results.get('ket_qua_fusion', {}).get('trang_thai_fusion', 'tinh_tao')
                        detection_results['buon_ngu'] = fusion_state == 'buon_ngu'
                        detection_results['ear'] = drowsiness_results.get('thong_tin_ml', {}).get('ear_trung_binh', 0.0)
                        detection_results['buon_ngu_conf'] = drowsiness_results.get('ket_qua_fusion', {}).get('do_tin_cay', 0.0)
                        processed_frame = self.draw_drowsiness_info(processed_frame, drowsiness_results)
                except Exception as e:
                    print(f"Error in drowsiness detection: {e}")

            # Phát hiện dây an toàn
            if self.dayantoan_available and hasattr(self, 'day_an_toan'):
                try:
                    co_day_an_toan, do_tin_cay = self.day_an_toan.nhan_dien_day_an_toan(processed_frame)
                    detection_results['day_an_toan'] = co_day_an_toan
                    detection_results['seatbelt_conf'] = do_tin_cay
                    self.draw_seatbelt_info(processed_frame, co_day_an_toan, do_tin_cay)
                except Exception as e:
                    print(f"Error in seatbelt detection: {e}")

            
        except Exception as e:
            print(f"Lỗi trong quá trình phát hiện: {e}")

        # Cập nhật UI
        self.root.after(0, self.update_detection_ui, detection_results)
        return processed_frame

    def draw_seatbelt_info(self, frame, co_day_an_toan, do_tin_cay):
        """Vẽ thông tin dây an toàn lên frame"""
        try:
            frame_height, frame_width = frame.shape[:2]
            text_x = frame_width - 350
            text_y = frame_height - 80
            
            mau_day = (0, 255, 0) if co_day_an_toan else (0, 0, 255)
            label_day = f"Day an toan: {'Co' if co_day_an_toan else 'Khong'} ({do_tin_cay:.2f})"
            
            cv2.rectangle(frame, (text_x - 10, text_y - 25), (text_x + 330, text_y + 5), (0, 0, 0), -1)
            cv2.rectangle(frame, (text_x - 10, text_y - 25), (text_x + 330, text_y + 5), mau_day, 2)
            cv2.putText(frame, label_day, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mau_day, 2)
            
            cv2.putText(frame, "Nhan 'q' de thoat, 's' de chup anh",
                       (frame_width-350, frame_height-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            print(f"Error drawing seatbelt info: {e}")
    
    def draw_drowsiness_info(self, frame, drowsiness_results):
        """Vẽ thông tin buồn ngủ lên frame"""
        try:
            h, w, _ = frame.shape
            face_landmarks_list = drowsiness_results.get('face_landmarks', [])
            ket_qua_fusion = drowsiness_results.get('ket_qua_fusion', {})
            thong_tin_ml = drowsiness_results.get('thong_tin_ml', {})

            # Draw face landmarks if any
            if face_landmarks_list and hasattr(self.phat_hien_buon_ngu, 'CHI_SO_MAT_TRAI'):
                for face_landmarks in face_landmarks_list:
                    for idx in self.phat_hien_buon_ngu.CHI_SO_MAT_TRAI + self.phat_hien_buon_ngu.CHI_SO_MAT_PHAI:
                        if idx < len(face_landmarks.landmark):
                            x = int(face_landmarks.landmark[idx].x * w)
                            y = int(face_landmarks.landmark[idx].y * h)
                            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "Không phát hiện khuôn mặt", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Draw drowsiness status text
            if ket_qua_fusion:
                status_text = f"TRẠNG THÁI: {ket_qua_fusion.get('trang_thai_fusion', '').upper()}"
                if ket_qua_fusion.get('trang_thai_fusion', '') == 'buon_ngu':
                    status_text = f"CẢNH BÁO: BUỒN NGỦ!"
                cv2.putText(frame, status_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, ket_qua_fusion.get('mau_fusion', (0,255,0)), 3)
                
                if ket_qua_fusion.get('trang_thai_fusion', '') in ['buon_ngu', 'canh_bao']:
                    cv2.putText(frame, f"Độ tin cậy: {ket_qua_fusion.get('do_tin_cay', 0.0):.2f}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, ket_qua_fusion.get('mau_fusion', (0,255,0)), 2)

            if thong_tin_ml and thong_tin_ml.get('ear_trung_binh', 0) > 0:
                cv2.putText(frame, f"EAR: {thong_tin_ml['ear_trung_binh']:.3f}", (10, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
        except Exception as e:
            print(f"Error drawing drowsiness info: {e}")
            
        return frame

    def update_detection_ui(self, results):
        """Cập nhật UI - Loại bỏ logic cảnh báo kép"""
        now = time.time()

        # Khởi tạo last_alert_time nếu chưa có
        if not hasattr(self, 'last_alert_time'):
            self.last_alert_time = {'buon_ngu': 0, 'day_an_toan': 0, 'dien_thoai': 0}

        # Reset tất cả về trạng thái bình thường trước
        self.status_buonngu.config(text="🟢 Buồn ngủ: Bình thường", fg="green")
        self.status_dayantoan.config(text="🟢 Dây an toàn: Có", fg="green")
        self.status_dienthoai.config(text="🟢 Điện thoại: Không sử dụng", fg="green")

        # Lấy các giá trị từ results
        is_buon_ngu = results['buon_ngu']
        is_day_an_toan = results['day_an_toan']
        is_dien_thoai = results['dien_thoai']
        phone_frame_count = results.get('phone_frame_count', 0)

        # Cập nhật buồn ngủ
        if is_buon_ngu:
            self.status_buonngu.config(text="🔴 Buồn ngủ: CẢNH BÁO!", fg="red")
            if now - self.last_alert_time['buon_ngu'] > 5:
                self.alert_count['buon_ngu'] += 1
                self.add_alert_log("⚠️ CẢNH BÁO: Tài xế buồn ngủ!")
                self.play_alert_sound("../sound/buonngu.wav")
                self.last_alert_time['buon_ngu'] = now

        # Cập nhật dây an toàn
        if not is_day_an_toan:
            self.status_dayantoan.config(text="🔴 Dây an toàn: Không có", fg="red")
            if now - self.last_alert_time['day_an_toan'] > 5:
                self.alert_count['day_an_toan'] += 1
                self.add_alert_log("⚠️ CẢNH BÁO: Không thắt dây an toàn!")
                self.play_alert_sound("../sound/dayantoan.wav")
                self.last_alert_time['day_an_toan'] = now

        # Xử lý cảnh báo điện thoại dựa trên frame count
        if phone_frame_count > 0:
            if is_dien_thoai:
                # Đã đủ frame để báo cảnh báo
                self.status_dienthoai.config(
                    text=f"🔴 Điện thoại: CẢNH BÁO! ({phone_frame_count}/{self.phone_detection_threshold})",
                    fg="red"
                )

                # Phát cảnh báo mỗi 5 giây
                if now - self.last_alert_time['dien_thoai'] > 5:
                    self.alert_count['dien_thoai'] += 1
                    self.add_alert_log(f"⚠️ CẢNH BÁO: Sử dụng điện thoại! (Xác nhận {phone_frame_count} frame)")
                    self.play_alert_sound("../sound/dienthoaiv3.wav")
                    self.last_alert_time['dien_thoai'] = now
            else:
                # Đang trong quá trình kiểm tra
                self.status_dienthoai.config(
                    text=f"🟡 Điện thoại: Đang xác minh... ({phone_frame_count}/{self.phone_detection_threshold})",
                    fg="orange"
                )

        # Cập nhật thông tin chi tiết
        self.ear_label.config(text=f"EAR: {results['ear']:.3f}")
        self.confidence_label.config(text=f"Độ tin cậy: {results.get('buon_ngu_conf', 0):.2f}")
        
        # Cập nhật thống kê
        stats_text = f"Buồn ngủ: {self.alert_count['buon_ngu']} | Dây AT: {self.alert_count['day_an_toan']} | Điện thoại: {self.alert_count['dien_thoai']}"
        self.stats_label.config(text=stats_text)
        
        # Cập nhật độ phân giải nếu có frame hiện tại
        if self.current_frame is not None:
            h, w = self.current_frame.shape[:2]
            self.resolution_label.config(text=f"Độ phân giải: {w}x{h}")

    def play_alert_sound(self, sound_file_path):
        """Phát âm thanh cảnh báo từ file"""
        if not self.audio_available:
            print(f"Audio not available, cannot play {sound_file_path}")
            return
            
        try:
            # Load and play the sound file
            sound = pygame.mixer.Sound(sound_file_path)
            sound.play()
            print(f"Played sound: {sound_file_path}")
            
        except pygame.error as e:
            print(f"Error playing sound {sound_file_path}: {e}")
        except Exception as e:
            print(f"Unexpected error playing sound {sound_file_path}: {e}")
    
    def display_frame(self, frame):
        """Hiển thị frame lên GUI"""
        try:
            # Resize frame để fit với video label
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()
            
            if label_width > 1 and label_height > 1:
                # Tính toán tỷ lệ để maintain aspect ratio
                frame_height, frame_width = frame.shape[:2]
                aspect_ratio = frame_width / frame_height
                
                if label_width / label_height > aspect_ratio:
                    # Fit by height
                    new_height = label_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Fit by width
                    new_width = label_width
                    new_height = int(new_width / aspect_ratio)
                
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = cv2.resize(frame, (640, 480))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            img_pil = Image.fromarray(frame_rgb)
            
            # Convert to PhotoImage
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Update label
            self.video_label.config(image=img_tk, text="")
            self.video_label.image = img_tk  # Keep a reference
            
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def add_alert_log(self, message):
        """Thêm thông báo vào log"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            
            self.alert_text.insert(tk.END, log_message)
            self.alert_text.see(tk.END)  # Scroll to bottom
            
            # Giới hạn số dòng log (giữ 100 dòng cuối)
            lines = self.alert_text.get("1.0", tk.END).split('\n')
            if len(lines) > 100:
                self.alert_text.delete("1.0", f"{len(lines) - 100}.0")
                
        except Exception as e:
            print(f"Error adding to log: {e}")
    
    def capture_screenshot(self):
        """Chụp ảnh màn hình"""
        try:
            if self.current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                
                # Tạo thư mục screenshots nếu chưa có
                os.makedirs("screenshots", exist_ok=True)
                filepath = os.path.join("screenshots", filename)
                
                cv2.imwrite(filepath, self.current_frame)
                self.add_alert_log(f"Đã chụp ảnh: {filename}")
                messagebox.showinfo("Thông báo", f"Đã lưu ảnh: {filepath}")
            else:
                messagebox.showwarning("Cảnh báo", "Không có hình ảnh để chụp!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể chụp ảnh: {str(e)}")
    
    def toggle_fullscreen(self):
        """Chuyển đổi chế độ toàn màn hình"""
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)
        
        if self.is_fullscreen:
            self.add_alert_log("Chuyển sang chế độ toàn màn hình")
        else:
            self.add_alert_log("Thoát chế độ toàn màn hình")
    
    def on_closing(self):
        """Xử lý khi đóng ứng dụng"""
        try:
            if messagebox.askokcancel("Thoát", "Bạn có chắc chắn muốn thoát?"):
                self.running = False
                if self.cap:
                    self.cap.release()
                
                # Quit pygame mixer
                if self.audio_available:
                    pygame.mixer.quit()
                
                self.root.destroy()
        except Exception as e:
            print(f"Error during closing: {e}")
            self.root.destroy()
    
    def train_buon_ngu(self):
        """Mở giao diện training buồn ngủ từ class khác"""
        try:
            # Import giao diện training từ file khác
            from giao_dien_train_bn import GUITrainBuonNgu
            
            # Tạo cửa sổ training mới
            new_window = tk.Toplevel(self.root)
            app = GUITrainBuonNgu(new_window)
            
        except ImportError as e:
            messagebox.showerror("Lỗi", f"Không thể import giao diện training: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở giao diện training: {str(e)}")
    
    def train_day_an_toan(self):
        """Mở giao diện training buồn ngủ từ class khác"""
        try:
            # Import giao diện training từ file khác
            from giaodien_antoan import GiaoDienHuanLuyen
            
            # Tạo cửa sổ training mới
            new_window = tk.Toplevel(self.root)
            app = GiaoDienHuanLuyen(new_window)
            
        except ImportError as e:
            messagebox.showerror("Lỗi", f"Không thể import giao diện training: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở giao diện training: {str(e)}")
    
    def train_dien_thoai(self):
        """Mở giao diện training buồn ngủ từ class khác"""
        try:
            # Import giao diện training từ file khác
            from giao_dien_train_dt import YOLOTrainingGUI
            
            # Tạo cửa sổ training mới
            new_window = tk.Toplevel(self.root)
            app = YOLOTrainingGUI(new_window)
            
        except ImportError as e:
            messagebox.showerror("Lỗi", f"Không thể import giao diện training: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở giao diện training: {str(e)}")
    def quay_buon_ngu(self):
        """Mở giao diện colect buồn ngủ từ class khác"""
        try:
            # Import giao diện colect từ file khác
            from GUI_buonngu import YOLODataCollectorGUI # Corrected class name
            
            # Tạo cửa sổ colect mới
            new_window = tk.Toplevel(self.root)
            app = YOLODataCollectorGUI(new_window) # Corrected class instantiation
            
        except ImportError as e:
            messagebox.showerror("Lỗi", f"Không thể import giao diện training: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở giao diện training: {str(e)}")     
    def quay_day_an_toan(self):
        """Mở giao diện collect buồn ngủ từ class khác"""
        try:
            # Import giao diện colect từ file khác
            from test_v2sbelt import SeatbeltDataCollectionGUI
            
            # Tạo cửa sổ colect mới
            new_window = tk.Toplevel(self.root)
            app = SeatbeltDataCollectionGUI(new_window)
            
        except ImportError as e:
            messagebox.showerror("Lỗi", f"Không thể import giao diện training: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở giao diện training: {str(e)}")     
    def quay_dien_thoai(self):
        """Mở giao diện collect buồn ngủ từ class khác"""
        try:
            # Import giao diện collect từ file khác
            from test_v2sdt import ModernDataCollectionGUI
            
            # Tạo cửa sổ colect mới
            new_window = tk.Toplevel(self.root)
            app = ModernDataCollectionGUI(new_window)
            
        except ImportError as e:
            messagebox.showerror("Lỗi", f"Không thể import giao diện training: {str(e)}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở giao diện training: {str(e)}")            
def main():
    """Hàm main"""
    try:
        root = tk.Tk()
        
        # Thiết lập icon và title
        root.title("Driver Monitor System v2.0")
        
        # Tạo ứng dụng
        app = DriverMonitorApp(root)
        
        # Xử lý sự kiện đóng cửa sổ
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Bind phím tắt
        root.bind('<F11>', lambda e: app.toggle_fullscreen())
        root.bind('<Escape>', lambda e: setattr(app, 'is_fullscreen', False) or root.attributes('-fullscreen', False))
        
        # Chạy ứng dụng
        root.mainloop()
        
    except Exception as e:
        print(f"Lỗi khởi động ứng dụng: {e}")
        messagebox.showerror("Lỗi", f"Không thể khởi động ứng dụng: {str(e)}")

if __name__ == "__main__":
    main()
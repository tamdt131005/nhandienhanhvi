import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import datetime
import json
import numpy as np
from pathlib import Path
import time
from PIL import Image, ImageTk
import threading
import glob

class SeatbeltDataCollectionGUI:
    def __init__(self, root):
        self.root = root if root else tk.Tk()  # hoặc không gán gì nếu bạn chưa cần dùng root
        self.root = root
        self.root.title("Công cụ Thu thập Dữ liệu Dây An toàn")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f0f0')
        
        # Biến camera
        self.cap = None
        self.camera_running = False
        self.current_frame = None
        
        # Cài đặt camera
        self.camera_index = tk.IntVar(value=0)
        self.frame_width = tk.IntVar(value=640)
        self.frame_height = tk.IntVar(value=480)
        
        # Thư mục lưu trữ
        self.save_dir = "seatbelt_dataset"
        
        # Định nghĩa các nhãn
        self.labels = {
            'with_seatbelt': 'Có đeo dây an toàn',
            'without_seatbelt': 'Không đeo dây an toàn',
        }
        
        # Thống kê
        self.stats = {label: tk.IntVar(value=0) for label in self.labels.keys()}
        self.total_images = tk.IntVar(value=0)
        
        # Nhãn được chọn
        self.selected_label = tk.StringVar(value='with_seatbelt')
        
        # Tạo giao diện
        self.create_widgets()
        self.setup_directories()
        self.update_stats_from_disk()  # Thêm dòng này
        
        # Bind keyboard shortcuts
        self.root.bind('<Key-1>', lambda e: self.set_label('with_seatbelt'))
        self.root.bind('<Key-2>', lambda e: self.set_label('without_seatbelt'))
        self.root.bind('<space>', lambda e: self.capture_image())
        self.root.bind('<Key-r>', lambda e: self.reset_stats())
        self.root.focus_set()
        
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left frame - Camera
        left_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Camera label
        self.camera_label = tk.Label(left_frame, text="Camera chưa được khởi động", 
                                   bg='black', fg='white', font=('Arial', 14))
        self.camera_label.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        # Right frame - Controls
        right_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, bd=2, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        right_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(right_frame, text="CÔNG CỤ THU THẬP DỮ LIỆU", 
                              font=('Arial', 14, 'bold'), bg='white', fg='#2c3e50')
        title_label.pack(pady=10)
        
        # Camera Settings
        settings_frame = tk.LabelFrame(right_frame, text="Cài đặt Camera", 
                                     font=('Arial', 10, 'bold'), bg='white')
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Camera index
        tk.Label(settings_frame, text="Camera Index:", bg='white').pack(anchor='w', padx=5)
        camera_index_frame = tk.Frame(settings_frame, bg='white')
        camera_index_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Spinbox(camera_index_frame, from_=0, to=10, textvariable=self.camera_index, 
                  width=10).pack(side=tk.LEFT)
        
        # Resolution
        tk.Label(settings_frame, text="Độ phân giải:", bg='white').pack(anchor='w', padx=5, pady=(10,0))
        resolution_frame = tk.Frame(settings_frame, bg='white')
        resolution_frame.pack(fill=tk.X, padx=5, pady=2)
        
        tk.Label(resolution_frame, text="W:", bg='white').pack(side=tk.LEFT)
        tk.Spinbox(resolution_frame, from_=320, to=1920, textvariable=self.frame_width, 
                  width=8, increment=160).pack(side=tk.LEFT, padx=2)
        tk.Label(resolution_frame, text="H:", bg='white').pack(side=tk.LEFT, padx=(10,0))
        tk.Spinbox(resolution_frame, from_=240, to=1080, textvariable=self.frame_height, 
                  width=8, increment=120).pack(side=tk.LEFT, padx=2)
        
        # Camera controls
        camera_control_frame = tk.Frame(settings_frame, bg='white')
        camera_control_frame.pack(pady=10)
        
        self.start_btn = tk.Button(camera_control_frame, text="Bắt đầu Camera", 
                                  command=self.start_camera, bg='#27ae60', fg='white',
                                  font=('Arial', 10, 'bold'), width=12)
        self.start_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(camera_control_frame, text="Dừng Camera", 
                                 command=self.stop_camera, bg='#e74c3c', fg='white',
                                 font=('Arial', 10, 'bold'), width=12, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        # Label Selection
        label_frame = tk.LabelFrame(right_frame, text="Chọn Nhãn", 
                                   font=('Arial', 10, 'bold'), bg='white')
        label_frame.pack(fill=tk.X, padx=10, pady=10)
        
        for key, value in self.labels.items():
            rb = tk.Radiobutton(label_frame, text=f"{value}", 
                               variable=self.selected_label, value=key,
                               bg='white', font=('Arial', 10))
            rb.pack(anchor='w', padx=10, pady=2)
        
        # Capture Button
        self.capture_btn = tk.Button(right_frame, text="CHỤP ẢNH (Space)", 
                                    command=self.capture_image, bg='#3498db', fg='white',
                                    font=('Arial', 12, 'bold'), height=2, state=tk.DISABLED)
        self.capture_btn.pack(fill=tk.X, padx=10, pady=10)
        
        # Statistics
        stats_frame = tk.LabelFrame(right_frame, text="Thống kê", 
                                   font=('Arial', 10, 'bold'), bg='white')
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Total images
        total_frame = tk.Frame(stats_frame, bg='white')
        total_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(total_frame, text="Tổng ảnh:", bg='white', font=('Arial', 10, 'bold')).pack(side=tk.LEFT)
        tk.Label(total_frame, textvariable=self.total_images, bg='white', 
                font=('Arial', 10, 'bold'), fg='#e74c3c').pack(side=tk.RIGHT)
        
        # Individual stats
        for key, value in self.labels.items():
            stat_frame = tk.Frame(stats_frame, bg='white')
            stat_frame.pack(fill=tk.X, padx=5, pady=1)
            tk.Label(stat_frame, text=f"{value}:", bg='white', font=('Arial', 9)).pack(side=tk.LEFT)
            tk.Label(stat_frame, textvariable=self.stats[key], bg='white', 
                    font=('Arial', 9, 'bold'), fg='#27ae60').pack(side=tk.RIGHT)
        
        # Action buttons
        action_frame = tk.Frame(right_frame, bg='white')
        action_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(action_frame, text="Reset Thống kê (R)", command=self.reset_stats,
                 bg='#f39c12', fg='white', font=('Arial', 10)).pack(fill=tk.X, pady=2)
        
        tk.Button(action_frame, text="Chọn Thư mục Lưu", command=self.choose_save_dir,
                 bg='#9b59b6', fg='white', font=('Arial', 10)).pack(fill=tk.X, pady=2)
        
        tk.Button(action_frame, text="Xuất Thông tin Dataset", command=self.export_dataset_info,
                 bg='#34495e', fg='white', font=('Arial', 10)).pack(fill=tk.X, pady=2)
        
        # Save directory display
        self.save_dir_label = tk.Label(right_frame, text=f"Lưu tại: ...{self.save_dir[-30:]}", 
                                      bg='white', font=('Arial', 9), fg='#7f8c8d', wraplength=300)
        self.save_dir_label.pack(padx=10, pady=5)
        
        # Keyboard shortcuts
        shortcuts_frame = tk.LabelFrame(right_frame, text="Phím tắt", 
                                       font=('Arial', 9, 'bold'), bg='white')
        shortcuts_frame.pack(fill=tk.X, padx=10, pady=5)
        
        shortcuts_text = [
            "1 - Có đeo dây an toàn",
            "2 - Không đeo dây an toàn", 
            "Space - Chụp ảnh",
            "R - Reset thống kê"
        ]
        
        for shortcut in shortcuts_text:
            tk.Label(shortcuts_frame, text=shortcut, bg='white', 
                    font=('Arial', 8), fg='#7f8c8d').pack(anchor='w', padx=5, pady=1)
    
    def setup_directories(self):
        """Tạo các thư mục cần thiết"""
        try:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            for label in self.labels.keys():
                Path(os.path.join(self.save_dir, label)).mkdir(parents=True, exist_ok=True)
            self.update_stats_from_disk()  # Thêm dòng này
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tạo thư mục: {e}")

    def start_camera(self):
        """Bắt đầu camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index.get())
            if not self.cap.isOpened():
                raise Exception(f"Không thể mở camera {self.camera_index.get()}")
            
            # Cài đặt độ phân giải
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width.get())
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height.get())
            
            self.camera_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.capture_btn.config(state=tk.NORMAL)
            
            # Bắt đầu thread cập nhật camera
            self.camera_thread = threading.Thread(target=self.update_camera)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Lỗi Camera", f"Không thể khởi động camera: {e}")
    
    def stop_camera(self):
        """Dừng camera"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.DISABLED)
        
        # Reset camera label
        self.camera_label.config(image='', text="Camera đã dừng")
    
    def update_camera(self):
        """Cập nhật hình ảnh camera"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                # Lật ảnh theo chiều ngang
                frame = cv2.flip(frame, 1)
                self.current_frame = frame.copy()
                
                # Chuyển đổi sang RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize để fit với label
                height, width, _ = frame_rgb.shape
                max_width = 640
                max_height = 480
                
                if width > max_width or height > max_height:
                    scale = min(max_width/width, max_height/height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
                
                # Chuyển sang PIL và Tkinter
                image_pil = Image.fromarray(frame_rgb)
                image_tk = ImageTk.PhotoImage(image_pil)
                
                # Cập nhật label
                self.camera_label.config(image=image_tk, text='')
                self.camera_label.image = image_tk
            
            time.sleep(0.03)  # ~30 FPS
    
    def set_label(self, label_key):
        """Đặt nhãn được chọn và cập nhật giao diện"""
        self.selected_label.set(label_key)
        # Duyệt qua các radiobutton để cập nhật trạng thái
        for child in self.root.winfo_children():
            for subchild in child.winfo_children():
                if isinstance(subchild, tk.LabelFrame) and subchild.cget('text') == 'Chọn Nhãn':
                    for rb in subchild.winfo_children():
                        if isinstance(rb, tk.Radiobutton):
                            if rb['value'] == label_key:
                                rb.select()
                            else:
                                rb.deselect()
    
    def capture_image(self):
        """Chụp và lưu ảnh"""
        if not self.camera_running or self.current_frame is None:
            messagebox.showwarning("Cảnh báo", "Camera chưa được khởi động!")
            return
        
        label_key = self.selected_label.get()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Tên file
        filename = f"{label_key}_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, label_key, filename)
        
        # Lưu ảnh
        success = cv2.imwrite(filepath, self.current_frame)
        
        if success:
            self.update_stats_from_disk()  # Thay vì tự tăng biến đếm, cập nhật lại từ ổ đĩa
            
            self.root.after(10, lambda: messagebox.showinfo("Thành công", f"Đã lưu ảnh: {filename}\nNhãn: {self.labels[label_key]}"))
        else:
            self.root.after(10, lambda: messagebox.showerror("Lỗi", f"Không thể lưu ảnh: {filepath}"))
    
    def update_stats_from_disk(self):
        """Cập nhật thống kê từ số lượng file thực tế trong thư mục"""
        total = 0
        for key in self.labels.keys():
            folder = os.path.join(self.save_dir, key)
            count = len(glob.glob(os.path.join(folder, "*.jpg")))
            self.stats[key].set(count)
            total += count
        self.total_images.set(total) # This correctly updates the total count
    
    def reset_stats(self):
        """Reset thống kê"""
        # Xóa tất cả ảnh trong các thư mục nhãn
        for key in self.labels.keys():
            folder = os.path.join(self.save_dir, key)
            for file in glob.glob(os.path.join(folder, "*.jpg")):
                try:
                    os.remove(file)
                except Exception:
                    pass
        self.update_stats_from_disk()
        messagebox.showinfo("Thông báo", "Đã reset thống kê!")
    
    def choose_save_dir(self):
        """Chọn thư mục lưu trữ"""
        directory = filedialog.askdirectory(initialdir=self.save_dir)
        if directory:
            self.save_dir = directory
            self.setup_directories()
            self.save_dir_label.config(text=f"Lưu tại: ...{self.save_dir[-30:]}")
            self.update_stats_from_disk()  # Thêm dòng này
    
    def export_dataset_info(self):
        """Xuất thông tin dataset ra file JSON"""
        dataset_info = {
            'total_images': self.total_images.get(),
            'labels': self.labels,
            'statistics': {key: var.get() for key, var in self.stats.items()},
            'created_at': datetime.datetime.now().isoformat(),
            'save_directory': os.path.abspath(self.save_dir)
        }
        
        info_file = os.path.join(self.save_dir, "dataset_info.json")
        try:
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, ensure_ascii=False, indent=2)
            
            messagebox.showinfo("Thành công", f"Thông tin dataset đã được lưu tại:\n{info_file}")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể xuất thông tin: {e}")
    
    def on_closing(self):
        """Xử lý khi đóng ứng dụng"""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SeatbeltDataCollectionGUI(root)
    
    # Xử lý khi đóng cửa sổ
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Chạy ứng dụng
    root.mainloop()
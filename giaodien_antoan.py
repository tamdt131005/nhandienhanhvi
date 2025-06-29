import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
import subprocess
import json

# Import class huấn luyện từ file chính
try:
    from train_dayantoan import HuanLuyenNhanDienDayAnToan
except ImportError:
    print("Không thể import HuanLuyenNhanDienDayAnToan. Vui lòng đảm bảo file train_dayantoan.py có trong cùng thư mục.")
    sys.exit(1)

class GiaoDienHuanLuyen:
    def __init__(self, root):
        self.root = root
        self.root.title("Huấn Luyện YOLOv8n - Nhận Diện Dây An Toàn")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        
        # Khởi tạo bộ huấn luyện
        self.bo_huan_luyen = None
        self.duong_dan_dataset = tk.StringVar(value="seatbelt_dataset")
        self.so_epoch = tk.IntVar(value=50)
        self.kich_thuoc_anh = tk.IntVar(value=640)
        self.batch_size = tk.IntVar(value=8)
        self.duong_dan_model = tk.StringVar(value="modelantoan/nhan_dien_day_an_toan_tot_nhat.pt")
        
        # Tạo giao diện
        self.tao_giao_dien()
        
        # Redirect stdout để hiển thị trong GUI
        self.redirect_stdout()
        
    def tao_giao_dien(self):
        """Tạo các thành phần giao diện"""
        
        # Title
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X, padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🚗 HUẤN LUYỆN YOLOV8N - NHẬN DIỆN DÂY AN TOÀN", 
                              font=("Arial", 16, "bold"), fg="white", bg="#2c3e50")
        title_label.pack(expand=True)
        
        # Main container
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        left_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=1)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        # Right panel - Log output
        right_frame = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Tạo các section trong left panel
        self.tao_section_cau_hinh(left_frame)
        self.tao_section_dataset(left_frame)
        self.tao_section_huan_luyen(left_frame)
        self.tao_section_danh_gia(left_frame)
        self.tao_section_du_doan(left_frame)
        
        # Tạo log output
        self.tao_log_output(right_frame)
        
    def tao_section_cau_hinh(self, parent):
        """Section cấu hình dataset"""
        frame = tk.LabelFrame(parent, text="📁 Cấu hình Dataset", font=("Arial", 10, "bold"), 
                             bg="white", fg="#2c3e50", padx=10, pady=5)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Dataset path
        tk.Label(frame, text="Đường dẫn Dataset:", bg="white").pack(anchor=tk.W)
        path_frame = tk.Frame(frame, bg="white")
        path_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Entry(path_frame, textvariable=self.duong_dan_dataset, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(path_frame, text="Chọn", command=self.chon_dataset, bg="#3498db", fg="white",
                 font=("Arial", 8)).pack(side=tk.RIGHT, padx=(5, 0))
                 
    def tao_section_dataset(self, parent):
        """Section xử lý dataset"""
        frame = tk.LabelFrame(parent, text="🔧 Xử lý Dataset", font=("Arial", 10, "bold"),
                             bg="white", fg="#2c3e50", padx=10, pady=5)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(frame, text="🏗️  Tạo cấu trúc YOLO", command=self.tao_cau_truc_yolo,
                 bg="#e74c3c", fg="white", font=("Arial", 9, "bold"), width=25).pack(pady=2)
        
        tk.Button(frame, text="📊 Xử lý ảnh và nhãn", command=self.xu_ly_anh_nhan,
                 bg="#f39c12", fg="white", font=("Arial", 9, "bold"), width=25).pack(pady=2)
        tk.Button(frame, text="📋 Tạo file cấu hình YAML", command=self.tao_yaml,
                 bg="#9b59b6", fg="white", font=("Arial", 9, "bold"), width=25).pack(pady=2)
                 
    def tao_section_huan_luyen(self, parent):
        """Section huấn luyện model"""  
        frame = tk.LabelFrame(parent, text="🚀 Huấn luyện Model", font=("Arial", 10, "bold"),
                             bg="white", fg="#2c3e50", padx=10, pady=5)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Training parameters
        params_frame = tk.Frame(frame, bg="white")
        params_frame.pack(fill=tk.X, pady=5)
        
        # Epoch
        tk.Label(params_frame, text="Số Epoch:", bg="white").grid(row=0, column=0, sticky="w", padx=5)
        tk.Entry(params_frame, textvariable=self.so_epoch, width=10).grid(row=0, column=1, padx=5)
        
        # Image size
        tk.Label(params_frame, text="Kích thước ảnh:", bg="white").grid(row=1, column=0, sticky="w", padx=5)
        tk.Entry(params_frame, textvariable=self.kich_thuoc_anh, width=10).grid(row=1, column=1, padx=5)
        
        # Batch size
        tk.Label(params_frame, text="Batch size:", bg="white").grid(row=2, column=0, sticky="w", padx=5)
        tk.Entry(params_frame, textvariable=self.batch_size, width=10).grid(row=2, column=1, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(frame, 
                                          variable=self.progress_var,
                                          maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Training button
        tk.Button(frame, text="🚀 Bắt đầu huấn luyện", command=self.bat_dau_huan_luyen,
                 bg="#27ae60", fg="white", font=("Arial", 10, "bold"), width=25).pack(pady=10)
                 
    def tao_section_danh_gia(self, parent):
        """Section đánh giá model"""
        frame = tk.LabelFrame(parent, text="📈 Đánh giá Model", font=("Arial", 10, "bold"),
                             bg="white", fg="#2c3e50", padx=10, pady=5)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(frame, text="📊 Đánh giá model", command=self.danh_gia_model,
                 bg="#34495e", fg="white", font=("Arial", 9, "bold"), width=25).pack(pady=2)
        
        tk.Button(frame, text="📈 Xem biểu đồ", command=self.xem_bieu_do,
                 bg="#16a085", fg="white", font=("Arial", 9, "bold"), width=25).pack(pady=2)
                 
    def tao_section_du_doan(self, parent):
        """Section dự đoán"""
        frame = tk.LabelFrame(parent, text="🔍 Dự đoán", font=("Arial", 10, "bold"),
                             bg="white", fg="#2c3e50", padx=10, pady=5)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model path
        tk.Label(frame, text="Đường dẫn Model:", bg="white").pack(anchor=tk.W)
        model_frame = tk.Frame(frame, bg="white")
        model_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Entry(model_frame, textvariable=self.duong_dan_model, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(model_frame, text="Chọn", command=self.chon_model, bg="#3498db", fg="white",
                 font=("Arial", 8)).pack(side=tk.RIGHT, padx=(5, 0))
        
        tk.Button(frame, text="🖼️  Dự đoán ảnh", command=self.du_doan_anh,
                 bg="#e67e22", fg="white", font=("Arial", 9, "bold"), width=25).pack(pady=2)
        
        tk.Button(frame, text="📹 Dự đoán video", command=self.du_doan_video,
                 bg="#8e44ad", fg="white", font=("Arial", 9, "bold"), width=25).pack(pady=2)
                 
    def tao_log_output(self, parent):
        """Tạo khu vực hiển thị log"""
        log_frame = tk.LabelFrame(parent, text="📄 Log Output", font=("Arial", 10, "bold"),
                                 bg="white", fg="#2c3e50", padx=10, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Text widget with scrollbar
        self.log_text = scrolledtext.ScrolledText(log_frame, width=60, height=30, 
                                                 font=("Courier", 9), bg="#2c3e50", fg="#ecf0f1")
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Buttons frame
        btn_frame = tk.Frame(log_frame, bg="white")
        btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Button(btn_frame, text="🗑️  Xóa log", command=self.xoa_log,
                 bg="#e74c3c", fg="white", font=("Arial", 8)).pack(side=tk.LEFT)
        
        tk.Button(btn_frame, text="💾 Lưu log", command=self.luu_log,
                 bg="#27ae60", fg="white", font=("Arial", 8)).pack(side=tk.LEFT, padx=(5, 0))
                 
    def redirect_stdout(self):
        """Chuyển hướng stdout để hiển thị trong GUI"""
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.update()
                
            def flush(self):
                pass
                
        sys.stdout = StdoutRedirector(self.log_text)
        
    def chon_dataset(self):
        """Chọn thư mục dataset"""
        folder = filedialog.askdirectory(title="Chọn thư mục dataset")
        if folder:
            self.duong_dan_dataset.set(folder)
            
    def chon_model(self):
        """Chọn file model"""
        file = filedialog.askopenfilename(title="Chọn file model", 
                                         filetypes=[("Model files", "*.pt"), ("All files", "*.*")])
        if file:
            self.duong_dan_model.set(file)
            
    def khoi_tao_bo_huan_luyen(self):
        """Khởi tạo bộ huấn luyện với dataset path mới"""
        try:
            self.bo_huan_luyen = HuanLuyenNhanDienDayAnToan(self.duong_dan_dataset.get())
            print(f"✓ Đã khởi tạo bộ huấn luyện với dataset: {self.duong_dan_dataset.get()}")
            return True
        except Exception as e:
            print(f"❌ Lỗi khởi tạo bộ huấn luyện: {e}")
            return False
            
    def tao_cau_truc_yolo(self):
        """Tạo cấu trúc thư mục YOLO"""
        def chay():
            if not self.khoi_tao_bo_huan_luyen():
                return
            try:
                self.bo_huan_luyen.tao_cau_truc_yolo()
                print("✅ Hoàn thành tạo cấu trúc YOLO!")
            except Exception as e:
                print(f"❌ Lỗi tạo cấu trúc YOLO: {e}")
                
        threading.Thread(target=chay, daemon=True).start()
        
    def xu_ly_anh_nhan(self):
        """Xử lý ảnh và nhãn"""
        def chay():
            if not self.khoi_tao_bo_huan_luyen():
                return
            try:
                tat_ca_du_lieu = self.bo_huan_luyen.xu_ly_anh_va_nhan()
                if len(tat_ca_du_lieu) > 0:
                    self.bo_huan_luyen.chia_va_sao_chep_du_lieu(tat_ca_du_lieu)
                    print("✅ Hoàn thành xử lý ảnh và nhãn!")
                else:
                    print("⚠️  Không tìm thấy dữ liệu để xử lý!")
            except Exception as e:
                print(f"❌ Lỗi xử lý ảnh và nhãn: {e}")
                
        threading.Thread(target=chay, daemon=True).start()
    def tao_yaml(self):
        """Tạo file cấu hình YAML"""
        def chay():
            if not self.khoi_tao_bo_huan_luyen():
                return
            try:
                self.bo_huan_luyen.tao_file_cau_hinh_yaml()
                print("✅ Hoàn thành tạo file YAML!")
            except Exception as e:
                print(f"❌ Lỗi tạo file YAML: {e}")
                
        threading.Thread(target=chay, daemon=True).start()
        
    def bat_dau_huan_luyen(self):
        """Bắt đầu quá trình huấn luyện"""
        def chay():
            if not self.khoi_tao_bo_huan_luyen():
                return
            try:
                self.progress_var.set(0)
                # Tạo file YAML trước khi huấn luyện
                duong_dan_yaml = self.bo_huan_luyen.tao_file_cau_hinh_yaml()

                def progress_callback(line):
                    # Ghi từng dòng vào log_text (textbox)
                    self.root.after(0, lambda: self.log_text.insert('end', line))
                    self.root.after(0, lambda: self.log_text.see('end'))
                # Gọi huấn luyện, truyền callback
                self.bo_huan_luyen.huan_luyen_model(
                    duong_dan_yaml=duong_dan_yaml,
                    so_epoch=self.so_epoch.get(),
                    kich_thuoc_anh=self.kich_thuoc_anh.get(),
                    kich_thuoc_batch=self.batch_size.get(),
                    progress_callback=progress_callback
                )            
            except Exception as e:
                self.root.after(0, lambda e=e: self.log_text.insert('end', f"Lỗi: {e}\n"))
                
        threading.Thread(target=chay, daemon=True).start()
        
    def danh_gia_model(self):
        """Đánh giá model - Phiên bản đã sửa lỗi"""
        def chay():
            if not self.khoi_tao_bo_huan_luyen():
                return
            try:
                print("\n" + "="*50)
                print("📊 ĐÁNH GIÁ MODEL")
                print("="*50)
                
                # Kiểm tra file model có tồn tại không
                model_path = self.duong_dan_model.get()
                if not os.path.exists(model_path):
                    print(f"❌ Không tìm thấy file model: {model_path}")
                    messagebox.showerror("Lỗi", f"Không tìm thấy file model: {model_path}")
                    return
                
                from ultralytics import YOLO
                import pandas as pd
                
                # Load model
                print(f"🔄 Đang tải model: {model_path}")
                model = YOLO(model_path)
                
                # Kiểm tra file dataset YAML
                yaml_path = "modelantoan\cau_hinh_day_an_toan.yaml"
                if not os.path.exists(yaml_path):
                    print(f"❌ Không tìm thấy file cấu hình: {yaml_path}")
                    print("⚠️ Vui lòng tạo file YAML trước khi đánh giá!")
                    messagebox.showerror("Lỗi", "Không tìm thấy file cấu hình data.yaml!\nVui lòng tạo file YAML trước.")
                    return
                
                # Đánh giá trên tập validation
                print("\n🔄 Đang đánh giá model trên tập validation...")
                results = model.val(data=yaml_path, verbose=True)
                
                # Kiểm tra kết quả có hợp lệ không
                if not hasattr(results, 'results_dict') or not results.results_dict:
                    print("❌ Không có kết quả đánh giá!")
                    messagebox.showerror("Lỗi", "Không thể đánh giá model. Kiểm tra lại dataset và model.")
                    return
                
                # Lấy metrics an toàn
                results_dict = results.results_dict
                
                # Tạo bảng kết quả
                try:
                    # Lấy giá trị metrics một cách an toàn
                    precision = results_dict.get('metrics/precision(B)', [0.0])
                    recall = results_dict.get('metrics/recall(B)', [0.0])
                    map50 = results_dict.get('metrics/mAP50(B)', 0.0)
                    map50_95 = results_dict.get('metrics/mAP50-95(B)', 0.0)
                    
                    # Đảm bảo precision và recall là list
                    if not isinstance(precision, list):
                        precision = [precision]
                    if not isinstance(recall, list):
                        recall = [recall]
                    
                    # Tính F1-score
                    f1_scores = []
                    for p, r in zip(precision, recall):
                        if p + r > 0:
                            f1_scores.append(2 * p * r / (p + r))
                        else:
                            f1_scores.append(0.0)
                    
                    # In kết quả tổng quan
                    print("\n📊 KẾT QUÁ ĐÁNH GIÁ TỔNG QUAN:")
                    print("─"*50)
                    print(f"mAP@50      : {map50:.3f}")
                    print(f"mAP@50-95   : {map50_95:.3f}")
                    print(f"Precision   : {precision[0]:.3f}" if precision else "Precision   : N/A")
                    print(f"Recall      : {recall[0]:.3f}" if recall else "Recall      : N/A")
                    print(f"F1-score    : {f1_scores[0]:.3f}" if f1_scores else "F1-score    : N/A")
                    print("─"*50)
                    
                    # Thông tin tốc độ
                    if hasattr(results, 'speed') and results.speed:
                        speeds = results.speed
                        print(f"\n⚡ TỐC ĐỘ XỬ LÝ:")
                        print(f"- Tiền xử lý  : {speeds.get('preprocess', 0):.1f}ms")
                        print(f"- Suy luận    : {speeds.get('inference', 0):.1f}ms") 
                        print(f"- Hậu xử lý   : {speeds.get('postprocess', 0):.1f}ms")
                    
                    print("\n✅ Hoàn thành đánh giá model!")
                    
                    # Hiển thị kết quả trong message box
                    messagebox.showinfo("Kết quả Đánh giá", 
                        f"📊 KẾT QUẢ ĐÁNH GIÁ MODEL:\n\n"
                        f"mAP@50: {map50:.3f}\n"
                        f"mAP@50-95: {map50_95:.3f}\n"
                        f"Precision: {precision[0]:.3f}\n"
                        f"Recall: {recall[0]:.3f}\n"
                        f"F1-score: {f1_scores[0]:.3f}\n\n"
                        f"⚡ Tốc độ suy luận: {speeds.get('inference', 0):.1f}ms/ảnh"
                        if hasattr(results, 'speed') and results.speed else "")
                    
                except Exception as e:
                    print(f"⚠️ Lỗi xử lý kết quả: {e}")
                    print("📊 Kết quả cơ bản:")
                    print(f"- Model đã được đánh giá thành công")
                    print(f"- Chi tiết: {results_dict}")
                    
                    messagebox.showinfo("Kết quả Đánh giá", 
                        "Model đã được đánh giá thành công!\n"
                        "Xem chi tiết trong log output.")
                    
            except ImportError as e:
                error_msg = f"❌ Lỗi import thư viện: {e}\nVui lòng cài đặt: pip install ultralytics pandas"
                print(error_msg)
                messagebox.showerror("Lỗi Import", error_msg)
                
            except FileNotFoundError as e:
                error_msg = f"❌ Không tìm thấy file: {e}"
                print(error_msg)
                messagebox.showerror("Lỗi File", error_msg)
                
            except Exception as e:
                error_msg = f"❌ Lỗi đánh giá model: {e}"
                print(error_msg)
                print(f"📝 Chi tiết lỗi: {type(e).__name__}")
                messagebox.showerror("Lỗi", error_msg)
                
        threading.Thread(target=chay, daemon=True).start()


    def du_doan_anh(self):
        """Dự đoán trên ảnh"""
        file_anh = filedialog.askopenfilename(
            title="Chọn ảnh để dự đoán",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if not file_anh:
            return
            
        def chay():
            if not self.khoi_tao_bo_huan_luyen():
                return
            try:
                print(f"🔍 Đang dự đoán ảnh: {file_anh}")
                ket_qua = self.bo_huan_luyen.du_doan_mau(file_anh, self.duong_dan_model.get())
                if ket_qua:
                    print("✅ Hoàn thành dự đoán!")
                else:
                    print("❌ Dự đoán thất bại!")
            except Exception as e:
                print(f"❌ Lỗi dự đoán: {e}")
                
        threading.Thread(target=chay, daemon=True).start()
        
    def du_doan_video(self):
        """Dự đoán trên video"""
        file_video = filedialog.askopenfilename(
            title="Chọn video để dự đoán",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        
        if not file_video:
            return
            
        def chay():
            try:
                from ultralytics import YOLO
                print(f"🎥 Đang dự đoán video: {file_video}")
                
                model = YOLO(self.duong_dan_model.get())
                ket_qua = model(file_video, save=True)
                
                print("✅ Hoàn thành dự đoán video!")
                print("📁 Kết quả đã được lưu trong thư mục runs/detect/")
                
            except Exception as e:
                print(f"❌ Lỗi dự đoán video: {e}")
                
        threading.Thread(target=chay, daemon=True).start()
        
    def xem_bieu_do(self):
        """Xem biểu đồ huấn luyện"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            
            # Tạo cửa sổ mới để hiển thị biểu đồ
            bieu_do_window = tk.Toplevel(self.root)
            bieu_do_window.title("Biểu đồ Huấn luyện")
            bieu_do_window.geometry("800x600")
            
            # Kiểm tra các file biểu đồ có tồn tại
            files_bieu_do = [
                "modelantoan/yolo_loss.png",
                "modelantoan/yolo_map.png", 
                "modelantoan/yolo_precision_recall.png"
            ]
            
            found_plots = []
            for file_path in files_bieu_do:
                if os.path.exists(file_path):
                    found_plots.append(file_path)
                    
            if not found_plots:
                tk.Label(bieu_do_window, text="Không tìm thấy biểu đồ!\nVui lòng huấn luyện model trước.",
                        font=("Arial", 12)).pack(expand=True)
                return
                
            # Hiển thị biểu đồ
            notebook = ttk.Notebook(bieu_do_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            for plot_file in found_plots:
                frame = ttk.Frame(notebook)
                plot_name = os.path.basename(plot_file).replace("yolo_", "").replace(".png", "").title()
                notebook.add(frame, text=plot_name)
                
                try:
                    img = Image.open(plot_file)
                    img = img.resize((750, 500), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    
                    label = tk.Label(frame, image=photo)
                    label.image = photo # Giữ reference
                    label.pack(expand=True)
                    
                except Exception as e:
                    tk.Label(frame, text=f"Lỗi hiển thị biểu đồ: {e}").pack(expand=True)
                    
        except Exception as e:
            print(f"❌ Lỗi xem biểu đồ: {e}")
            messagebox.showerror("Lỗi", f"Lỗi xem biểu đồ: {e}")
    def train_yolo(self, train_args, progress_callback=None):
        """Chạy huấn luyện YOLO với subprocess và gửi output về callback"""
        process = subprocess.Popen(
            train_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        for line in process.stdout:
            # Gửi từng dòng về callback
            if progress_callback:
                progress_callback(line)
                
        process.wait()

    def xoa_log(self):
        """Xóa nội dung log"""
        self.log_text.delete(1.0, tk.END)
        
    def luu_log(self):
        """Lưu log ra file"""
        file_path = filedialog.asksaveasfilename(
            title="Lưu log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                messagebox.showinfo("Thành công", f"Đã lưu log vào {file_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi lưu log: {e}")    
    def cap_nhat_tien_trinh(self, epoch, metrics, progress_str):
        """Cập nhật tiến trình huấn luyện trên giao diện"""
        try:
            # Update progress bar and force GUI update
            self.progress_var.set(float(progress_str.split('%')[0]))
            self.root.update_idletasks()
        except Exception as e:
            print(f"⚠️ Lỗi cập nhật tiến trình: {e}")

def main():
    """Hàm main chạy ứng dụng"""
    root = tk.Tk()
    app = GiaoDienHuanLuyen(root)
    
    # Xử lý khi đóng cửa sổ
    def on_closing():
        if messagebox.askokcancel("Thoát", "Bạn có muốn thoát ứng dụng?"):
            root.destroy()
            
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()

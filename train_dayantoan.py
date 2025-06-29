import os
import json
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import glob
import matplotlib.pyplot as plt
import pandas as pd

class HuanLuyenNhanDienDayAnToan:
    def __init__(self, duong_dan_du_lieu="seatbelt_dataset"):
        self.duong_dan_du_lieu = Path(duong_dan_du_lieu)
        self.duong_dan_yolo = Path("yolo_seatbelt_dataset")
        
        # Định nghĩa các lớp - chỉ 2 lớp
        self.cac_lop = {
            'with_seatbelt': 0,      # Có đeo dây an toàn
            'without_seatbelt': 1   # Không đeo dây an toàn
        }
        self.ten_cac_lop = ['co_day_an_toan', 'khong_day_an_toan']
        self.gpu_vram_gb = 4  # Tối ưu riêng cho RTX 3050 Ti 4GB
        
    def tao_cau_truc_yolo(self):
        """Tạo cấu trúc thư mục YOLO"""
        print("Đang tạo cấu trúc dataset YOLO...")
        
        # Tạo thư mục gốc
        self.duong_dan_yolo.mkdir(exist_ok=True)
        
        # Tạo các thư mục con
        for phan_chia in ['train', 'val', 'test']:
            (self.duong_dan_yolo / phan_chia / 'images').mkdir(parents=True, exist_ok=True)
            (self.duong_dan_yolo / phan_chia / 'labels').mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Đã tạo cấu trúc thư mục tại {self.duong_dan_yolo}")
        
    def tao_nhan_gia(self, duong_dan_anh, ten_lop):
        """Tạo annotation giả cho ảnh (nếu chưa có dữ liệu thực)"""
        try:
            anh = cv2.imread(str(duong_dan_anh))
            if anh is None:
                return []
                
            chieu_cao, chieu_rong = anh.shape[:2]
            
            # Tạo bounding box giả ở vùng ngực
            # Trong thực tế, cần có annotation thật từ việc gán nhãn dữ liệu
            tam_x = 0.5      # Tâm theo chiều ngang
            tam_y = 0.4      # Vùng ngực thường ở trên trung tâm
            do_rong = 0.3    # Độ rộng bounding box
            do_cao = 0.4     # Độ cao bounding box
            
            ma_lop = self.cac_lop.get(ten_lop, 2)
            
            return [f"{ma_lop} {tam_x:.6f} {tam_y:.6f} {do_rong:.6f} {do_cao:.6f}"]
            
        except Exception as loi:
            print(f"Lỗi tạo annotation cho {duong_dan_anh}: {loi}")
            return []
    
    def xu_ly_anh_va_nhan(self):
        """Xử lý ảnh và tạo nhãn"""
        print("Đang xử lý ảnh và tạo nhãn...")
        
        tat_ca_du_lieu = []
        
        # Duyệt qua từng thư mục lớp - chỉ 2 lớp
        for ten_lop in ['with_seatbelt', 'without_seatbelt']:
            duong_dan_lop = self.duong_dan_du_lieu / ten_lop
            
            if not duong_dan_lop.exists():
                print(f"⚠️  Không tìm thấy thư mục {duong_dan_lop}, bỏ qua...")
                continue
                
            print(f"Đang xử lý {ten_lop}...")
            
            # Tìm tất cả ảnh
            cac_file_anh = []
            for duoi_file in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                cac_file_anh.extend(glob.glob(str(duong_dan_lop / duoi_file)))
                
            print(f"Tìm thấy {len(cac_file_anh)} ảnh trong {ten_lop}")
            
            for duong_dan_anh in cac_file_anh:
                duong_dan_anh = Path(duong_dan_anh)
                tat_ca_du_lieu.append({
                    'duong_dan_anh': duong_dan_anh,
                    'ten_lop': ten_lop,
                    'ma_lop': self.cac_lop[ten_lop]
                })
        
        print(f"Tổng số ảnh tìm thấy: {len(tat_ca_du_lieu)}")
        return tat_ca_du_lieu
    
    def chia_va_sao_chep_du_lieu(self, tat_ca_du_lieu):
        """Chia dữ liệu và copy vào thư mục YOLO"""
        print("Đang chia và sao chép dữ liệu...")
        
        if len(tat_ca_du_lieu) == 0:
            print("❌ Không tìm thấy dữ liệu!")
            return
            
        # Chia dữ liệu: 70% train, 20% val, 10% test
        du_lieu_train, du_lieu_temp = train_test_split(tat_ca_du_lieu, test_size=0.3, random_state=42)
        du_lieu_val, du_lieu_test = train_test_split(du_lieu_temp, test_size=0.33, random_state=42)
        
        cac_phan_chia = {
            'train': du_lieu_train,
            'val': du_lieu_val,
            'test': du_lieu_test
        }
        
        for ten_phan_chia, du_lieu_phan_chia in cac_phan_chia.items():
            print(f"Đang xử lý phần {ten_phan_chia}: {len(du_lieu_phan_chia)} ảnh...")
            
            for chi_so, du_lieu in enumerate(du_lieu_phan_chia):
                # Sao chép ảnh
                anh_nguon = du_lieu['duong_dan_anh']
                anh_dich = self.duong_dan_yolo / ten_phan_chia / 'images' / f"{ten_phan_chia}_{chi_so:04d}.jpg"
                
                try:
                    shutil.copy2(anh_nguon, anh_dich)
                    
                    # Tạo file nhãn
                    duong_dan_nhan = self.duong_dan_yolo / ten_phan_chia / 'labels' / f"{ten_phan_chia}_{chi_so:04d}.txt"
                    
                    # Tạo annotation (trong thực tế cần có annotation thật)
                    cac_nhan = self.tao_nhan_gia(anh_nguon, du_lieu['ten_lop'])
                    
                    with open(duong_dan_nhan, 'w') as file:
                        file.write('\n'.join(cac_nhan))
                        
                except Exception as loi:
                    print(f"Lỗi xử lý {anh_nguon}: {loi}")
                    
        print("✓ Hoàn thành chia và sao chép dữ liệu!")
    
    def tao_file_cau_hinh_yaml(self):
        """Tạo file cấu hình YAML cho YOLO"""
        cau_hinh = {
            'path': str(self.duong_dan_yolo.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.ten_cac_lop),
            'names': self.ten_cac_lop
        }
        
        duong_dan_yaml = self.duong_dan_yolo / 'cau_hinh_day_an_toan.yaml'
        
        with open(duong_dan_yaml, 'w', encoding='utf-8') as file:
            yaml.dump(cau_hinh, file, default_flow_style=False, allow_unicode=True)
            
        print(f"✓ Đã tạo file cấu hình: {duong_dan_yaml}")
        return duong_dan_yaml
    
    def kiem_tra_gpu(self):
        """Kiểm tra và cấu hình GPU tối ưu"""
        import torch
        
        if torch.cuda.is_available():
            so_gpu = torch.cuda.device_count()
            ten_gpu = torch.cuda.get_device_name(0)
            bo_nho_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"🚀 Phát hiện {so_gpu} GPU:")
            print(f"   - GPU chính: {ten_gpu}")
            print(f"   - Bộ nhớ GPU: {bo_nho_gpu:.1f} GB")
            
            # Tối ưu bộ nhớ GPU
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # Tăng tốc cho input size cố định
            torch.backends.cudnn.deterministic = False  # Cho phép thuật toán nhanh hơn
            
            return True, so_gpu, bo_nho_gpu, ten_gpu
        else:
            print("⚠️  Không phát hiện GPU, sẽ sử dụng CPU")
            return False, 0, 0, None
    
    def tinh_batch_size_toi_uu(self, bo_nho_gpu, kich_thuoc_anh=640):
        """Tính batch size tối ưu dựa trên bộ nhớ GPU"""
        if bo_nho_gpu >= 24:  # RTX 4090, A6000, v.v.
            return 32 if kich_thuoc_anh <= 640 else 16
        elif bo_nho_gpu >= 16:  # RTX 4080, A5000, v.v.
            return 24 if kich_thuoc_anh <= 640 else 12
        elif bo_nho_gpu >= 12:  # RTX 4070 Ti, 3080 Ti, v.v.
            return 16 if kich_thuoc_anh <= 640 else 8
        elif bo_nho_gpu >= 8:   # RTX 4060 Ti, 3070, v.v.
            return 12 if kich_thuoc_anh <= 640 else 6
        elif bo_nho_gpu >= 6:   # RTX 3060, 4060, v.v.
            return 8 if kich_thuoc_anh <= 640 else 4
        elif bo_nho_gpu >= 4:   # GTX 3050, RTX 2060, v.v. (4GB VRAM)
            print("🔧 Phát hiện GPU 4GB - áp dụng tối ưu đặc biệt cho GTX 3050/RTX 2060")
            if kich_thuoc_anh <= 416:
                return 6  # Ảnh nhỏ hơn có thể batch lớn hơn
            elif kich_thuoc_anh <= 640:
                return 4  # Batch size an toàn cho 640px
            else:
                return 2  # Ảnh lớn cần batch nhỏ
        else:  # GPU nhỏ hơn 4GB
            return 2 if kich_thuoc_anh <= 640 else 1

    def huan_luyen_model(self, duong_dan_yaml, so_epoch=50, kich_thuoc_anh=640, kich_thuoc_batch=None, progress_callback=None):
        """Huấn luyện model YOLOv8n với tối ưu GPU"""
        print("=== Chuẩn Bị Huấn Luyện YOLOv8n ===")
        
        # Kiểm tra GPU
        co_gpu, so_gpu, bo_nho_gpu, ten_gpu = self.kiem_tra_gpu()
        
        # Tính batch size tối ưu nếu không được chỉ định
        if kich_thuoc_batch is None:
            if co_gpu:
                kich_thuoc_batch = self.tinh_batch_size_toi_uu(bo_nho_gpu, kich_thuoc_anh)
                print(f"📊 Batch size tối ưu được tính: {kich_thuoc_batch}")
            else:
                kich_thuoc_batch = 4  # Batch size nhỏ cho CPU
                print(f"💻 Sử dụng batch size CPU: {kich_thuoc_batch}")

        try:
            print(f"🔧 Đang tải YOLOv8n model...")
            # Load model YOLOv8n
            model = YOLO('yolov8n.pt')  # Tự động tải xuống nếu chưa có
            print("✓ Đã tải model thành công!")
            
            # Cấu hình device
            if co_gpu and so_gpu > 1:
                device = [i for i in range(so_gpu)]  # Đa GPU
                print(f"🚀 Sử dụng {so_gpu} GPU song song: {device}")
            elif co_gpu:
                device = 0  # GPU đơn
                print(f"🚀 Sử dụng GPU 0")
            else:
                device = 'cpu'
                print(f"💻 Sử dụng CPU")
            
            print(f"\n=== Bắt Đầu Huấn Luyện ===")
            print(f"📈 Epochs: {so_epoch}")
            print(f"🖼️  Kích thước ảnh: {kich_thuoc_anh}")
            print(f"📦 Batch size: {kich_thuoc_batch}")
            print(f"⚡ Device: {device}")
            
            # Callback handler for progress tracking
            if progress_callback:
                def on_train(*args, **kwargs):
                    try:
                        trainer = kwargs.get('model').trainer
                        if not trainer or not hasattr(trainer, 'metrics'):
                            return
                            
                        metrics = trainer.metrics
                        if not metrics:
                            return
                            
                        # Get the actual values from YOLO metrics
                        box_loss = metrics.get('train/box_loss', 0)
                        cls_loss = metrics.get('train/cls_loss', 0)
                        dfl_loss = metrics.get('train/dfl_loss', 0)
                        
                        # Format GPU memory
                        gpu_mem = f"{metrics.get('gpu_mem', 0):.1f}"
                        
                        # Get current epoch info
                        epoch = trainer.epoch + 1  # YOLO uses 0-based epochs
                        epochs = trainer.epochs
                        
                        # Get instances and image size
                        instances = metrics.get('instances', 0)
                        img_size = metrics.get('img_size', 640)
                        
                        # Prepare metrics dictionary
                        callback_data = {
                            'epoch': epoch,
                            'epochs': epochs,
                            'gpu_mem': gpu_mem,
                            'box_loss': box_loss,
                            'cls_loss': cls_loss,
                            'dfl_loss': dfl_loss,
                            'instances': instances,
                            'size': img_size
                        }
                        
                        progress_callback(**callback_data)
                    except Exception as e:
                        print(f"⚠️ Callback error: {str(e)}")
                        
                # Register callback with YOLO
                model.add_callback('on_train_epoch_end', on_train)

            # Tối ưu đặc biệt cho GPU 4GB
            toi_uu_cau_hinh = {}
            if co_gpu and ten_gpu and '3050' in ten_gpu.lower() and bo_nho_gpu <= self.gpu_vram_gb + 0.5:
                print(f"⚡ Phát hiện RTX 3050 Ti 4GB - ép cấu hình tối ưu!")
                kich_thuoc_anh = 416
                if kich_thuoc_batch > 4:
                    kich_thuoc_batch = 4
                toi_uu_cau_hinh.update({
                    'cache': 'gpu',
                    'workers': 2,
                    'patience': 12,
                    'nbs': 64,
                    'mosaic': 0.0,
                    'mixup': 0.0,
                    'copy_paste': 0.0,
                    'plots': False,
                    'scale': 0.3,
                    'translate': 0.08
                })
                print(f"   - imgsz: {kich_thuoc_anh}, batch: {kich_thuoc_batch}, workers: 2, cache: gpu, mosaic/mixup/copy_paste: 0.0")
            elif co_gpu and bo_nho_gpu <= self.gpu_vram_gb:
                print(f"🔧 Đang sử dụng GPU {bo_nho_gpu:.1f}GB, tối ưu cho 4GB VRAM!")
                toi_uu_cau_hinh['cache'] = 'gpu'
                toi_uu_cau_hinh['workers'] = 2
                toi_uu_cau_hinh['patience'] = 15
                toi_uu_cau_hinh['nbs'] = 64
                if kich_thuoc_anh > 416:
                    kich_thuoc_anh = 416
                    print(f"   - Điều chỉnh kích thước ảnh: {kich_thuoc_anh}px")
                if kich_thuoc_batch > 4:
                    kich_thuoc_batch = 4
                    print(f"   - Điều chỉnh batch size: {kich_thuoc_batch}")
            elif co_gpu:
                toi_uu_cau_hinh['cache'] = 'gpu'
                toi_uu_cau_hinh['workers'] = 4
                toi_uu_cau_hinh['patience'] = 20
                toi_uu_cau_hinh['nbs'] = 64
            else:
                toi_uu_cau_hinh['cache'] = False
                toi_uu_cau_hinh['workers'] = 4
                toi_uu_cau_hinh['patience'] = 20
            # Huấn luyện model với cấu hình tối ưu
            ket_qua = model.train(
                data=str(duong_dan_yaml),
                epochs=so_epoch,
                imgsz=kich_thuoc_anh,
                batch=kich_thuoc_batch,
                name='nhan_dien_day_an_toan',
                save=True,
                device=device,
                amp=True,
                close_mosaic=10,
                lr0=0.012,
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                **toi_uu_cau_hinh
            )
            
            # Tạo thư mục modelantoan nếu chưa có
            thu_muc_model = Path('modelantoan')
            thu_muc_model.mkdir(exist_ok=True)

            # Xác định thư mục output YOLO (tự động lấy từ model.trainer.save_dir nếu có)
            try:
                yolo_output_dir = Path(model.trainer.save_dir)
            except Exception:
                yolo_output_dir = Path('runs/detect/nhan_dien_day_an_toan')

            # Copy model tốt nhất vào thư mục modelantoan
            model_tot_nhat = yolo_output_dir / 'weights' / 'best.pt'
            if model_tot_nhat.exists():
                model_cuoi_cung = thu_muc_model / 'nhan_dien_day_an_toan_tot_nhat.pt'
                shutil.copy2(model_tot_nhat, model_cuoi_cung)
                print(f"✓ Đã sao chép model tốt nhất tới: {model_cuoi_cung}")
            else:
                print(f"❌ Không tìm thấy file best.pt tại {model_tot_nhat}")

            # Copy last model vào thư mục modelantoan
            model_cuoi = yolo_output_dir / 'weights' / 'last.pt'
            if model_cuoi.exists():
                model_cuoi_duong_dan = thu_muc_model / 'nhan_dien_day_an_toan_cuoi.pt'
                shutil.copy2(model_cuoi, model_cuoi_duong_dan)
                print(f"✓ Đã sao chép model cuối tới: {model_cuoi_duong_dan}")
            else:
                print(f"❌ Không tìm thấy file last.pt tại {model_cuoi}")

            # Copy file cấu hình
            cau_hinh_cuoi_cung = thu_muc_model / 'cau_hinh_day_an_toan.yaml'
            shutil.copy2(duong_dan_yaml, cau_hinh_cuoi_cung)
            print(f"✓ Đã sao chép cấu hình tới: {cau_hinh_cuoi_cung}")

            # Vẽ biểu đồ thống kê sau train
            self.ve_bieu_do_thong_ke_yolo(yolo_output_dir)

            print("✓ Hoàn thành huấn luyện!")
            print(f"✓ Model đã lưu tại: {thu_muc_model / 'nhan_dien_day_an_toan_tot_nhat.pt'}")
            return ket_qua
            
        except Exception as loi:
            print(f"❌ Huấn luyện thất bại: {loi}")
            return None
    
    def ve_bieu_do_thong_ke_yolo(self, output_dir):
        """Vẽ biểu đồ loss, mAP, precision, recall từ file results.csv của YOLO"""
        results_csv = Path(output_dir) / 'results.csv'
        if not results_csv.exists():
            print(f"❌ Không tìm thấy file {results_csv} để vẽ biểu đồ!")
            return
        try:
            df = pd.read_csv(results_csv)
            print(f"Các cột trong results.csv: {list(df.columns)}")
            # Vẽ loss nếu đủ cột
            if all(col in df.columns for col in ['epoch','train/box_loss','train/cls_loss','val/box_loss','val/cls_loss']):
                plt.figure(figsize=(10,5))
                plt.plot(df['epoch'], df['train/box_loss'], label='train/box_loss')
                plt.plot(df['epoch'], df['train/cls_loss'], label='train/cls_loss')
                plt.plot(df['epoch'], df['val/box_loss'], label='val/box_loss')
                plt.plot(df['epoch'], df['val/cls_loss'], label='val/cls_loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('YOLO Loss per Epoch')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('modelantoan/yolo_loss.png')
                plt.close()
            else:
                print('⚠️ Không đủ cột để vẽ loss!')

            # Vẽ mAP nếu có cột phù hợp
            map_col = None
            for col in ['metrics/mAP_0.5', 'metrics/mAP50', 'map_50', 'mAP_0.5']:
                if col in df.columns:
                    map_col = col
                    break
            map_col2 = None
            for col in ['metrics/mAP_0.5:0.95', 'metrics/mAP50-95', 'map_50_95', 'mAP_0.5:0.95']:
                if col in df.columns:
                    map_col2 = col
                    break
            if map_col:
                plt.figure(figsize=(8,5))
                plt.plot(df['epoch'], df[map_col], label=map_col)
                if map_col2:
                    plt.plot(df['epoch'], df[map_col2], label=map_col2)
                plt.xlabel('Epoch')
                plt.ylabel('mAP')
                plt.title('YOLO mAP per Epoch')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('modelantoan/yolo_map.png')
                plt.close()
            else:
                print('⚠️ Không tìm thấy cột mAP phù hợp để vẽ!')

            # Vẽ precision, recall nếu có
            prec_col = next((c for c in ['metrics/precision(B)','precision','metrics/precision'] if c in df.columns), None)
            rec_col = next((c for c in ['metrics/recall(B)','recall','metrics/recall'] if c in df.columns), None)
            if prec_col and rec_col:
                plt.figure(figsize=(8,5))
                plt.plot(df['epoch'], df[prec_col], label='Precision')
                plt.plot(df['epoch'], df[rec_col], label='Recall')
                plt.xlabel('Epoch')
                plt.ylabel('Value')
                plt.title('YOLO Precision & Recall per Epoch')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig('modelantoan/yolo_precision_recall.png')
                plt.close()
            else:
                print('⚠️ Không đủ cột để vẽ precision/recall!')

            # Lưu lại các giá trị val cần thiết nếu có
            val_cols = [c for c in ['epoch','val/box_loss','val/cls_loss',map_col,prec_col,rec_col] if c]
            val_cols = list(dict.fromkeys(val_cols))
            if all(col in df.columns for col in val_cols):
                df[val_cols].to_csv('modelantoan/yolo_val_stats.csv', index=False)
                print('✓ Đã lưu biểu đồ và thống kê val vào thư mục modelantoan!')
            else:
                print('⚠️ Không đủ cột để lưu thống kê val!')
        except Exception as e:
            print(f"❌ Lỗi khi vẽ biểu đồ/thống kê: {e}")

    def danh_gia_model(self, duong_dan_model=None):
        """Đánh giá model đã huấn luyện"""
        if duong_dan_model is None:
            duong_dan_model = "modelantoan/nhan_dien_day_an_toan_tot_nhat.pt"
            
        try:
            model = YOLO(duong_dan_model)
            ket_qua = model.val(data=str(self.duong_dan_yolo / 'cau_hinh_day_an_toan.yaml'))
            print("✓ Hoàn thành đánh giá!")
            return ket_qua
        except Exception as loi:
            print(f"❌ Đánh giá thất bại: {loi}")
            return None
    
    def du_doan_mau(self, duong_dan_anh, duong_dan_model=None):
        """Test dự đoán trên ảnh mẫu"""
        if duong_dan_model is None:
            duong_dan_model = "modelantoan/nhan_dien_day_an_toan_tot_nhat.pt"
            
        try:
            model = YOLO(duong_dan_model)
            ket_qua = model(duong_dan_anh)
            
            # Hiển thị kết quả
            for r in ket_qua:
                print(f"Phát hiện đối tượng trong {duong_dan_anh}:")
                for hop in r.boxes:
                    ma_lop = int(hop.cls[0])
                    do_tin_cay = float(hop.conf[0])
                    ten_lop = self.ten_cac_lop[ma_lop]
                    print(f"  - {ten_lop}: {do_tin_cay:.2f}")
                    
            return ket_qua
        except Exception as loi:
            print(f"❌ Dự đoán thất bại: {loi}")
            return None

def chuong_trinh_chinh():
    """Quy trình huấn luyện chính"""
    print("=== Huấn Luyện YOLOv8n Nhận Diện Dây An Toàn ===\n")
    
    # Khởi tạo bộ huấn luyện
    bo_huan_luyen = HuanLuyenNhanDienDayAnToan("seatbelt_dataset")
    
    # Bước 1: Tạo cấu trúc dataset
    bo_huan_luyen.tao_cau_truc_yolo()
    
    # Bước 2: Xử lý dữ liệu
    tat_ca_du_lieu = bo_huan_luyen.xu_ly_anh_va_nhan()
    
    if len(tat_ca_du_lieu) == 0:
        print("❌ Không tìm thấy dữ liệu huấn luyện! Vui lòng kiểm tra cấu trúc dataset.")
        return
        
    # Bước 3: Chia và copy dữ liệu
    bo_huan_luyen.chia_va_sao_chep_du_lieu(tat_ca_du_lieu)
    
    # Bước 4: Tạo cấu hình YAML
    cau_hinh_yaml = bo_huan_luyen.tao_file_cau_hinh_yaml()
    
    # Bước 5: Huấn luyện model
    print("\n=== Bắt Đầu Huấn Luyện ===")
    ket_qua = bo_huan_luyen.huan_luyen_model(cau_hinh_yaml, so_epoch=80, kich_thuoc_batch=8)
    
    if ket_qua:
        print("\n=== Tóm Tắt Huấn Luyện ===")
        print(f"✓ Model huấn luyện thành công!")
        print(f"✓ Model tốt nhất đã lưu tại: modelantoan/nhan_dien_day_an_toan_tot_nhat.pt")
        print(f"✓ Model cuối đã lưu tại: modelantoan/nhan_dien_day_an_toan_cuoi.pt")
        print(f"✓ Cấu hình đã lưu tại: modelantoan/cau_hinh_day_an_toan.yaml")
        
        # Bước 6: Đánh giá model
        print("\n=== Đánh Giá Model ===")
        bo_huan_luyen.danh_gia_model()
        
        print("\n=== Hoàn Thành Huấn Luyện! ===")
        print("Model đã được lưu vào thư mục 'modelantoan'")
        print("Cách sử dụng model:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('modelantoan/nhan_dien_day_an_toan_tot_nhat.pt')")
        print("  ket_qua = model('duong/dan/anh/test.jpg')")

# Hướng dẫn sử dụng
if __name__ == "__main__":
    # Chạy quy trình huấn luyện
    chuong_trinh_chinh()
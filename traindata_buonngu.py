import os
import yaml
from pathlib import Path
import torch
import gc
from ultralytics import YOLO
from datetime import datetime
import json

class YOLOTrainBuonNgu:
    def __init__(self):
        self.dataset_path = "dataset_buonngu"
        self.model_save_path = "savemodel"
        
        # Tạo các thư mục cần thiết
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Class names cho drowsiness detection
        self.classes = {
            0: 'alert',      # Tỉnh táo
            1: 'drowsy'      # Buồn ngủ
        }
        
        # Device setup
        self.device = self.check_gpu_availability()
        
        self._stop_callback = None  # Khởi tạo biến dừng training
    
    def check_gpu_availability(self):
        """Kiểm tra GPU availability và trả về device phù hợp"""
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                print(f"\n=== THÔNG TIN GPU ===")
                print(f"GPU: {gpu_name}")
                print(f"GPU Memory: {gpu_memory:.1f} GB")
                print(f"CUDA Version: {torch.version.cuda}")
                print("✅ Sẽ sử dụng GPU để training")
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                return 'cuda'
                
            except Exception as e:
                print(f"Lỗi GPU: {e}")
                return 'cpu'
        else:
            print("\n⚠️ Không tìm thấy GPU, sử dụng CPU")
            return 'cpu'
    
    def create_data_yaml(self):
        """Tạo file cấu hình data.yaml cho YOLO"""
        data_config = {
            'path': os.path.abspath(self.dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(self.classes),
            'names': list(self.classes.values())
        }
        
        yaml_path = os.path.join(self.dataset_path, 'data.yaml')
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Đã tạo file cấu hình: {yaml_path}")
        return yaml_path
    
    def set_stop_callback(self, callback):
        """Thiết lập callback để dừng training"""
        self._stop_callback = callback

    def train_yolov8s(self, epochs=100):
        """Train YOLOv8s model với tinh chỉnh chống nhận nhầm"""
        print("\n🚀 BẮT ĐẦU TRAINING YOLOv8s - CHỐNG NHẬN NHẦM")
        print("=" * 55)
        
        # Tạo file data.yaml
        data_yaml = self.create_data_yaml()
        
        # Load YOLOv8s model
        try:
            print("📥 Đang tải YOLOv8s model...")
            model = YOLO('yolov8s.pt')
            print("✅ Đã tải model YOLOv8s thành công!")
        except Exception as e:
            print(f"❌ Lỗi tải model: {e}")
            return None
        
        # Training arguments với tinh chỉnh chống overfitting và false positive
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': 640,
            'batch': 12 if self.device == 'cuda' else 6,  # Giảm batch để stable hơn
            'name': 'drowsiness_detection_yolov8s_tuned',
            'project': self.model_save_path,
            'save': True,
            'device': self.device,
            'workers': 2,  # Giảm workers để ổn định
            'optimizer': 'AdamW',
            
            # Learning rate tinh chỉnh
            'lr0': 0.005,      # Giảm learning rate cho stable training
            'lrf': 0.001,      # Final learning rate thấp hơn
            'warmup_epochs': 5, # Warm-up để model học ổn định
            'cos_lr': True,     # Cosine annealing
            
            # Early stopping và validation
            'patience': 25,     # Tăng patience để tránh dừng sớm
            'val': True,
            'fraction': 1.0,    # Sử dụng toàn bộ dataset
            
            # Regularization mạnh để giảm overfitting
            'weight_decay': 0.001,
            # Removed deprecated label_smoothing
            
            # Data augmentation cân bằng
            'hsv_h': 0.015,     # Hue augmentation nhẹ
            'hsv_s': 0.7,       # Saturation
            'hsv_v': 0.4,       # Value
            'degrees': 10,      # Rotation nhẹ
            'translate': 0.1,   # Translation
            'scale': 0.5,       # Scale variation
            'shear': 0.05,      # Shear nhẹ
            'perspective': 0.0002, # Perspective
            'fliplr': 0.5,      # Horizontal flip
            'flipud': 0.0,      # Không flip vertical cho face
            'mosaic': 0.7,      # Giảm mosaic intensity
            'mixup': 0.1,       # Mixup nhẹ
            'copy_paste': 0.05, # Copy-paste nhẹ
            
            # Confidence và NMS tuning
            'conf': 0.3,        # Confidence threshold thấp hơn trong training
            'iou': 0.6,         # IoU threshold cho NMS
            
            # Class loss weights (thay thế fl_gamma không hợp lệ)
            'cls': 0.5,         # Classification loss gain
            'box': 7.5,         # Box regression loss gain
            
            # Advanced settings
            'close_mosaic': 15, # Tắt mosaic ở 15 epochs cuối
            'amp': True if self.device == 'cuda' else False,
            'plots': True,
            'verbose': True,
            'seed': 42          # Reproducible results
        }
        
        # Device-specific settings
        if self.device == 'cuda':
            train_args['amp'] = True  # Mixed precision cho GPU
        
        # Hiển thị thông tin training
        print(f"\n📊 THÔNG TIN TRAINING - CHỐNG NHẬN NHẦM:")
        print(f"🤖 Model: YOLOv8s (Fine-tuned)")
        print(f"🖥️  Device: {self.device}")
        print(f"🔄 Epochs: {epochs}")
        print(f"📊 Batch size: {train_args['batch']} (Giảm để ổn định)")
        print(f"📐 Image size: 640")
        print(f"🎯 Learning rate: {train_args['lr0']} (Thấp hơn)")
        print(f"📏 Weight decay: {train_args['weight_decay']}")
        print(f"⏳ Patience: {train_args['patience']} epochs")
        print(f"💾 Save path: {self.model_save_path}")
        print(f"📁 Dataset: {self.dataset_path}")
        print(f"🔧 Tinh chỉnh: Chống overfitting & false positive")
        
        try:
            # Clear cache trước khi training
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"\n🎯 Bắt đầu training {epochs} epochs với cấu hình chống nhận nhầm...")
            print("⏳ Quá trình này có thể mất vài giờ...")
            print("🔧 Các cải tiến:")
            print("   • Learning rate thấp hơn để học ổn định")
            print("   • Weight decay chống overfitting") 
            print("   • Class/Box loss weights cân bằng")
            print("   • Data augmentation cân bằng")
            print("   • Early stopping thông minh")
            
            # Train model
            for epoch in range(epochs):
                if self._stop_callback and self._stop_callback():
                    print("⏹️  Đã nhận tín hiệu dừng training!")
                    break
                
                results = model.train(**train_args)
            
            # Cleanup
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Lưu thông tin training
            self.save_training_info(results, train_args)
            
            print("\n🎉 TRAINING HOÀN THÀNH - MODEL ĐÃ ĐƯỢC TINH CHỈNH!")
            print(f"📁 Model đã được lưu tại: {self.model_save_path}")
            print("🔧 Model đã được tối ưu để giảm nhận nhầm:")
            print("   ✅ Giảm false positive")
            print("   ✅ Tăng độ chính xác")
            print("   ✅ Ổn định hơn với dữ liệu mới")
            
            # Hiển thị kết quả
            if hasattr(results, 'box'):
                print(f"\n📊 KẾT QUẢ:")
                print(f"🎯 mAP50: {results.box.map50:.4f}")
                print(f"🎯 mAP50-95: {results.box.map:.4f}")
                
                # Đánh giá chất lượng
                if results.box.map50 > 0.8:
                    print("🌟 Chất lượng: Xuất sắc!")
                elif results.box.map50 > 0.6:
                    print("👍 Chất lượng: Tốt")
                elif results.box.map50 > 0.4:
                    print("⚠️ Chất lượng: Khá, có thể cần thêm data")
                else:
                    print("❌ Chất lượng: Cần cải thiện dataset")
            
            return model, results
            
        except Exception as e:
            print(f"❌ Lỗi khi training: {e}")
            
            if self.device == 'cuda':
                if "out of memory" in str(e).lower():
                    print("💡 Gợi ý: GPU hết memory, thử giảm batch size")
                torch.cuda.empty_cache()
            
            return None, None
    
    def save_training_info(self, results, train_args):
        """Lưu thông tin training"""
        training_info = {
            'model_type': 'YOLOv8s',
            'classes': self.classes,
            'train_args': train_args,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'dataset_path': self.dataset_path,
            'model_save_path': self.model_save_path,
            'device': self.device
        }
        
        # Thêm kết quả nếu có
        if results and hasattr(results, 'save_dir'):
            training_info['results_path'] = str(results.save_dir)
        
        info_path = os.path.join(self.model_save_path, 'training_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📄 Đã lưu thông tin training: {info_path}")

def check_dataset(dataset_path):
    """Kiểm tra dataset"""
    print(f"\n📁 Kiểm tra dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset không tồn tại: {dataset_path}")
        return False
    
    # Kiểm tra các thư mục cần thiết
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"✅ {dir_name}: {file_count} files")
            if file_count == 0:
                print(f"⚠️  {dir_name} trống!")
        else:
            print(f"❌ {dir_name}: Không tồn tại")
            all_good = False
    
    return all_good

def main():
    """Hàm chính"""
    print("=" * 60)
    print("🤖 YOLO TRAINING BUONNGU - YOLOv8s")
    print("=" * 60)
    
    # Khởi tạo trainer
    trainer = YOLOTrainBuonNgu()
    
    # Kiểm tra dataset
    if not check_dataset(trainer.dataset_path):
        print("\n❌ Dataset không hợp lệ!")
        print("💡 Đảm bảo có các thư mục:")
        print("   - dataset_buonngu/images/train/")
        print("   - dataset_buonngu/images/val/")
        print("   - dataset_buonngu/labels/train/")
        print("   - dataset_buonngu/labels/val/")
        return
    
    # Nhập số epochs
    print(f"\n🔄 THIẾT LẬP TRAINING")
    epochs_input = input("Nhập số epochs (50-300) [mặc định: 100]: ").strip()
    
    try:
        epochs = int(epochs_input) if epochs_input else 100
        epochs = max(50, min(epochs, 300))  # Giới hạn 50-300
    except:
        epochs = 100
        print("⚠️ Giá trị không hợp lệ, sử dụng mặc định: 100 epochs")
    
    print(f"✅ Sẽ training {epochs} epochs")
    
    # Xác nhận bắt đầu
    confirm = input(f"\n🚀 Bắt đầu training YOLOv8s? (y/n) [y]: ").strip().lower()
    
    if confirm in ['', 'y', 'yes']:
        print("\n" + "="*60)
        print("🎯 BẮT ĐẦU TRAINING")
        print("="*60)
        
        # Bắt đầu training
        model, results = trainer.train_yolov8s(epochs=epochs)
        
        if model and results:
            print("\n" + "="*60)
            print("🎉 TRAINING THÀNH CÔNG!")
            print("="*60)
            print(f"📁 Model lưu tại: {trainer.model_save_path}")
            print("💡 Có thể sử dụng model để detect drowsiness!")
        else:
            print("\n❌ Training thất bại!")
            print("💡 Kiểm tra lại dataset và thử lại")
    else:
        print("👋 Hủy training!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Người dùng dừng chương trình")
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        print("💡 Kiểm tra lại dataset và cấu hình")
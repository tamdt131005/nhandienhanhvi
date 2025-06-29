import os
import yaml
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import shutil
import cv2
import logging
import time
from typing import Optional, Dict, List, Tuple
import warnings
import gc
import psutil
import threading

# Cấu hình encoding và font tiếng Việt
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
warnings.filterwarnings('ignore', category=UserWarning)

class PhoneDetectionTrainer:
    def __init__(self, data_path="data", output_path="runs/detect", use_gpu=True):
        """
        Khởi tạo trainer với các biện pháp xử lý lỗi
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.model_path = "yolov8n.pt"
        self.use_gpu = use_gpu
        self.device = self._check_gpu_availability()
        self.stop_training = False
        self.model = None
        self.training_thread = None
        
        # Thiết lập logging
        self._setup_logging()
        
        # Kiểm tra và tạo thư mục cần thiết
        self._ensure_directories()
        
        # Kiểm tra hệ thống
        self._check_system_resources()

    def _setup_logging(self):
        """Thiết lập logging để theo dõi quá trình training"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _ensure_directories(self):
        """Đảm bảo tất cả thư mục cần thiết được tạo"""
        try:
            directories = [
                self.data_path,
                self.output_path,
                Path("model"),
                Path("logs"),
                self.data_path / "train" / "images",
                self.data_path / "train" / "labels",
                self.data_path / "val" / "images", 
                self.data_path / "val" / "labels"
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                
            self.logger.info("✅ Đã tạo tất cả thư mục cần thiết")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi tạo thư mục: {e}")
            raise

    def _check_system_resources(self):
        """Kiểm tra tài nguyên hệ thống"""
        try:
            # Kiểm tra RAM
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 2:
                self.logger.warning(f"⚠️ RAM khả dụng thấp: {available_gb:.1f}GB")
                
            # Kiểm tra dung lượng đĩa
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            
            if free_gb < 5:
                self.logger.warning(f"⚠️ Dung lượng đĩa thấp: {free_gb:.1f}GB")
                
            self.logger.info(f"💾 RAM khả dụng: {available_gb:.1f}GB")
            self.logger.info(f"💽 Dung lượng đĩa: {free_gb:.1f}GB")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Không thể kiểm tra tài nguyên hệ thống: {e}")

    def stop(self):
        """Dừng quá trình training một cách an toàn"""
        self.logger.info("\n" + "="*60)
        self.logger.info("⏹️ ĐANG DỪNG QUÁ TRÌNH TRAINING")
        self.logger.info("="*60)
        self.logger.info("→ Đã nhận lệnh dừng")
        self.logger.info("→ Đang chờ epoch hiện tại kết thúc...")
        
        self.stop_training = True
        
        # Dừng YOLO trainer nếu có
        if hasattr(self, 'model') and self.model is not None:
            if hasattr(self.model, 'trainer') and self.model.trainer is not None:
                if hasattr(self.model.trainer, 'stop'):
                    self.model.trainer.stop = True
                    self.logger.info("→ Đã gửi tín hiệu dừng đến YOLO trainer")
                    return True
        
        self.logger.warning("❌ Không thể dừng training (model chưa được khởi tạo)")
        return False

    def _check_gpu_availability(self):
        """Kiểm tra GPU và trả về device phù hợp"""
        try:
            if self.use_gpu and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                print(f"🎮 GPU khả dụng: {gpu_name}")
                print(f"🔢 Số lượng GPU: {gpu_count}")
                print(f"💾 VRAM: {memory_gb:.1f}GB")
                print(f"🔧 CUDA version: {torch.version.cuda}")
                
                # Kiểm tra VRAM có đủ không
                if memory_gb < 4:
                    print("⚠️ VRAM có thể không đủ cho training, hãy giảm batch_size")
                
                return 'cuda'
            else:
                if self.use_gpu:
                    print("⚠️ GPU được yêu cầu nhưng không khả dụng, sử dụng CPU")
                else:
                    print("🖥️ Sử dụng CPU cho training")
                return 'cpu'
                
        except Exception as e:
            print(f"❌ Lỗi kiểm tra GPU: {e}")
            return 'cpu'
    
    def create_dataset_yaml(self):
        """Tạo file cấu hình dataset cho YOLO training"""
        try:
            # Kiểm tra đường dẫn tồn tại
            if not self.data_path.exists():
                raise FileNotFoundError(f"Thư mục data không tồn tại: {self.data_path}")
            
            # Cấu hình dataset
            dataset_config = {
                'path': str(self.data_path.resolve()),
                'train': 'train/images',
                'val': 'val/images',
                'test': 'val/images',
                
                # Tên class dựa trên cấu trúc thư mục
                'names': {
                    0: 'no_phone',
                    1: 'using_phone'
                },
                'nc': 2  # Số lượng class
            }
            
            # Lưu dataset.yaml
            yaml_path = self.data_path / "dataset.yaml"
            with open(yaml_path, 'w', encoding='utf-8') as f:
                yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
                
            self.logger.info(f"✅ Đã lưu cấu hình dataset: {yaml_path}")
            return yaml_path
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi tạo file dataset.yaml: {e}")
            raise
    
    def verify_dataset_structure(self):
        """Kiểm tra cấu trúc dataset có đúng chuẩn YOLO không"""
        self.logger.info("="*50)
        self.logger.info("🔍 Đang kiểm tra cấu trúc Dataset")
        self.logger.info("="*50)
        
        try:
            required_dirs = [
                "train/images",
                "train/labels", 
                "val/images",
                "val/labels"
            ]
            
            counts = {}
            errors = []
            
            for dir_path in required_dirs:
                full_path = self.data_path / dir_path
                if full_path.exists():
                    try:
                        files = [f for f in full_path.glob("*") if f.is_file()]
                        file_count = len(files)
                        counts[dir_path] = file_count
                        self.logger.info(f"✅ {dir_path}: {file_count} files")
                        
                        # Kiểm tra định dạng file
                        if "images" in dir_path:
                            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                            invalid_files = [f for f in files if f.suffix.lower() not in valid_extensions]
                            if invalid_files:
                                errors.append(f"File ảnh không hợp lệ trong {dir_path}: {[f.name for f in invalid_files[:5]]}")
                                
                        elif "labels" in dir_path:
                            invalid_files = [f for f in files if f.suffix.lower() != '.txt']
                            if invalid_files:
                                errors.append(f"File label không hợp lệ trong {dir_path}: {[f.name for f in invalid_files[:5]]}")
                                
                    except Exception as e:
                        errors.append(f"Lỗi đọc thư mục {dir_path}: {e}")
                        counts[dir_path] = 0
                else:
                    errors.append(f"Thiếu thư mục: {dir_path}")
                    counts[dir_path] = 0
                    self.logger.error(f"❌ {dir_path}: Không tồn tại!")
            
            # Kiểm tra tương ứng image-label
            train_images = counts.get('train/images', 0)
            train_labels = counts.get('train/labels', 0) 
            val_images = counts.get('val/images', 0)
            val_labels = counts.get('val/labels', 0)
            
            self.logger.info(f"\n📊 Kiểm tra tính nhất quán dữ liệu:")
            self.logger.info(f"Train - Images: {train_images}, Labels: {train_labels} {'✅' if train_images == train_labels else '❌'}")
            self.logger.info(f"Val - Images: {val_images}, Labels: {val_labels} {'✅' if val_images == val_labels else '❌'}")
            
            # Cảnh báo
            if train_images == 0:
                errors.append("Không có ảnh training!")
            if val_images == 0:
                errors.append("Không có ảnh validation!")
            if train_images != train_labels:
                errors.append(f"Số lượng ảnh và label training không khớp: {train_images} vs {train_labels}")
            if val_images != val_labels:
                errors.append(f"Số lượng ảnh và label validation không khớp: {val_images} vs {val_labels}")
            
            # Kiểm tra nội dung label files
            self._validate_label_files()
            
            if errors:
                self.logger.error("❌ Phát hiện lỗi trong dataset:")
                for error in errors:
                    self.logger.error(f"   • {error}")
                return False
            
            return train_images > 0 and val_images > 0
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi kiểm tra cấu trúc dataset: {e}")
            return False
    
    def _validate_label_files(self):
        """Kiểm tra nội dung các file label"""
        try:
            for split in ['train', 'val']:
                labels_dir = self.data_path / split / "labels"
                if not labels_dir.exists():
                    continue
                    
                error_files = []
                for label_file in list(labels_dir.glob("*.txt"))[:10]:  # Kiểm tra 10 file đầu
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            for i, line in enumerate(lines):
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split()
                                if len(parts) != 5:
                                    error_files.append(f"{label_file.name}:line{i+1}")
                                    break
                                # Kiểm tra class_id
                                try:
                                    class_id = int(parts[0])
                                    if class_id not in [0, 1]:
                                        error_files.append(f"{label_file.name}:class_id={class_id}")
                                        break
                                except ValueError:
                                    error_files.append(f"{label_file.name}:invalid_class_id")
                                    break
                    except Exception as e:
                        error_files.append(f"{label_file.name}:read_error")
                
                if error_files:
                    self.logger.warning(f"⚠️ File label có vấn đề trong {split}: {error_files[:5]}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Không thể kiểm tra chi tiết label files: {e}")

    def analyze_dataset(self):
        """Phân tích thành phần dataset"""
        self.logger.info("\n" + "="*50)
        self.logger.info("📊 Phân tích Dataset")
        self.logger.info("="*50)
        
        try:
            train_labels_dir = self.data_path / "train" / "labels"
            val_labels_dir = self.data_path / "val" / "labels"
            
            def count_classes(labels_dir, split_name):
                class_counts = {0: 0, 1: 0}  # no_phone: 0, using_phone: 1
                total_objects = 0
                empty_files = 0
                
                if not labels_dir.exists():
                    self.logger.warning(f"⚠️ Thư mục {split_name} labels không tồn tại")
                    return class_counts, total_objects
                
                for label_file in labels_dir.glob("*.txt"):
                    try:
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            if not lines or all(not line.strip() for line in lines):
                                empty_files += 1
                                continue
                                
                            for line in lines:
                                line = line.strip()
                                if line:
                                    try:
                                        class_id = int(line.split()[0])
                                        if class_id in class_counts:
                                            class_counts[class_id] += 1
                                            total_objects += 1
                                    except (ValueError, IndexError):
                                        self.logger.warning(f"⚠️ Dòng không hợp lệ trong {label_file}: {line}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Lỗi đọc file {label_file}: {e}")
                
                self.logger.info(f"📋 {split_name}:")
                self.logger.info(f"  • Không dùng điện thoại: {class_counts[0]} objects")
                self.logger.info(f"  • Đang dùng điện thoại: {class_counts[1]} objects")
                self.logger.info(f"  • Tổng objects: {total_objects}")
                if empty_files > 0:
                    self.logger.warning(f"  • File label trống: {empty_files}")
                    
                return class_counts, total_objects
            
            train_counts, train_total = count_classes(train_labels_dir, "Training")
            val_counts, val_total = count_classes(val_labels_dir, "Validation")
            
            # Tính phần trăm
            if train_total > 0:
                train_no_phone_pct = (train_counts[0] / train_total) * 100
                train_using_phone_pct = (train_counts[1] / train_total) * 100
                self.logger.info(f"📊 Phân bố Training: Không dùng {train_no_phone_pct:.1f}%, Đang dùng {train_using_phone_pct:.1f}%")
                
            if val_total > 0:
                val_no_phone_pct = (val_counts[0] / val_total) * 100
                val_using_phone_pct = (val_counts[1] / val_total) * 100
                self.logger.info(f"📊 Phân bố Validation: Không dùng {val_no_phone_pct:.1f}%, Đang dùng {val_using_phone_pct:.1f}%")
            
            # Cảnh báo nếu dữ liệu không cân bằng
            if train_total > 0:
                imbalance_ratio = max(train_counts.values()) / min(train_counts.values()) if min(train_counts.values()) > 0 else float('inf')
                if imbalance_ratio > 3:
                    self.logger.warning(f"⚠️ Dữ liệu training không cân bằng (tỷ lệ: {imbalance_ratio:.1f}:1)")
                    
        except Exception as e:
            self.logger.error(f"❌ Lỗi phân tích dataset: {e}")

    def _optimize_training_params(self, batch_size, img_size):
        """Tối ưu tham số training dựa trên tài nguyên hệ thống"""
        try:
            # Kiểm tra VRAM nếu dùng GPU
            if self.device == 'cuda':
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if gpu_memory_gb < 6:  # VRAM thấp
                    batch_size = min(batch_size, 16)
                    img_size = min(img_size, 416)
                    self.logger.warning(f"⚠️ VRAM thấp, giảm batch_size={batch_size}, img_size={img_size}")
                elif gpu_memory_gb < 8:
                    batch_size = min(batch_size, 24)
                    
            # Kiểm tra RAM
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 4:
                batch_size = min(batch_size, 8)
                self.logger.warning(f"⚠️ RAM thấp, giảm batch_size={batch_size}")
                
            return batch_size, img_size
            
        except Exception as e:
            self.logger.warning(f"⚠️ Không thể tối ưu tham số: {e}")
            return batch_size, img_size

    def train_model(self, epochs=100, img_size=640, batch_size=32, patience=50):
        """Train YOLOv8n model cho phone detection với xử lý lỗi toàn diện"""
        self.logger.info("\n" + "="*60)
        self.logger.info("🚀 BẮT ĐẦU TRAINING YOLO PHONE DETECTION")
        self.logger.info("="*60)

        try:
            # Reset trạng thái
            self.stop_training = False
            self.logger.info("✅ Đã reset trạng thái training")
            
            # Tối ưu tham số
            batch_size, img_size = self._optimize_training_params(batch_size, img_size)
            
            # Tạo dataset yaml
            self.logger.info("\n📋 Đang tạo file cấu hình dataset...")
            dataset_yaml = self.create_dataset_yaml()
            
            # Kiểm tra cấu trúc dataset  
            self.logger.info("\n🔍 Đang kiểm tra cấu trúc dữ liệu...")
            if not self.verify_dataset_structure():
                raise ValueError("Cấu trúc dữ liệu không hợp lệ!")
                
            # Phân tích dataset
            self.analyze_dataset()
            
            # Kiểm tra model pretrained
            self._download_pretrained_model()
            
            # Khởi tạo YOLO model
            self.logger.info(f"\n" + "="*50)
            self.logger.info("🤖 Khởi tạo Model")
            self.logger.info("="*50)
            self.logger.info(f"📁 Loading YOLOv8n model: {self.model_path}")
            self.logger.info(f"🎮 Training device: {self.device}")
            
            self.model = YOLO(self.model_path)
            
            # Cấu hình training với xử lý lỗi
            training_args = self._get_training_config(
                dataset_yaml, epochs, img_size, batch_size, patience
            )
            
            self._log_training_config(training_args)
            
            # Bắt đầu training với error handling
            self.logger.info("\n🎯 BẮT ĐẦU TRAINING...")
            self.logger.info("="*50)
            
            results = self._execute_training(training_args)
            
            if results and not self.stop_training:
                self._post_training_tasks(results)
                return results
            else:
                self.logger.warning("⚠️ Training đã bị dừng hoặc thất bại")
                return None
                
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ Training bị dừng bởi người dùng (Ctrl+C)")
            self.stop_training = True
            return None
        except Exception as e:
            self.logger.error(f"❌ Training thất bại: {e}")
            self._cleanup_on_error()
            return None
        finally:
            self._cleanup_after_training()

    def _download_pretrained_model(self):
        """Tải model pretrained nếu chưa có"""
        try:
            if not Path(self.model_path).exists():
                self.logger.info(f"📥 Đang tải pretrained model: {self.model_path}")
                # YOLO sẽ tự động tải model khi khởi tạo
                temp_model = YOLO(self.model_path)
                self.logger.info("✅ Đã tải xong pretrained model")
            else:
                self.logger.info("✅ Pretrained model đã tồn tại")
        except Exception as e:
            self.logger.error(f"❌ Lỗi tải pretrained model: {e}")
            raise

    def _get_training_config(self, dataset_yaml, epochs, img_size, batch_size, patience):
        """Tạo cấu hình training với các tham số tối ưu"""
        return {
            'data': str(dataset_yaml),
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'name': 'phone_detection',
            'project': str(self.output_path),
            'device': self.device,
            'workers': min(8, os.cpu_count() or 4),
            'patience': patience,
            'save': True,
            'save_period': 10,
            'cache': False,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.1,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'cls': 0.5,
            'box': 7.5,
            'dfl': 1.5,
            'close_mosaic': 10,
            'resume': False,
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,
            'profile': False,
            'freeze': None,
            'multi_scale': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'val': True,
            'split': 'val',
            'save_json': False,
            'save_hybrid': False,
            'conf': None,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'dnn': False,
            'plots': True,
            'source': None,
            'show': False,
            'save_txt': False,
            'save_conf': False,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'vid_stride': 1,
            'stream_buffer': False,
            'line_width': None,
            'visualize': False,
            'augment': False,
            'agnostic_nms': False,
            'retina_masks': False,
            'boxes': True,
            'format': 'torchscript',
            'keras': False,
            'optimize': False,
            'int8': False,
            'dynamic': False,
            'simplify': False,
            'opset': None,
            'workspace': 4,
            'nms': False,
            'seed': 0,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': False,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }

    def _log_training_config(self, training_args):
        """Log cấu hình training"""
        self.logger.info("\n" + "="*50)
        self.logger.info("⚙️ Cấu hình Training")
        self.logger.info("="*50)
        
        key_params = [
            'epochs', 'imgsz', 'batch', 'device', 'optimizer', 
            'lr0', 'patience', 'workers', 'cache'
        ]
        
        for key in key_params:
            if key in training_args:
                self.logger.info(f"  📌 {key}: {training_args[key]}")

    def _execute_training(self, training_args):
        """Thực hiện training với monitoring"""
        start_time = time.time()
        
        try:
            # Chạy training trong thread riêng để có thể monitor
            results = self.model.train(**training_args)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"\n🎉 Training hoàn thành thành công!")
            self.logger.info(f"⏱️ Thời gian training: {duration/60:.1f} phút")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi trong quá trình training: {e}")
            raise

    def _post_training_tasks(self, results):
        """Các tác vụ sau training"""
        try:
            results_dir = self.output_path / "phone_detection"
            
            if results_dir.exists():
                self.logger.info("\n📊 Đang tạo biểu đồ và tổ chức kết quả...")
                self.create_visualization_plots(results_dir)
                
                model_dir = Path("model") / "phone_detection5"
                self.create_summary_report(model_dir, results)
                
                self.logger.info("✅ Đã hoàn thành tất cả tác vụ sau training")
                
        except Exception as e:
            self.logger.error(f"⚠️ Lỗi trong post-training tasks: {e}")

    def _cleanup_on_error(self):
        """Dọn dẹp khi có lỗi"""
        try:
            # Giải phóng GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Giải phóng RAM
            gc.collect()
            
            self.logger.info("🧹 Đã dọn dẹp tài nguyên sau lỗi")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Không thể dọn dẹp hoàn toàn: {e}")

    def _cleanup_after_training(self):
        """Dọn dẹp sau khi training kết thúc"""
        try:
            # Giải phóng GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Giải phóng RAM
            gc.collect()
            
            self.logger.info("🧹 Đã dọn dẹp tài nguyên")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Lỗi dọn dẹp: {e}")

    def create_visualization_plots(self, results_dir):
        """Tạo các biểu đồ trực quan hóa kết quả training"""
        self.logger.info("\n" + "="*50)
        self.logger.info("📊 Tạo biểu đồ trực quan")
        self.logger.info("="*50)
        
        try:
            plt.style.use('default')
            
            # Đọc kết quả training
            results_csv = results_dir / "results.csv"
            if not results_csv.exists():
                self.logger.warning("⚠️ Không tìm thấy file results.csv")
                return
            
            df = pd.read_csv(results_csv)
            self.logger.info(f"✅ Đã đọc {len(df)} epochs từ results.csv")
            
            # Tạo figure với subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('YOLOv8 Phone Detection - Kết quả Training', fontsize=16, y=0.98)
            
            # 1. Loss curves
            ax1 = axes[0, 0]
            if 'train/box_loss' in df.columns:
                ax1.plot(df.index, df['train/box_loss'], label='Train Box Loss', color='blue', alpha=0.7)
            if 'val/box_loss' in df.columns:
                ax1.plot(df.index, df['val/box_loss'], label='Val Box Loss', color='red', alpha=0.7)
            if 'train/cls_loss' in df.columns:
                ax1.plot(df.index, df['train/cls_loss'], label='Train Cls Loss', color='green', alpha=0.7)
            if 'val/cls_loss' in df.columns:
                ax1.plot(df.index, df['val/cls_loss'], label='Val Cls Loss', color='orange', alpha=0.7)
                
            ax1.set_title('Training và Validation Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. mAP metrics
            ax2 = axes[0, 1]
            if 'metrics/mAP50(B)' in df.columns:
                ax2.plot(df.index, df['metrics/mAP50(B)'], label='mAP50', color='purple', linewidth=2)
            if 'metrics/mAP50-95(B)' in df.columns:
                ax2.plot(df.index, df['metrics/mAP50-95(B)'], label='mAP50-95', color='brown', linewidth=2)
                
            ax2.set_title('mAP Metrics')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('mAP')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            # 3. Precision và Recall
            ax3 = axes[1, 0]
            if 'metrics/precision(B)' in df.columns:
                ax3.plot(df.index, df['metrics/precision(B)'], label='Precision', color='darkgreen', linewidth=2)
            if 'metrics/recall(B)' in df.columns:
                ax3.plot(df.index, df['metrics/recall(B)'], label='Recall', color='darkred', linewidth=2)
                
            ax3.set_title('Precision và Recall')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            # 4. Learning Rate
            ax4 = axes[1, 1]
            if 'lr/pg0' in df.columns:
                ax4.plot(df.index, df['lr/pg0'], label='Learning Rate', color='black', linewidth=2)
            ax4.set_title('Learning Rate')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('LR')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
            
            plt.tight_layout()
            
            # Lưu biểu đồ
            plots_dir = results_dir / "custom_plots"
            plots_dir.mkdir(exist_ok=True)
            
            plot_path = plots_dir / "training_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Đã lưu biểu đồ: {plot_path}")
            
            # Tạo biểu đồ so sánh classes
            self._create_class_performance_plot(df, plots_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi tạo biểu đồ: {e}")
            plt.close('all')

    def _create_class_performance_plot(self, df, plots_dir):
        """Tạo biểu đồ hiệu suất từng class"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Tìm columns cho từng class
            class_columns = {}
            for col in df.columns:
                if 'no_phone' in col.lower():
                    if 'precision' not in class_columns:
                        class_columns['precision'] = []
                    if 'recall' not in class_columns:
                        class_columns['recall'] = []
                    if 'precision' in col.lower():
                        class_columns['precision'].append(col)
                    elif 'recall' in col.lower():
                        class_columns['recall'].append(col)
            
            # Plot nếu có data
            colors = ['blue', 'red', 'green', 'orange']
            for i, (metric, cols) in enumerate(class_columns.items()):
                for j, col in enumerate(cols):
                    if col in df.columns:
                        plt.plot(df.index, df[col], 
                                label=f'{metric.title()} - {col.split("/")[-1]}',
                                color=colors[(i*2+j) % len(colors)],
                                linewidth=2, alpha=0.8)
            
            plt.title('Hiệu suất theo từng Class', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            plt.tight_layout()
            
            class_plot_path = plots_dir / "class_performance.png"
            plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"✅ Đã lưu biểu đồ class performance: {class_plot_path}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Không thể tạo biểu đồ class performance: {e}")
            plt.close('all')

    def create_summary_report(self, model_dir, results):
        """Tạo báo cáo tổng kết training"""
        self.logger.info("\n📋 Tạo báo cáo tổng kết...")
        
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            report_path = model_dir / "training_report.md"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Báo cáo Training YOLOv8 Phone Detection\n\n")
                f.write(f"**Ngày tạo:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("## Thông tin Model\n")
                f.write("- **Architecture:** YOLOv8n\n")
                f.write("- **Task:** Object Detection (Phone Usage)\n")
                f.write(f"- **Device:** {self.device}\n")
                f.write("- **Classes:** no_phone (0), using_phone (1)\n\n")
                
                f.write("## Cấu hình Training\n")
                if hasattr(results, 'args'):
                    args = results.args
                    f.write(f"- **Epochs:** {getattr(args, 'epochs', 'N/A')}\n")
                    f.write(f"- **Batch Size:** {getattr(args, 'batch', 'N/A')}\n")
                    f.write(f"- **Image Size:** {getattr(args, 'imgsz', 'N/A')}\n")
                    f.write(f"- **Optimizer:** {getattr(args, 'optimizer', 'N/A')}\n")
                    f.write(f"- **Learning Rate:** {getattr(args, 'lr0', 'N/A')}\n")
                
                f.write("\n## Kết quả Training\n")
                if hasattr(results, 'results_dict'):
                    metrics = results.results_dict
                    f.write(f"- **Best mAP50:** {metrics.get('metrics/mAP50(B)', 'N/A'):.4f}\n")
                    f.write(f"- **Best mAP50-95:** {metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}\n")
                    f.write(f"- **Precision:** {metrics.get('metrics/precision(B)', 'N/A'):.4f}\n")
                    f.write(f"- **Recall:** {metrics.get('metrics/recall(B)', 'N/A'):.4f}\n")
                
                f.write("\n## Files được tạo\n")
                f.write("- `best.pt` - Model với kết quả tốt nhất\n")
                f.write("- `last.pt` - Model ở epoch cuối\n")
                f.write("- `results.csv` - Chi tiết metrics từng epoch\n")
                f.write("- `confusion_matrix.png` - Ma trận nhầm lẫn\n")
                f.write("- `training_metrics.png` - Biểu đồ training\n")
                
                f.write("\n## Cách sử dụng\n")
                f.write("```python\n")
                f.write("from ultralytics import YOLO\n")
                f.write("model = YOLO('path/to/best.pt')\n")
                f.write("results = model('path/to/image.jpg')\n")
                f.write("```\n")
            
            self.logger.info(f"✅ Đã tạo báo cáo: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi tạo báo cáo: {e}")

    def copy_best_model(self):
        """Copy model tốt nhất vào thư mục model"""
        try:
            results_dir = self.output_path / "phone_detection"
            weights_dir = results_dir / "weights"
            
            if not weights_dir.exists():
                self.logger.warning("⚠️ Không tìm thấy thư mục weights")
                return False
            
            best_model = weights_dir / "best.pt" 
            last_model = weights_dir / "last.pt"
            
            model_dir = Path("model")
            model_dir.mkdir(exist_ok=True)
            
            if best_model.exists():
                shutil.copy2(best_model, model_dir / "phone_detection_best.pt")
                self.logger.info(f"✅ Đã copy best model: {model_dir / 'phone_detection_best.pt'}")
                
            if last_model.exists():
                shutil.copy2(last_model, model_dir / "phone_detection_last.pt") 
                self.logger.info(f"✅ Đã copy last model: {model_dir / 'phone_detection_last.pt'}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi copy model: {e}")
            return False

    def evaluate_model(self, model_path=None):
        """Đánh giá model trên validation set"""
        self.logger.info("\n" + "="*50)
        self.logger.info("🔍 Đánh giá Model")
        self.logger.info("="*50)
        
        try:
            if model_path is None:
                model_path = Path("model") / "phone_detection_best.pt"
                
            if not Path(model_path).exists():
                self.logger.error(f"❌ Không tìm thấy model: {model_path}")
                return None
                
            # Load model
            model = YOLO(model_path)
            self.logger.info(f"✅ Đã load model: {model_path}")
            
            # Đánh giá trên validation set
            dataset_yaml = self.data_path / "dataset.yaml"
            if not dataset_yaml.exists():
                dataset_yaml = self.create_dataset_yaml()
                
            results = model.val(data=str(dataset_yaml), device=self.device)
            
            self.logger.info("📊 Kết quả đánh giá:")
            self.logger.info(f"  • mAP50: {results.box.map50:.4f}")
            self.logger.info(f"  • mAP50-95: {results.box.map:.4f}")
            self.logger.info(f"  • Precision: {results.box.mp:.4f}")
            self.logger.info(f"  • Recall: {results.box.mr:.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi đánh giá model: {e}")
            return None

    def predict_image(self, image_path, model_path=None, conf_threshold=0.5):
        """Dự đoán trên một ảnh"""
        try:
            if model_path is None:
                model_path = Path("model") / "phone_detection_best.pt"
                
            if not Path(model_path).exists():
                self.logger.error(f"❌ Không tìm thấy model: {model_path}")
                return None
                
            # Load model
            model = YOLO(model_path)
            
            # Dự đoán
            results = model(image_path, conf=conf_threshold, device=self.device)
            
            self.logger.info(f"✅ Đã dự đoán ảnh: {image_path}")
            
            # Hiển thị kết quả
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    self.logger.info(f"📱 Phát hiện {len(boxes)} objects:")
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        class_name = "using_phone" if class_id == 1 else "no_phone"
                        self.logger.info(f"  • Object {i+1}: {class_name} (confidence: {confidence:.3f})")
                else:
                    self.logger.info("📱 Không phát hiện object nào")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Lỗi dự đoán: {e}")
            return None

    def monitor_training_progress(self):
        """Monitor tiến trình training"""
        try:
            results_dir = self.output_path / "phone_detection"
            results_csv = results_dir / "results.csv"
            
            if not results_csv.exists():
                return None
                
            df = pd.read_csv(results_csv)
            latest_epoch = len(df) - 1
            
            if latest_epoch >= 0:
                latest_metrics = df.iloc[-1]
                
                progress_info = {
                    'epoch': latest_epoch,
                    'train_loss': latest_metrics.get('train/box_loss', 0),
                    'val_loss': latest_metrics.get('val/box_loss', 0),
                    'mAP50': latest_metrics.get('metrics/mAP50(B)', 0),
                    'precision': latest_metrics.get('metrics/precision(B)', 0),
                    'recall': latest_metrics.get('metrics/recall(B)', 0)
                }
                
                return progress_info
                
        except Exception as e:
            self.logger.warning(f"⚠️ Không thể monitor progress: {e}")
            return None

def main():
    """Hàm main để chạy training"""
    print("🚀 YOLOv8 Phone Detection Trainer - Phiên bản cải tiến")
    print("="*60)
    
    try:
        # Khởi tạo trainer
        trainer = PhoneDetectionTrainer(
            data_path="data",
            output_path="runs/detect", 
            use_gpu=True
        )
        
        print("\n📋 Menu chức năng:")
        print("1. Kiểm tra cấu trúc dữ liệu")
        print("2. Phân tích dataset")
        print("3. Training model")
        print("4. Đánh giá model")
        print("5. Dự đoán ảnh")
        print("6. Thoát")
        
        while True:
            try:
                choice = input("\n👉 Chọn chức năng (1-6): ").strip()
                
                if choice == '1':
                    trainer.verify_dataset_structure()
                    
                elif choice == '2':
                    trainer.analyze_dataset()
                    
                elif choice == '3':
                    print("\n⚙️ Cấu hình training:")
                    epochs = int(input("Số epochs (mặc định 100): ") or "100")
                    batch_size = int(input("Batch size (mặc định 32): ") or "32")
                    img_size = int(input("Image size (mặc định 640): ") or "640")
                    patience = int(input("Patience (mặc định 50): ") or "50")
                    
                    print(f"\n🎯 Bắt đầu training với:")
                    print(f"  • Epochs: {epochs}")
                    print(f"  • Batch size: {batch_size}")
                    print(f"  • Image size: {img_size}")
                    print(f"  • Patience: {patience}")
                    
                    results = trainer.train_model(
                        epochs=epochs,
                        batch_size=batch_size, 
                        img_size=img_size,
                        patience=patience
                    )
                    
                    if results:
                        trainer.copy_best_model()
                        print("\n🎉 Training hoàn thành thành công!")
                    else:
                        print("\n❌ Training thất bại hoặc bị dừng")
                
                elif choice == '4':
                    trainer.evaluate_model()
                    
                elif choice == '5':
                    image_path = input("Đường dẫn ảnh: ").strip()
                    if image_path and Path(image_path).exists():
                        conf = float(input("Confidence threshold (0.5): ") or "0.5")
                        trainer.predict_image(image_path, conf_threshold=conf)
                    else:
                        print("❌ Ảnh không tồn tại!")
                        
                elif choice == '6':
                    print("👋 Tạm biệt!")
                    break
                    
                else:
                    print("❌ Lựa chọn không hợp lệ!")
                    
            except KeyboardInterrupt:
                print("\n⏹️ Đã dừng chương trình")
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                
    except Exception as e:
        print(f"❌ Lỗi khởi tạo: {e}")

if __name__ == "__main__":
    main()
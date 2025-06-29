import cv2
import numpy as np
import os
import json
import math
import shutil
import random
from datetime import datetime
from pathlib import Path
import mediapipe as mp
import yaml
from sklearn.model_selection import train_test_split

class ThuThapDuLieuYOLO:
    def __init__(self):
        # Khoi tao MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,  # Giảm xuống 1 để tập trung vào 1 khuôn mặt
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Tăng confidence để có detection chính xác hơn
            min_tracking_confidence=0.7
        )
        
        # Dinh nghia chi so cho cac diem landmarks cua mat (chuẩn 68-point)
        self.CHI_SO_MAT_TRAI = [362, 385, 387, 263, 373, 380]
        self.CHI_SO_MAT_PHAI = [33, 160, 158, 133, 153, 144]
        
        # Cau hinh YOLO
        self.thu_muc_dataset = "dataset_buonngu"
        self.thu_muc_raw = "raw_data"
        self.classes = ['alert', 'drowsy']  # Đổi tên class cho rõ ràng
        self.kich_thuoc_anh = 640  # Kích thước chuẩn YOLO
        
        # Cau hinh chia du lieu - cải tiến để cân bằng hơn
        self.ty_le_train = 0.7
        self.ty_le_val = 0.2
        self.ty_le_test = 0.1
        
        # Thong ke du lieu
        self.so_mau_alert = 0
        self.so_mau_drowsy = 0
        self.file_metadata = os.path.join(self.thu_muc_dataset, "metadata.json")
        
        # Ngưỡng EAR cho phân loại (có thể điều chỉnh)
        self.NGUONG_BUON_NGU = 0.27
        self.history_ear = []  # Lưu lịch sử EAR để smooth detection
        self.history_size = 8
        
        # Data augmentation settings
        self.enable_augmentation = True
        self.augment_ratio = 0.3  # 30% dữ liệu sẽ được augment
        
        # Tao cau truc thu muc YOLO
        self.tao_cau_truc_yolo()
        self.tai_metadata()
    
    def tao_cau_truc_yolo(self):
        """Tao cau truc thu muc chuan cho YOLO dataset"""
        thu_mucs = [
            self.thu_muc_dataset,
            self.thu_muc_raw,
            os.path.join(self.thu_muc_raw, "images"),
            os.path.join(self.thu_muc_raw, "labels"),
            os.path.join(self.thu_muc_dataset, "images"),
            os.path.join(self.thu_muc_dataset, "images", "train"),
            os.path.join(self.thu_muc_dataset, "images", "val"),
            os.path.join(self.thu_muc_dataset, "images", "test"),  # Thêm test set
            os.path.join(self.thu_muc_dataset, "labels"),
            os.path.join(self.thu_muc_dataset, "labels", "train"),
            os.path.join(self.thu_muc_dataset, "labels", "val"),
            os.path.join(self.thu_muc_dataset, "labels", "test")   # Thêm test labels
        ]
        
        for thu_muc in thu_mucs:
            if not os.path.exists(thu_muc):
                os.makedirs(thu_muc)
                print(f"Da tao thu muc: {thu_muc}")
        
        self.tao_config_yaml()
    
    def tao_config_yaml(self):
        """Tao file cau hinh YAML cho YOLO"""
        config = {
            'path': os.path.abspath(self.thu_muc_dataset),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',  # Thêm test set
            'names': {
                0: 'alert',
                1: 'drowsy'
            },
            'nc': 2
        }
        
        config_path = os.path.join(self.thu_muc_dataset, "data.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"Da tao file config: {config_path}")
    
    def tai_metadata(self):
        """Tai thong tin metadata"""
        try:
            if os.path.exists(self.file_metadata):
                with open(self.file_metadata, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    self.so_mau_drowsy = metadata.get('drowsy', 0)
                    self.so_mau_alert = metadata.get('alert', 0)
                print(f"Da tai metadata: Alert: {self.so_mau_alert}, Drowsy: {self.so_mau_drowsy}")
        except Exception as e:
            print(f"Loi khi tai metadata: {e}")
    
    def luu_metadata(self):
        """Luu thong tin metadata"""
        try:
            metadata = {
                'alert': self.so_mau_alert,
                'drowsy': self.so_mau_drowsy,
                'total': self.so_mau_alert + self.so_mau_drowsy,
                'balance_ratio': self.so_mau_alert / max(self.so_mau_drowsy, 1),
                'updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'classes': self.classes,
                'ear_threshold': self.NGUONG_BUON_NGU
            }
            with open(self.file_metadata, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Loi khi luu metadata: {e}")
    
    def tinh_toan_ear(self, cac_diem_mat):
        """Tinh toan Eye Aspect Ratio (EAR) - cải tiến để chính xác hơn"""
        if len(cac_diem_mat) < 6:
            return 0.0
            
        # Tính khoảng cách theo công thức EAR chuẩn
        # EAR = (|p1-p5| + |p2-p4|) / (2 * |p0-p3|)
        vertical_1 = np.linalg.norm(cac_diem_mat[1] - cac_diem_mat[5])
        vertical_2 = np.linalg.norm(cac_diem_mat[2] - cac_diem_mat[4])
        horizontal = np.linalg.norm(cac_diem_mat[0] - cac_diem_mat[3])
        
        if horizontal == 0:
            return 0.0
            
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def trich_xuat_diem_mat(self, face_landmarks, chi_so_mat):
        """Trích xuất điểm landmark của mắt với error handling tốt hơn"""
        cac_diem = []
        h, w = 480, 640  # Default resolution
        
        for idx in chi_so_mat:
            try:
                landmark = face_landmarks.landmark[idx]
                cac_diem.append(np.array([landmark.x * w, landmark.y * h]))
            except (IndexError, AttributeError):
                # Nếu không tìm thấy landmark, dùng giá trị mặc định
                cac_diem.append(np.array([0.0, 0.0]))
        
        return np.array(cac_diem)
    
    def phat_hien_khuon_mat_v2(self, frame, face_landmarks):
        """Phát hiện bounding box khuôn mặt được cải tiến"""
        h, w, _ = frame.shape
        
        # Lấy tất cả điểm landmark
        x_coords = []
        y_coords = []
        
        for landmark in face_landmarks.landmark:
            x_coords.append(landmark.x * w)
            y_coords.append(landmark.y * h)
        
        if not x_coords or not y_coords:
            return None
        
        # Tính bounding box với margin phù hợp
        margin_x = 0.1 * (max(x_coords) - min(x_coords))
        margin_y = 0.1 * (max(y_coords) - min(y_coords))
        
        x_min = max(0, int(min(x_coords) - margin_x))
        x_max = min(w, int(max(x_coords) + margin_x))
        y_min = max(0, int(min(y_coords) - margin_y))
        y_max = min(h, int(max(y_coords) + margin_y))
        
        # Đảm bảo bounding box có tỷ lệ hợp lý
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        
        if bbox_w < 50 or bbox_h < 50:  # Bounding box quá nhỏ
            return None
        
        # Chuyển sang định dạng YOLO (normalized)
        center_x = (x_min + x_max) / 2.0 / w
        center_y = (y_min + y_max) / 2.0 / h
        width = bbox_w / w
        height = bbox_h / h
        
        return center_x, center_y, width, height, (x_min, y_min, x_max, y_max)
    
    def danh_gia_buon_ngu_v2(self, face_landmarks):
        """Đánh giá buồn ngủ với smooth detection"""
        mat_trai = self.trich_xuat_diem_mat(face_landmarks, self.CHI_SO_MAT_TRAI)
        mat_phai = self.trich_xuat_diem_mat(face_landmarks, self.CHI_SO_MAT_PHAI)
        
        ear_trai = self.tinh_toan_ear(mat_trai)
        ear_phai = self.tinh_toan_ear(mat_phai)
        ear_trung_binh = (ear_trai + ear_phai) / 2.0
        
        # Thêm vào lịch sử để smooth
        self.history_ear.append(ear_trung_binh)
        if len(self.history_ear) > self.history_size:
            self.history_ear.pop(0)
        
        # Tính EAR smooth
        smooth_ear = np.mean(self.history_ear) if self.history_ear else ear_trung_binh
        
        # Phân loại với ngưỡng
        is_drowsy = smooth_ear < self.NGUONG_BUON_NGU
        confidence = abs(smooth_ear - self.NGUONG_BUON_NGU) / self.NGUONG_BUON_NGU
        
        return smooth_ear, is_drowsy, confidence
    
    def data_augmentation(self, image):
        """Thêm data augmentation để tăng cường dữ liệu"""
        if not self.enable_augmentation or random.random() > self.augment_ratio:
            return image
        
        # Random brightness
        if random.random() > 0.5:
            brightness = random.uniform(0.7, 1.3)
            image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        # Random noise
        if random.random() > 0.5:
            noise = np.random.randint(0, 25, image.shape, dtype=np.uint8)
            image = cv2.add(image, noise)
        
        # Random blur
        if random.random() > 0.7:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        return image
    
    def validate_sample_quality(self, frame, face_landmarks, ear_value):
        """Kiểm tra chất lượng sample trước khi lưu"""
        # Kiểm tra độ sáng của ảnh
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 50 or brightness > 200:  # Quá tối hoặc quá sáng
            return False, "Brightness out of range"
        
        # Kiểm tra độ mờ
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:  # Ảnh quá mờ
            return False, "Image too blurry"
        
        # Kiểm tra EAR hợp lệ
        if ear_value <= 0 or ear_value > 0.5:
            return False, "Invalid EAR value"
        
        # Kiểm tra số lượng landmarks
        if len(face_landmarks.landmark) < 468:
            return False, "Insufficient landmarks"
        
        return True, "OK"
    
    def luu_mau_yolo_v2(self, frame, face_landmarks, loai_mau=None):
        """Lưu mẫu YOLO được cải tiến"""
        if not face_landmarks:
            print("Khong phat hien khuon mat")
            return False
        
        try:
            # Phát hiện bounding box
            bbox_result = self.phat_hien_khuon_mat_v2(frame, face_landmarks)
            if bbox_result is None:
                print("Khong the tao bounding box")
                return False
            
            center_x, center_y, width, height, bbox_coords = bbox_result
            
            # Đánh giá buồn ngủ
            ear_value, is_drowsy, confidence = self.danh_gia_buon_ngu_v2(face_landmarks)
            
            # Kiểm tra chất lượng sample
            is_valid, quality_msg = self.validate_sample_quality(frame, face_landmarks, ear_value)
            if not is_valid:
                print(f"Sample quality issue: {quality_msg}")
                return False
            
            # Xác định class
            if loai_mau is not None:
                class_id = loai_mau
                class_name = self.classes[class_id]
            else:
                # Auto detection với confidence threshold
                if confidence < 0.3:  # Confidence thấp, không lưu
                    print(f"Low confidence: {confidence:.3f}")
                    return False
                
                class_id = 1 if is_drowsy else 0
                class_name = self.classes[class_id]
            
            # Cập nhật counter
            if class_id == 0:
                self.so_mau_alert += 1
            else:
                self.so_mau_drowsy += 1
            
            # Data augmentation
            processed_frame = self.data_augmentation(frame.copy())
            
            # Resize theo chuẩn YOLO
            resized_frame, scale, x_offset, y_offset = self.resize_image_v2(processed_frame)
            
            # Adjust bounding box
            new_center_x = (center_x * frame.shape[1] * scale + x_offset) / self.kich_thuoc_anh
            new_center_y = (center_y * frame.shape[0] * scale + y_offset) / self.kich_thuoc_anh
            new_width = width * scale
            new_height = height * scale
            
            # Clamp values
            new_center_x = max(0, min(1, new_center_x))
            new_center_y = max(0, min(1, new_center_y))
            new_width = max(0, min(1, new_width))
            new_height = max(0, min(1, new_height))
            
            # Tạo tên file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            ten_file = f"{class_name}_{timestamp}_ear{ear_value:.3f}"
            
            # Lưu file
            duong_dan_anh = os.path.join(self.thu_muc_raw, "images", f"{ten_file}.jpg")
            duong_dan_label = os.path.join(self.thu_muc_raw, "labels", f"{ten_file}.txt")
            
            # Lưu ảnh với chất lượng cao
            cv2.imwrite(duong_dan_anh, resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Lưu label YOLO
            with open(duong_dan_label, 'w') as f:
                f.write(f"{class_id} {new_center_x:.6f} {new_center_y:.6f} {new_width:.6f} {new_height:.6f}\n")
            
            # Lưu metadata chi tiết
            metadata = {
                'timestamp': timestamp,
                'class_id': class_id,
                'class_name': class_name,
                'ear_value': float(ear_value),
                'confidence': float(confidence),
                'is_drowsy': is_drowsy,
                'bbox_yolo': [float(new_center_x), float(new_center_y), float(new_width), float(new_height)],
                'image_quality': quality_msg,
                'augmented': self.enable_augmentation and random.random() <= self.augment_ratio
            }
            
            info_path = os.path.join(self.thu_muc_raw, f"{ten_file}_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.luu_metadata()
            
            print(f"✓ Saved {class_name} - EAR: {ear_value:.3f}, Conf: {confidence:.3f}")
            return True
            
        except Exception as e:
            print(f"Loi khi luu mau: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def resize_image_v2(self, image):
        """Resize ảnh với letterbox để giữ tỷ lệ"""
        h, w = image.shape[:2]
        scale = self.kich_thuoc_anh / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Tạo ảnh với padding
        new_image = np.full((self.kich_thuoc_anh, self.kich_thuoc_anh, 3), 114, dtype=np.uint8)
        
        # Center the image
        y_offset = (self.kich_thuoc_anh - new_h) // 2
        x_offset = (self.kich_thuoc_anh - new_w) // 2
        new_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return new_image, scale, x_offset, y_offset
    
    def chia_du_lieu_train_val_test(self):
        """Chia dữ liệu thành train/val/test với stratified sampling"""
        print("\n=== CHIA DU LIEU TRAIN/VAL/TEST ===")
        
        # Xóa dữ liệu cũ
        self.xoa_du_lieu_cu()
        
        # Lấy danh sách file
        thu_muc_anh = os.path.join(self.thu_muc_raw, "images")
        if not os.path.exists(thu_muc_anh):
            print("Khong tim thay thu muc raw/images!")
            return False
        
        files = [f for f in os.listdir(thu_muc_anh) if f.endswith('.jpg')]
        if len(files) == 0:
            print("Khong tim thay file anh nao!")
            return False
        
        # Phân loại theo class
        alert_files = [f for f in files if f.startswith('alert_')]
        drowsy_files = [f for f in files if f.startswith('drowsy_')]
        
        print(f"Alert files: {len(alert_files)}")
        print(f"Drowsy files: {len(drowsy_files)}")
        
        # Kiểm tra balance
        if len(alert_files) == 0 or len(drowsy_files) == 0:
            print("Cảnh báo: Thiếu dữ liệu cho một trong hai class!")
        
        balance_ratio = len(alert_files) / max(len(drowsy_files), 1)
        print(f"Balance ratio (alert/drowsy): {balance_ratio:.2f}")
        
        # Stratified split cho mỗi class
        def split_files(file_list, class_name):
            if len(file_list) < 3:
                print(f"Cảnh báo: Class {class_name} có quá ít dữ liệu ({len(file_list)} files)")
                return file_list, [], []
            
            # First split: train vs (val+test)
            train_files, temp_files = train_test_split(
                file_list, 
                train_size=self.ty_le_train, 
                random_state=42,
                shuffle=True
            )
            
            # Second split: val vs test
            if len(temp_files) >= 2:
                val_size = self.ty_le_val / (self.ty_le_val + self.ty_le_test)
                val_files, test_files = train_test_split(
                    temp_files,
                    train_size=val_size,
                    random_state=42,
                    shuffle=True
                )
            else:
                val_files = temp_files
                test_files = []
            
            return train_files, val_files, test_files
        
        # Split cho mỗi class
        alert_train, alert_val, alert_test = split_files(alert_files, "alert")
        drowsy_train, drowsy_val, drowsy_test = split_files(drowsy_files, "drowsy")
        
        # Combine các split
        all_train = alert_train + drowsy_train
        all_val = alert_val + drowsy_val
        all_test = alert_test + drowsy_test
        
        # Shuffle
        random.shuffle(all_train)
        random.shuffle(all_val)
        random.shuffle(all_test)
        
        # Copy files
        self.copy_files_v2(all_train, "train")
        self.copy_files_v2(all_val, "val")
        self.copy_files_v2(all_test, "test")
        
        # Cập nhật config
        self.tao_config_yaml()
        
        print(f"\n=== KET QUA CHIA DU LIEU ===")
        print(f"Train: {len(all_train)} files ({len(alert_train)} alert, {len(drowsy_train)} drowsy)")
        print(f"Val: {len(all_val)} files ({len(alert_val)} alert, {len(drowsy_val)} drowsy)")
        print(f"Test: {len(all_test)} files ({len(alert_test)} alert, {len(drowsy_test)} drowsy)")
        print(f"Total: {len(files)} files")
        
        return True
    
    def copy_files_v2(self, file_list, split_type):
        """Copy files với error handling tốt hơn"""
        if len(file_list) == 0:
            print(f"Khong co file nao de copy cho {split_type}")
            return
        
        src_img_dir = os.path.join(self.thu_muc_raw, "images")
        src_lbl_dir = os.path.join(self.thu_muc_raw, "labels")
        
        dst_img_dir = os.path.join(self.thu_muc_dataset, "images", split_type)
        dst_lbl_dir = os.path.join(self.thu_muc_dataset, "labels", split_type)
        
        success_count = 0
        
        for img_file in file_list:
            try:
                lbl_file = img_file.replace('.jpg', '.txt')
                
                src_img = os.path.join(src_img_dir, img_file)
                src_lbl = os.path.join(src_lbl_dir, lbl_file)
                
                dst_img = os.path.join(dst_img_dir, img_file)
                dst_lbl = os.path.join(dst_lbl_dir, lbl_file)
                
                # Copy image
                if os.path.exists(src_img):
                    shutil.copy2(src_img, dst_img)
                else:
                    print(f"Missing image: {src_img}")
                    continue
                
                # Copy label
                if os.path.exists(src_lbl):
                    shutil.copy2(src_lbl, dst_lbl)
                else:
                    print(f"Missing label: {src_lbl}")
                    continue
                
                success_count += 1
                
            except Exception as e:
                print(f"Error copying {img_file}: {e}")
        
        print(f"Successfully copied {success_count}/{len(file_list)} files to {split_type}")
    
    def xoa_du_lieu_cu(self):
        """Xóa dữ liệu cũ trong train/val/test"""
        dirs = [
            os.path.join(self.thu_muc_dataset, "images", "train"),
            os.path.join(self.thu_muc_dataset, "images", "val"),
            os.path.join(self.thu_muc_dataset, "images", "test"),
            os.path.join(self.thu_muc_dataset, "labels", "train"),
            os.path.join(self.thu_muc_dataset, "labels", "val"),
            os.path.join(self.thu_muc_dataset, "labels", "test")
        ]
        
        for dir_path in dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
    
    def hien_thi_thong_ke_v2(self):
        """Hiển thị thống kê chi tiết"""
        print("\n" + "="*50)
        print("THONG KE DU LIEU CHI TIET")
        print("="*50)
        
        # Raw data stats
        raw_img_dir = os.path.join(self.thu_muc_raw, "images")
        if os.path.exists(raw_img_dir):
            raw_files = [f for f in os.listdir(raw_img_dir) if f.endswith('.jpg')]
            alert_raw = len([f for f in raw_files if f.startswith('alert_')])
            drowsy_raw = len([f for f in raw_files if f.startswith('drowsy_')])
            
            print(f"RAW DATA:")
            print(f"  Alert: {alert_raw}")
            print(f"  Drowsy: {drowsy_raw}")
            print(f"  Total: {len(raw_files)}")
            if drowsy_raw > 0:
                print(f"  Balance ratio: {alert_raw/drowsy_raw:.2f}")
        
        # Train/Val/Test stats
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.thu_muc_dataset, "images", split)
            if os.path.exists(split_dir):
                files = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
                alert_count = len([f for f in files if f.startswith('alert_')])
                drowsy_count = len([f for f in files if f.startswith('drowsy_')])
                
                print(f"\n{split.upper()}:")
                print(f"  Alert: {alert_count}")
                print(f"  Drowsy: {drowsy_count}")
                print(f"  Total: {len(files)}")
                if drowsy_count > 0:
                    print(f"  Balance: {alert_count/drowsy_count:.2f}")
        
        # EAR threshold info
        print(f"\nCONFIGURATION:")
        print(f"  EAR Threshold: {self.NGUONG_BUON_NGU}")
        print(f"  Image Size: {self.kich_thuoc_anh}x{self.kich_thuoc_anh}")
        print(f"  Augmentation: {'Enabled' if self.enable_augmentation else 'Disabled'}")
        print(f"  Classes: {', '.join(self.classes)}")
        
        print("\n" + "="*50)
    
    def bat_dau_thu_thap_realtime(self):
        """Bắt đầu thu thập dữ liệu real-time từ webcam"""
        print("\n=== BAT DAU THU THAP DU LIEU REAL-TIME ===")
        print("Phim:")
        print("  SPACE: Luu mau tu dong")
        print("  'a': Luu mau Alert (tinh tao)")
        print("  'd': Luu mau Drowsy (buon ngu)")
        print("  's': Hien thi thong ke")
        print("  'c': Xoa du lieu")
        print("  'p': Chia du lieu train/val/test")
        print("  'q': Thoat")
        print("  't': Dieu chinh nguong EAR")
        print("  'r': Reset history EAR")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Khong the mo camera!")
            return
        
        # Thiết lập camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Biến điều khiển
        auto_save = False
        frame_count = 0
        save_interval = 30  # Lưu mỗi 30 frames khi auto mode
        last_save_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Khong the doc frame tu camera!")
                break
            
            frame_count += 1
            
            # Flip frame để như gương
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Xử lý MediaPipe
            results = self.face_mesh.process(rgb_frame)
            
            current_ear = 0.0
            is_drowsy = False
            confidence = 0.0
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Tính EAR và phân loại
                    current_ear, is_drowsy, confidence = self.danh_gia_buon_ngu_v2(face_landmarks)
                    
                    # Chỉ tính toán bounding box mà không vẽ lên frame
                    bbox_result = self.phat_hien_khuon_mat_v2(frame, face_landmarks)
                    
                    # Auto save
                    if auto_save and frame_count % save_interval == 0:
                        current_time = cv2.getTickCount()
                        if (current_time - last_save_time) / cv2.getTickFrequency() > 1.0:  # Tối thiểu 1 giây
                            if confidence > 0.3:  # Chỉ lưu khi confidence cao
                                self.luu_mau_yolo_v2(frame, face_landmarks)
                                last_save_time = current_time
            
            # Hiển thị thông tin trên frame
            self.ve_thong_tin_len_frame(frame, current_ear, is_drowsy, confidence, auto_save)
            
            # Hiển thị frame
            cv2.imshow('Thu Thap Du Lieu YOLO - Face Drowsiness Detection', frame)
            
            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space - toggle auto save
                auto_save = not auto_save
                print(f"Auto save: {'ON' if auto_save else 'OFF'}")
            elif key == ord('a'):  # Save Alert
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        if self.luu_mau_yolo_v2(frame, face_landmarks, loai_mau=0):
                            print(f"Saved ALERT sample - Total: {self.so_mau_alert}")
            elif key == ord('d'):  # Save Drowsy
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        if self.luu_mau_yolo_v2(frame, face_landmarks, loai_mau=1):
                            print(f"Saved DROWSY sample - Total: {self.so_mau_drowsy}")
            elif key == ord('s'):  # Show stats
                self.hien_thi_thong_ke_v2()
            elif key == ord('c'):  # Clear data
                self.xoa_tat_ca_du_lieu()
            elif key == ord('p'):  # Split data
                self.chia_du_lieu_train_val_test()
            elif key == ord('t'):  # Adjust threshold
                self.dieu_chinh_nguong_ear()
            elif key == ord('r'):  # Reset EAR history
                self.history_ear = []
                print("Da reset lich su EAR")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Da ket thuc thu thap du lieu")
    
    def ve_thong_tin_len_frame(self, frame, ear, is_drowsy, confidence, auto_save):
        """Vẽ thông tin cơ bản lên frame - chỉ hiển thị text thông tin"""
        h, w = frame.shape[:2]
        
        # Background cho text - chỉ hiển thị thông tin cơ bản
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Thông tin cơ bản
        texts = [
            f"EAR: {ear:.4f}",
            f"Status: {'DROWSY' if is_drowsy else 'ALERT'}",
            f"Auto Save: {'ON' if auto_save else 'OFF'}",
            f"Samples: A:{self.so_mau_alert} D:{self.so_mau_drowsy}"
        ]
        
        for i, text in enumerate(texts):
            color = (0, 255, 255) if i == 1 and is_drowsy else (255, 255, 255)
            cv2.putText(frame, text, (15, 30 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    def dieu_chinh_nguong_ear(self):
        """Điều chỉnh ngưỡng EAR"""
        print(f"\nNguong EAR hien tai: {self.NGUONG_BUON_NGU}")
        try:
            new_threshold = float(input("Nhap nguong moi (0.1-0.4): "))
            if 0.1 <= new_threshold <= 0.4:
                self.NGUONG_BUON_NGU = new_threshold
                print(f"Da cap nhat nguong EAR: {self.NGUONG_BUON_NGU}")
                self.luu_metadata()
            else:
                print("Nguong phai trong khoang 0.1-0.4")
        except ValueError:
            print("Gia tri khong hop le!")
    
    def xoa_tat_ca_du_lieu(self):
        """Xóa tất cả dữ liệu"""
        confirm = input("Ban co chac chan muon xoa tat ca du lieu? (y/N): ")
        if confirm.lower() == 'y':
            try:
                # Xóa raw data
                raw_dirs = [
                    os.path.join(self.thu_muc_raw, "images"),
                    os.path.join(self.thu_muc_raw, "labels")
                ]
                
                for dir_path in raw_dirs:
                    if os.path.exists(dir_path):
                        for file in os.listdir(dir_path):
                            os.remove(os.path.join(dir_path, file))
                
                # Reset counters
                self.so_mau_alert = 0
                self.so_mau_drowsy = 0
                self.history_ear = []
                
                # Xóa split data
                self.xoa_du_lieu_cu()
                
                self.luu_metadata()
                print("Da xoa tat ca du lieu!")
                
            except Exception as e:
                print(f"Loi khi xoa du lieu: {e}")
        else:
            print("Huy xoa du lieu")
    
    def kiem_tra_chat_luong_dataset(self):
        """Kiểm tra chất lượng dataset"""
        print("\n=== KIEM TRA CHAT LUONG DATASET ===")
        
        raw_img_dir = os.path.join(self.thu_muc_raw, "images")
        raw_lbl_dir = os.path.join(self.thu_muc_raw, "labels")
        
        if not os.path.exists(raw_img_dir):
            print("Khong tim thay thu muc images!")
            return
        
        img_files = [f for f in os.listdir(raw_img_dir) if f.endswith('.jpg')]
        
        issues = []
        ear_values = {'alert': [], 'drowsy': []}
        
        for img_file in img_files:
            # Kiểm tra file label tương ứng
            lbl_file = img_file.replace('.jpg', '.txt')
            lbl_path = os.path.join(raw_lbl_dir, lbl_file)
            
            if not os.path.exists(lbl_path):
                issues.append(f"Missing label: {lbl_file}")
                continue
            
            # Đọc label
            try:
                with open(lbl_path, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            bbox = [float(x) for x in parts[1:]]
                            
                            # Kiểm tra bbox hợp lệ
                            if not all(0 <= x <= 1 for x in bbox):
                                issues.append(f"Invalid bbox in {lbl_file}: {bbox}")
                        else:
                            issues.append(f"Invalid label format in {lbl_file}")
                    else:
                        issues.append(f"Empty label file: {lbl_file}")
            except Exception as e:
                issues.append(f"Error reading {lbl_file}: {e}")
            
            # Trích xuất EAR từ tên file
            try:
                if '_ear' in img_file:
                    ear_str = img_file.split('_ear')[1].split('_')[0].replace('.jpg', '')
                    ear_val = float(ear_str)
                    
                    if img_file.startswith('alert_'):
                        ear_values['alert'].append(ear_val)
                    elif img_file.startswith('drowsy_'):
                        ear_values['drowsy'].append(ear_val)
            except:
                pass
        
        # Báo cáo
        print(f"Tong so file anh: {len(img_files)}")
        print(f"So loi phat hien: {len(issues)}")
        
        if issues:
            print("\nCAC LOI PHAT HIEN:")
            for issue in issues[:10]:  # Hiển thị 10 lỗi đầu
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... va {len(issues) - 10} loi khac")
        
        # Thống kê EAR
        if ear_values['alert'] or ear_values['drowsy']:
            print(f"\nTHONG KE EAR:")
            if ear_values['alert']:
                print(f"Alert EAR - Mean: {np.mean(ear_values['alert']):.4f}, Std: {np.std(ear_values['alert']):.4f}")
                print(f"           Min: {np.min(ear_values['alert']):.4f}, Max: {np.max(ear_values['alert']):.4f}")
            
            if ear_values['drowsy']:
                print(f"Drowsy EAR - Mean: {np.mean(ear_values['drowsy']):.4f}, Std: {np.std(ear_values['drowsy']):.4f}")
                print(f"            Min: {np.min(ear_values['drowsy']):.4f}, Max: {np.max(ear_values['drowsy']):.4f}")
            
            # Kiểm tra overlap
            if ear_values['alert'] and ear_values['drowsy']:
                alert_min = np.min(ear_values['alert'])
                drowsy_max = np.max(ear_values['drowsy'])
                
                if alert_min <= drowsy_max:
                    print(f"\nCANH BAO: Co overlap giua Alert min ({alert_min:.4f}) va Drowsy max ({drowsy_max:.4f})")
                    print(f"Nen dieu chinh nguong hoac loc du lieu")
        
        return len(issues) == 0

def main():
    """Hàm main để chạy chương trình"""
    print("=== CHUONG TRINH THU THAP DU LIEU YOLO CHO PHAT HIEN BUON NGU ===")
    print("Phien ban cai tien v2.0")
    
    collector = ThuThapDuLieuYOLO()
    
    while True:
        print("\n" + "="*50)
        print("MENU CHINH")
        print("="*50)
        print("1. Bat dau thu thap du lieu real-time")
        print("2. Hien thi thong ke chi tiet")
        print("3. Chia du lieu train/val/test")
        print("4. Kiem tra chat luong dataset")
        print("5. Dieu chinh nguong EAR")
        print("6. Xoa tat ca du lieu")
        print("7. Cau hinh data augmentation")
        print("0. Thoat")
        
        try:
            choice = input("\nChon chuc nang (0-7): ").strip()
            
            if choice == '0':
                print("Tam biet!")
                break
            elif choice == '1':
                collector.bat_dau_thu_thap_realtime()
            elif choice == '2':
                collector.hien_thi_thong_ke_v2()
            elif choice == '3':
                collector.chia_du_lieu_train_val_test()
            elif choice == '4':
                collector.kiem_tra_chat_luong_dataset()
            elif choice == '5':
                collector.dieu_chinh_nguong_ear()
            elif choice == '6':
                collector.xoa_tat_ca_du_lieu()
            elif choice == '7':
                current = "Enabled" if collector.enable_augmentation else "Disabled"
                print(f"Data augmentation hien tai: {current}")
                toggle = input("Ban co muon thay doi? (y/N): ")
                if toggle.lower() == 'y':
                    collector.enable_augmentation = not collector.enable_augmentation
                    new_status = "Enabled" if collector.enable_augmentation else "Disabled"
                    print(f"Da thay doi thanh: {new_status}")
            else:
                print("Lua chon khong hop le!")
                
        except KeyboardInterrupt:
            print("\nChuong trinh bi ngat boi nguoi dung")
            break
        except Exception as e:
            print(f"Loi: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
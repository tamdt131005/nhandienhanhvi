import os
import random
import shutil
import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm  # Thêm progress bar

# Cấu hình 
class_info = {
    'no_phone': {'id': 0, 'name': 'KHÔNG DÙNG ĐIỆN THOẠI'},
    'using_phone': {'id': 1, 'name': 'ĐANG DÙNG ĐIỆN THOẠI'}
}

# Thư mục dữ liệu
data_root = Path('data')
TRAIN_IMG_DIR = data_root / 'train' / 'images'
VAL_IMG_DIR = data_root / 'val' / 'images' 
TRAIN_LABEL_DIR = data_root / 'train' / 'labels'
VAL_LABEL_DIR = data_root / 'val' / 'labels'

# Cấu hình chia dữ liệu
SPLIT_RATIO = 0.8  # 80% train, 20% val
MIN_IMAGES_PER_CLASS = 100  # Số ảnh tối thiểu mỗi class
MAX_IMAGES_PER_CLASS = 3000  # Số ảnh tối đa mỗi class 
TARGET_SIZE = (640, 640)  # Kích thước ảnh cho YOLOv8
IMG_EXTS = ('.jpg', '.jpeg', '.png')

def ensure_dir(path):
    """Tạo thư mục nếu chưa tồn tại"""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Đã tạo thư mục: {path}")

def create_label(class_id, img_path, output_path):
    """
    Tạo file label YOLO format:
    <class_id> <center_x> <center_y> <width> <height>
    Các giá trị được normalize về [0,1]
    """
    try:
        # Đọc kích thước ảnh
        img = cv2.imread(str(img_path))
        if img is None:
            return False
            
        h, w = img.shape[:2]
        
        # Tạo bounding box ở giữa ảnh (chiếm 80% ảnh)
        box_w = w * 0.8
        box_h = h * 0.8
        center_x = w / 2
        center_y = h / 2
        
        # Normalize các giá trị về [0,1]
        norm_center_x = center_x / w
        norm_center_y = center_y / h
        norm_width = box_w / w
        norm_height = box_h / h
        
        # Ghi label theo format YOLO
        with open(output_path, 'w') as f:
            f.write(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        return True
    except Exception as e:
        print(f"⚠️ Lỗi tạo label cho {img_path}: {e}")
        return False

def process_image(src_path, dst_path):
    """Xử lý ảnh: resize về kích thước chuẩn"""
    try:
        img = cv2.imread(str(src_path))
        if img is None:
            return False
        
        # Resize giữ nguyên tỉ lệ
        h, w = img.shape[:2]
        scale = min(TARGET_SIZE[0]/w, TARGET_SIZE[1]/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h))
        
        # Tạo ảnh nền đen kích thước chuẩn
        canvas = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
        
        # Paste ảnh vào giữa
        x_offset = (TARGET_SIZE[0] - new_w) // 2
        y_offset = (TARGET_SIZE[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Lưu ảnh với chất lượng cao
        cv2.imwrite(str(dst_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 100])
        return True
    except Exception as e:
        print(f"⚠️ Lỗi xử lý ảnh {src_path}: {e}")
        return False

def split_dataset():
    """Chia dataset và tạo label cho YOLOv8"""
    print("\n=== CHIA DỮ LIỆU CHO YOLOV8 TRAINING ===\n")
    
    # Tạo thư mục
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR]:
        ensure_dir(d)
    
    # Thống kê và chia dữ liệu
    stats = {'train': {}, 'val': {}}
    
    for class_name, info in class_info.items():
        src_dir = data_root / class_name
        if not src_dir.exists():
            print(f"⚠️ Không tìm thấy thư mục {src_dir}")
            continue
            
        # Lấy danh sách ảnh
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(IMG_EXTS)]
        random.shuffle(files)  # Xáo trộn ngẫu nhiên
        
        # Giới hạn số lượng ảnh
        total = len(files)
        if total < MIN_IMAGES_PER_CLASS:
            print(f"⚠️ Class {info['name']} có quá ít ảnh ({total} < {MIN_IMAGES_PER_CLASS})")
            continue
        if total > MAX_IMAGES_PER_CLASS:
            print(f"ℹ️ Giới hạn số ảnh của {info['name']} xuống {MAX_IMAGES_PER_CLASS}")
            files = files[:MAX_IMAGES_PER_CLASS]
            total = MAX_IMAGES_PER_CLASS
            
        # Chia train/val
        n_train = int(total * SPLIT_RATIO)
        
        print(f"\n📊 Xử lý {info['name']}:")
        print(f"   - Tổng số ảnh: {total}")
        print(f"   - Train: {n_train}")
        print(f"   - Val: {total - n_train}")
        
        # Copy và xử lý ảnh, tạo label với progress bar
        failed = 0
        pbar = tqdm(files, desc=f"Xử lý {info['name']}")
        for i, fname in enumerate(pbar):
            src_path = src_dir / fname
            
            if i < n_train:
                # Train set
                img_dst = TRAIN_IMG_DIR / fname
                label_dst = TRAIN_LABEL_DIR / (Path(fname).stem + '.txt')
                is_train = True
            else:
                # Validation set
                img_dst = VAL_IMG_DIR / fname
                label_dst = VAL_LABEL_DIR / (Path(fname).stem + '.txt')
                is_train = False
                
            # Xử lý và copy ảnh
            if process_image(src_path, img_dst):
                # Tạo label
                if create_label(info['id'], img_dst, label_dst):
                    split = 'train' if is_train else 'val'
                    stats[split][class_name] = stats[split].get(class_name, 0) + 1
                else:
                    failed += 1
                    if os.path.exists(img_dst):
                        os.remove(img_dst)
            else:
                failed += 1
                
        if failed > 0:
            print(f"⚠️ Có {failed} ảnh bị lỗi trong quá trình xử lý")
    
    # In thống kê cuối cùng
    print("\n=== THỐNG KÊ SAU KHI CHIA ===")
    for split in ['train', 'val']:
        total = sum(stats[split].values())
        if total == 0:
            continue
        print(f"\n{split.upper()}:")
        for class_name, count in stats[split].items():
            print(f"   - {class_info[class_name]['name']}: {count} ({count/total*100:.1f}%)")
        print(f"   - Tổng: {total}")

if __name__ == '__main__':
    try:
        split_dataset()
        print("\n✅ Hoàn thành chia dữ liệu cho YOLOv8!")
        
        # Tạo file dataset.yaml
        yaml_content = {
            'path': str(data_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {0: 'no_phone', 1: 'using_phone'}
        }
        
        with open(data_root / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(yaml_content, f, sort_keys=False)
        print("✅ Đã tạo file dataset.yaml!")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
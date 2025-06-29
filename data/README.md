# Phone Usage Detection Dataset

## Thông tin Dataset
- **Tổng số ảnh**: 101
- **Classes**: 2 (no_phone, using_phone)
- **Train**: 80 ảnh
- **Validation**: 21 ảnh
- **Ngày tạo**: 2025-06-27 16:42:53

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

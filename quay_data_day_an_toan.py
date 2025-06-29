import cv2
import os
import datetime
import json
import numpy as np
from pathlib import Path
import time

class DataCollectionTool:
    def __init__(self, save_dir="seatbelt_dataset", camera_index=0):
        # Lưu trực tiếp cùng cấp file chạy
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.save_dir = os.path.join(current_dir, save_dir)
        
        self.camera_index = camera_index
        self.cap = None
        
        # Định nghĩa các nhãn
        self.labels = {
            '1': 'with_seatbelt',      # Có đeo dây an toàn
            '2': 'without_seatbelt',   # Không đeo dây an toàn
            '3': 'unclear'             # Không rõ ràng
        }
        
        # Màu sắc cho mỗi nhãn
        self.colors = {
            'with_seatbelt': (0, 255, 0),      # Xanh lá
            'without_seatbelt': (0, 0, 255),   # Đỏ
            'unclear': (0, 255, 255)           # Vàng
        }
        
        # Thống kê
        self.stats = {label: 0 for label in self.labels.values()}
        self.total_images = 0
        
        # Tạo thư mục lưu trữ
        self.setup_directories()
        
    def setup_directories(self):
        """Tạo các thư mục cần thiết"""
        try:
            # Tạo thư mục chính cùng cấp file chạy
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_dir = os.path.join(current_dir, "seatbelt_dataset")
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            
            # Tạo thư mục cho từng nhãn
            for label in self.labels.values():
                Path(os.path.join(self.save_dir, label)).mkdir(parents=True, exist_ok=True)
            
            print(f"Đã tạo thư mục lưu trữ: {self.save_dir}")
            print(f"Đường dẫn tuyệt đối: {os.path.abspath(self.save_dir)}")
            
        except Exception as e:
            print(f"Lỗi khi tạo thư mục: {e}")
            # Fallback: tạo trong thư mục hiện tại nếu không thể tạo ở ngoài
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_dir = os.path.join(current_dir, "seatbelt_dataset_local")
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
            
            for label in self.labels.values():
                Path(os.path.join(self.save_dir, label)).mkdir(parents=True, exist_ok=True)
            
            print(f"Đã tạo thư mục dự phòng: {self.save_dir}")
        
    def initialize_camera(self):
        """Khởi tạo camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"Không thể mở camera {self.camera_index}")
        
        # Cài đặt độ phân giải
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Camera đã được khởi tạo thành công!")
        
    def save_image_with_label(self, frame, label_key):
        """Lưu ảnh với nhãn (không lưu metadata)"""
        if label_key not in self.labels:
            return False
        
        label_name = self.labels[label_key]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        # Tên file
        filename = f"{label_name}_{timestamp}.jpg"
        filepath = os.path.join(self.save_dir, label_name, filename)
        
        # Đảm bảo thư mục tồn tại
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Lưu ảnh
        success = cv2.imwrite(filepath, frame)
        
        if success:
            # Cập nhật thống kê
            self.stats[label_name] += 1
            self.total_images += 1
            
            print(f"Đã lưu: {filename} - Nhãn: {label_name}")
            print(f"Đường dẫn: {filepath}")
            return True
        else:
            print(f"Lỗi khi lưu ảnh: {filepath}")
            return False
    
    def save_metadata(self, filename, label, timestamp):
        """Không lưu metadata nữa"""
        pass
    
    def draw_interface(self, frame):
        """Vẽ giao diện hướng dẫn"""
        height, width = frame.shape[:2]
        
        # Vẽ background cho text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 170), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Tiêu đề
        cv2.putText(frame, "CONG CU THU THAP DU LIEU", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Đường dẫn lưu trữ (rút gọn)
        save_path_short = "..." + self.save_dir[-30:] if len(self.save_dir) > 35 else self.save_dir
        cv2.putText(frame, f"Luu tai: {save_path_short}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Hướng dẫn
        instructions = [
            "Nhan phim:",
            "1 - Co deo day an toan",
            "2 - Khong deo day an toan", 
            "3 - Khong ro rang",
            "S - Luu anh hien tai",
            "R - Reset thong ke",
            "Q - Thoat"
        ]
        
        for i, instruction in enumerate(instructions):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            cv2.putText(frame, instruction, (20, 70 + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Thống kê
        stats_text = f"Tong: {self.total_images} | Co deo: {self.stats['with_seatbelt']} | Khong deo: {self.stats['without_seatbelt']} | Khong ro: {self.stats['unclear']}"
        cv2.putText(frame, stats_text, (20, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return frame
    
    def show_capture_feedback(self, frame, label_key):
        """Hiển thị feedback khi chụp ảnh"""
        if label_key in self.labels:
            label_name = self.labels[label_key]
            color = self.colors[label_name]
            
            # Vẽ border màu
            cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, frame.shape[0]-5), color, 5)
            
            # Text thông báo
            text = f"DA LUU: {label_name.upper()}"
            cv2.putText(frame, text, (frame.shape[1]//2 - 150, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        return frame
    
    def run(self):
        """Chạy ứng dụng thu thập dữ liệu"""
        try:
            self.initialize_camera()
            
            print("\n" + "="*70)
            print("CÔNG CỤ THU THẬP DỮ LIỆU DÂY AN TOÀN")
            print("="*70)
            print(f"Thư mục lưu trữ: {self.save_dir}")
            print("Hướng dẫn sử dụng:")
            print("- Nhấn phím '1': Lưu ảnh với nhãn 'Có đeo dây an toàn'")
            print("- Nhấn phím '2': Lưu ảnh với nhãn 'Không đeo dây an toàn'")
            print("- Nhấn phím '3': Lưu ảnh với nhãn 'Không rõ ràng'")
            print("- Nhấn phím 'S': Lưu ảnh hiện tại (cần chọn nhãn trước)")
            print("- Nhấn phím 'R': Reset thống kê")
            print("- Nhấn phím 'Q': Thoát chương trình")
            print("="*70)
            
            last_capture_time = 0
            capture_feedback_duration = 1.0  # 1 giây
            last_key = None
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Không thể đọc frame từ camera!")
                    break
                
                # Lật ảnh theo chiều ngang (như gương)
                frame = cv2.flip(frame, 1)
                
                # Vẽ giao diện
                display_frame = self.draw_interface(frame.copy())
                
                # Hiển thị feedback sau khi chụp
                current_time = time.time()
                if current_time - last_capture_time < capture_feedback_duration and last_key:
                    display_frame = self.show_capture_feedback(display_frame, last_key)
                
                # Hiển thị frame
                cv2.imshow('Data Collection Tool - Seatbelt Dataset', display_frame)
                
                # Xử lý phím bấm
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key in [ord('1'), ord('2'), ord('3')]:
                    success = self.save_image_with_label(frame, chr(key))
                    if success:
                        last_capture_time = current_time
                        last_key = chr(key)
                elif key == ord('s') or key == ord('S'):
                    # Lưu ảnh với nhãn được chọn gần nhất
                    if last_key and last_key in self.labels:
                        success = self.save_image_with_label(frame, last_key)
                        if success:
                            last_capture_time = current_time
                    else:
                        print("Vui lòng chọn nhãn (1, 2, hoặc 3) trước khi lưu!")
                elif key == ord('r') or key == ord('R'):
                    # Reset thống kê
                    self.stats = {label: 0 for label in self.labels.values()}
                    self.total_images = 0
                    print("Đã reset thống kê!")
        
        except KeyboardInterrupt:
            print("\nNgười dùng dừng chương trình!")
        except Exception as e:
            print(f"Lỗi: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Dọn dẹp tài nguyên"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # In thống kê cuối cùng
        print("\n" + "="*70)
        print("THỐNG KÊ THU THẬP DỮ LIỆU")
        print("="*70)
        print(f"Tổng số ảnh đã thu thập: {self.total_images}")
        for label, count in self.stats.items():
            print(f"- {label}: {count} ảnh")
        print(f"Dữ liệu được lưu tại: {os.path.abspath(self.save_dir)}")
        print("="*70)
    
    def export_dataset_info(self):
        """Xuất thông tin dataset"""
        dataset_info = {
            'total_images': self.total_images,
            'labels': self.labels,
            'statistics': self.stats,
            'created_at': datetime.datetime.now().isoformat(),
            'save_directory': os.path.abspath(self.save_dir)
        }
        
        info_file = os.path.join(self.save_dir, "dataset_dien_thoai_info.json")
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        print(f"Thông tin dataset đã được lưu tại: {info_file}")

# Sử dụng công cụ
if __name__ == "__main__":
    # Tạo instance với tên thư mục tùy chỉnh
    collector = DataCollectionTool(
        save_dir="seatbelt_dataset",  # Tên thư mục (sẽ được tạo ở ngoài thư mục code)
        camera_index=0                # Index của camera (0 = camera mặc định)
    )
    
    try:
        # Chạy công cụ thu thập
        collector.run()
        
        # Xuất thông tin dataset
        collector.export_dataset_info()
        
    except Exception as e:
        print(f"Lỗi khi chạy công cụ: {e}")
        input("Nhấn Enter để thoát...")
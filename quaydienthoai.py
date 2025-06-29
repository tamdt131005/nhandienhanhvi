import cv2
import os
import time
import numpy as np
from datetime import datetime
import json

class DataCollector:
    def __init__(self):
        """
        Thu thập dữ liệu từ webcam cho model nhận diện điện thoại
        """
        self.camera = None
        self.is_recording = False
        self.current_class = None
        self.image_count = {"no_phone": 0, "using_phone": 0}
        self.session_stats = {
            "start_time": None,
            "total_images": 0,
            "images_per_class": {"no_phone": 0, "using_phone": 0}
        }
        
        # Cấu hình
        self.config = {
            "image_size": (640, 480),
            "save_size": (224, 224),
            "fps": 30,
            "quality": 95,
            "auto_capture_interval": 0.5  # giây
        }
        
        self.setup_directories()
        self.load_existing_counts()
    
    def setup_directories(self):
        """
        Tạo cấu trúc thư mục chỉ gồm data/no_phone và data/using_phone
        """
        directories = [
            "data/no_phone",
            "data/using_phone"
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("✅ Đã tạo cấu trúc thư mục:")
        for directory in directories:
            print(f"   📁 {directory}")

    def load_existing_counts(self):
        """
        Đếm số ảnh hiện có trong data/no_phone và data/using_phone
        """
        for class_name in ["no_phone", "using_phone"]:
            path = f"data/{class_name}"
            if os.path.exists(path):
                count = len([f for f in os.listdir(path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            else:
                count = 0
            self.image_count[class_name] = count
        print(f"\n📊 Số ảnh hiện có:")
        print(f"   🚫 Không dùng điện thoại: {self.image_count['no_phone']}")
        print(f"   📱 Đang dùng điện thoại: {self.image_count['using_phone']}")

    def initialize_camera(self, camera_id=0):
        """
        Khởi tạo camera
        """
        try:
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["image_size"][0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["image_size"][1])
            self.camera.set(cv2.CAP_PROP_FPS, self.config["fps"])
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Không thể đọc từ camera")
                
            print(f"✅ Camera đã sẵn sàng (ID: {camera_id})")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo camera: {e}")
            return False
    
    def save_image(self, frame, class_name):
        """
        Lưu ảnh với timestamp vào data/no_phone hoặc data/using_phone
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{class_name}_{timestamp}_{self.image_count[class_name]:06d}.jpg"
        filepath = f"data/{class_name}/{filename}"
        
        # Resize ảnh cho training
        frame = cv2.resize(frame, self.config["save_size"])
        
        # Lưu ảnh
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, self.config["quality"]])
        
        self.image_count[class_name] += 1
        self.session_stats["total_images"] += 1
        self.session_stats["images_per_class"][class_name] += 1
        
        return filepath
    
    def draw_interface(self, frame):
        """
        Vẽ giao diện trên frame
        """
        height, width = frame.shape[:2]
        
        # Background cho thông tin
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width-10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Tiêu đề
        cv2.putText(frame, "THU THAP DU LIEU NHAN DIEN DIEN THOAI", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Hướng dẫn
        instructions = [
            "Phim TAT: [1] Dang dung dien thoai  [2] Khong dung dien thoai",
            "[SPACE] Chup anh  [A] Auto chup  [S] Dung auto  [Q] Thoat",
            f"So anh: Khong DT: {self.image_count['no_phone']} | Dung DT: {self.image_count['using_phone']}"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (20, 60 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Status hiện tại
        if self.current_class:
            class_text = "DANG DUNG DIEN THOAI" if self.current_class == "using_phone" else "KHONG DUNG DIEN THOAI"
            color = (0, 0, 255) if self.current_class == "using_phone" else (0, 255, 0)
            
            cv2.rectangle(frame, (10, height-80), (width-10, height-10), color, 3)
            cv2.putText(frame, f"DANG CHUP: {class_text}", 
                       (20, height-45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if self.is_recording:
                cv2.circle(frame, (width-50, 30), 15, (0, 0, 255), -1)
                cv2.putText(frame, "AUTO", (width-70, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def auto_distribute_images(self):
        """
        Không còn chức năng phân chia ảnh, chỉ hiện thông báo
        """
    
    def start_collection(self, camera_id=0):
        """
        Bắt đầu thu thập dữ liệu
        """
        if not self.initialize_camera(camera_id):
            return
        
        self.session_stats["start_time"] = datetime.now()
        last_auto_capture = 0
        
        print("\n🎥 Bắt đầu thu thập dữ liệu...")
        print("📋 Hướng dẫn:")
        print("   [1] - Chế độ 'Đang dùng điện thoại'")
        print("   [2] - Chế độ 'Không dùng điện thoại'") 
        print("   [SPACE] - Chụp ảnh thủ công")
        print("   [A] - Bật chế độ tự động chụp")
        print("   [S] - Tắt chế độ tự động chụp")
        print("   [D] - Phân chia ảnh vào train/val/test")
        print("   [Q] - Thoát\n")
        
        try:
            while True:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Không thể đọc frame từ camera")
                    break
                
                # Flip frame để dễ nhìn
                frame = cv2.flip(frame, 1)
                
                # Vẽ giao diện
                display_frame = self.draw_interface(frame.copy())
                
                # Auto capture
                current_time = time.time()
                if (self.is_recording and self.current_class and 
                    current_time - last_auto_capture >= self.config["auto_capture_interval"]):
                    
                    filepath = self.save_image(frame, self.current_class)
                    print(f"📸 Auto: {os.path.basename(filepath)}")
                    last_auto_capture = current_time
                    
                    # Flash effect
                    flash_frame = np.ones_like(display_frame) * 255
                    cv2.imshow('Data Collection', flash_frame)
                    cv2.waitKey(50)
                
                cv2.imshow('Data Collection', display_frame)
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('1'):
                    self.current_class = "using_phone"
                    print("📱 Chế độ: ĐANG DÙNG ĐIỆN THOẠI")
                elif key == ord('2'):
                    self.current_class = "no_phone"
                    print("🚫 Chế độ: KHÔNG DÙNG ĐIỆN THOẠI")
                elif key == ord(' ') and self.current_class:
                    filepath = self.save_image(frame, self.current_class)
                    print(f"📸 Manual: {os.path.basename(filepath)}")
                elif key == ord('a') and self.current_class:
                    self.is_recording = True
                    print("🔴 Bắt đầu tự động chụp")
                elif key == ord('s'):
                    self.is_recording = False
                    print("⏹️ Dừng tự động chụp")
                elif key == ord('d'):
                    print("📊 Đang phân chia dữ liệu...")
                    self.auto_distribute_images()
                    print("✅ Hoàn thành phân chia!")
                    
        except KeyboardInterrupt:
            print("\n⏹️ Dừng thu thập dữ liệu")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Dọn dẹp tài nguyên
        """
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Thống kê session
        if self.session_stats["start_time"]:
            duration = datetime.now() - self.session_stats["start_time"]
            print(f"\n📊 THỐNG KÊ PHIÊN LÀM VIỆC:")
            print(f"   ⏱️ Thời gian: {duration}")
            print(f"   📸 Tổng ảnh chụp: {self.session_stats['total_images']}")
            print(f"   🚫 Không dùng ĐT: {self.session_stats['images_per_class']['no_phone']}")
            print(f"   📱 Đang dùng ĐT: {self.session_stats['images_per_class']['using_phone']}")
            
        # Lưu metadata
        self.save_session_metadata()
        print("✅ Đã lưu metadata phiên làm việc")
    
    def save_session_metadata(self):
        """
        Lưu thông tin metadata
        """
        metadata = {
            "session_time": datetime.now().isoformat(),
            "total_images_collected": self.session_stats["total_images"],
            "images_per_class": self.session_stats["images_per_class"],
            "current_total_images": self.image_count,
            "config": self.config
        }
        
        metadata_file = f"data/session_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

def main():
    """
    Chương trình chính
    """
    print("=== THU THẬP DỮ LIỆU NHẬN DIỆN ĐIỆN THOẠI ===")
    collector = DataCollector()
    collector.start_collection()

if __name__ == "__main__":
    main()
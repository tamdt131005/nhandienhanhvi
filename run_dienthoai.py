import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

class PhoneDetector:
    def __init__(self, model_path='runs/detect/phone_detection4/weights/best.pt', confidence=0.7):
        """
        Khởi tạo Phone Detector
        Args:
            model_path: Đường dẫn đến model đã train
            confidence: Ngưỡng confidence để hiển thị detection
        """
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.cap = None
        
        # Class names
        self.class_names = {0: 'no_phone', 1: 'using_phone'}
        
        # Colors cho từng class (BGR format)
        self.colors = {
            0: (0, 255, 0),    # Green cho no_phone
            1: (0, 0, 255),    # Red cho using_phone
        }
        
        # Thống kê
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        self.phone_detected_count = 0  # Đếm số frame liên tiếp phát hiện dùng điện thoại
        self.phone_detected_min_frames = 4  # Số frame liên tiếp tối thiểu để báo cáo phát hiện điện thoại
        
    def load_model(self):
        """Load YOLO model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Không tìm thấy model tại: {self.model_path}")
            print("💡 Hãy chạy training script trước để tạo model")
            return False
            
        try:
            print(f"📦 Đang load model từ: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("✅ Model loaded thành công!")
            return True
        except Exception as e:
            print(f"❌ Lỗi load model: {str(e)}")
            return False
    
    def init_webcam(self, camera_id=0):
        """Khởi tạo webcam"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                print(f"❌ Không thể mở camera {camera_id}")
                return False
                
            # Cấu hình webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("📷 Webcam khởi tạo thành công!")
            return True
            
        except Exception as e:
            print(f"❌ Lỗi khởi tạo webcam: {str(e)}")
            return False
    
    def draw_detections(self, frame, results, phone_frame_count=0, phone_detection_threshold=0):
        """Vẽ detection boxes, labels và cảnh báo lên frame, bao gồm cả bộ đếm điện thoại"""
        warning_detected = False
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Lấy thông tin box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Chỉ hiển thị nếu confidence >= threshold
                if confidence >= self.confidence:
                    # Lấy tên class và màu
                    class_name = self.class_names.get(class_id, f"Class_{class_id}")
                    color = self.colors.get(class_id, (255, 255, 255))
                    
                    # Vẽ bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Tạo label
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Tính kích thước text
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # Vẽ background cho text
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        color, -1
                    )
                    
                    # Vẽ text
                    cv2.putText(
                        frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )
                    
                    # Nếu phát hiện using_phone thì bật cảnh báo
                    if class_id == 1:
                        warning_detected = True
        
        # Hiển thị cảnh báo nếu phát hiện using_phone
        if warning_detected:
            warning_text = "DANG DUNG DIEN THOAI"
            # Draw the warning text in red
            cv2.putText(frame, warning_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Hiển thị mức độ confidence lớn (lấy confidence từ box đầu tiên nếu có, hoặc 0)
            display_conf = results[0].boxes[0].conf[0].cpu().numpy() if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0 else 0.0
            cv2.putText(frame, f"MUC DO: {display_conf*100:.0f}%", (frame.shape[1]-250, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)

        # Vẽ bộ đếm frame điện thoại
        if phone_detection_threshold > 0: # Chỉ vẽ nếu threshold hợp lệ
             # Adjusted position to avoid overlap with the new warning text
             cv2.putText(frame, f"Phone Counter: {phone_frame_count}/{phone_detection_threshold}",
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame
    
    def draw_info(self, frame):
        """Vẽ thông tin FPS và instructions"""
        height, width = frame.shape[:2]
        
        # Tính FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # Vẽ FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ confidence threshold
        conf_text = f"Confidence: {self.confidence:.2f}"
        cv2.putText(frame, conf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Vẽ instructions
        instructions = [
            "ESC: Thoat",
            "UP/DOWN: Tang/giam confidence", 
            "SPACE: Chup anh"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, height - 80 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def save_screenshot(self, frame):
        """Lưu screenshot"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Đã lưu screenshot: {filename}")
    
    def run(self):
        """Chạy detection loop chính"""
        print("🎯 YOLO8 Phone Detection - Webcam")
        print("=" * 50)
        
        # Load model
        if not self.load_model():
            return
        
        # Khởi tạo webcam
        if not self.init_webcam():
            return
        try:
            while True:
                # Đọc frame từ webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Không thể đọc frame từ webcam")
                    break
                
                # Flip frame (như nhìn gương)
                frame = cv2.flip(frame, 1)
                
                # Chạy detection
                results = self.model(frame, conf=self.confidence, verbose=False)
                
                # Vẽ detections
                frame = self.draw_detections(frame, results)
                
                # Vẽ thông tin
                frame = self.draw_info(frame)
                
                # Hiển thị frame
                cv2.imshow('Phone Detection', frame)
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q') or key == ord('Q'):  # ESC hoặc Q
                    break
                elif key == ord(' '):  # SPACE - chụp ảnh
                    self.save_screenshot(frame)
                elif key == ord('r') or key == ord('R'):  # Reset FPS
                    self.frame_count = 0
                    self.start_time = time.time()
                    self.fps = 0
        except KeyboardInterrupt:
            print("\n⚠️  Dừng bởi người dùng")
        except Exception as e:
            print(f"❌ Lỗi trong quá trình detection: {str(e)}")
        finally:
            # Cleanup
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Đã dọn dẹp resources")
    
    def nhan_dien_dien_thoai(self, frame):
        """Nhận diện sử dụng điện thoại trên frame, trả về (có_điện_thoại, độ_tin_cậy, raw_results, frame_count)"""
        if self.model is None:
            self.load_model()
        try:
            results = self.model(frame, conf=self.confidence, verbose=False)
            co_dien_thoai_raw = False # Kết quả phát hiện thô (chưa qua ngưỡng frame)
            conf = 0.0
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    # Chỉ xét class 'using_phone' (class_id = 1)
                    if class_id == 1 and confidence >= self.confidence:
                         co_dien_thoai_raw = True
                         if confidence > conf:
                            conf = confidence

            # Cập nhật bộ đếm frame liên tiếp
            if co_dien_thoai_raw:
                self.phone_detected_count += 1
            else:
                self.phone_detected_count = 0 # Reset nếu không phát hiện

            # Kết quả cuối cùng sau khi áp dụng ngưỡng frame
            co_dien_thoai_final = self.phone_detected_count >= self.phone_detected_min_frames

            return co_dien_thoai_final, conf, results, self.phone_detected_count
        except Exception as e:
            print(f"Lỗi nhận diện điện thoại: {e}")
            return False, 0.0, None, 0

def main():
    # Cấu hình
    model_path = 'runs/detect/phone_detection4/weights/best.pt'
    confidence_threshold = 0.7
    camera_id = 0  # 0 = webcam mặc định, 1 = camera thứ 2
    
    # Kiểm tra model tồn tại
    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model!")
        print("💡 Các model có thể sử dụng:")
        
        # Tìm các model khác
        possible_paths = [
            'runs/detect/phone_detection3/weights/epoch90.pt',
            'runs/detect/phone_detection2/best.pt'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"   ✅ {path}")
                model_path = path
                break
        else:
            print("   ❌ Không tìm thấy model nào!")
            print("   🔧 Hãy chạy training script trước")
            return
    
    # Tạo và chạy detector
    detector = PhoneDetector(model_path=model_path, confidence=confidence_threshold)
    detector.run()

if __name__ == "__main__":
    main()
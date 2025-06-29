import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import threading
import queue
from collections import defaultdict
import pygame

class WebcamNhanDienDayAnToan:
    def __init__(self, duong_dan_model="modelantoan/best.pt"):
        """
        Khởi tạo webcam nhận diện dây an toàn
        
        Args:
            duong_dan_model: Đường dẫn tới model đã huấn luyện
        """
        self.duong_dan_model = duong_dan_model
        self.model = None
        self.cap = None
        self.dang_chay = False
        
        # Cấu hình hiển thị - chỉ 2 loại
        self.ten_cac_lop = ['Có dây an toàn', 'Không dây an toàn']
        self.mau_sac_lop = {
            0: (0, 255, 0),    # Xanh lá - có dây an toàn
            1: (0, 0, 255),    # Đỏ - không có dây an toàn  
        }
        
        # Trạng thái hiện tại - chỉ 3 trạng thái đơn giản
        self.trang_thai_hien_tai = "Không phát hiện"
        self.mau_trang_thai = (128, 128, 128)  # Xám - không phát hiện
        self.thoi_gian_cap_nhat_trang_thai = time.time()
        self.do_tin_cay_cao_nhat = 0.0
        
        # Cảnh báo âm thanh - chỉ khi không đeo dây an toàn
        self.am_thanh_canh_bao = True
        self.lan_canh_bao_cuoi = 0
        self.khoang_cach_canh_bao = 2  # 2 giây giữa các lần cảnh báo
        
        # FPS tracking đơn giản
        self.fps_hien_tai = 0
        self.thoi_gian_frame_truoc = time.time()
        
        # Khởi tạo pygame cho âm thanh
        try:
            pygame.mixer.init()
            self.co_am_thanh = True
        except:
            self.co_am_thanh = False
            print("⚠️ Không thể khởi tạo âm thanh")
    
    def safe_print(self, *args, **kwargs):
        try:
            print(*args, **kwargs)
        except Exception:
            pass
    
    def khoi_tao_model(self):
        """Khởi tạo model YOLO"""
        try:
            if not Path(self.duong_dan_model).exists():
                self.safe_print(f"❌ Không tìm thấy model tại: {self.duong_dan_model}")
                return False
                
            self.safe_print(f"🔧 Đang tải model từ {self.duong_dan_model}...")
            self.model = YOLO(self.duong_dan_model)
            self.safe_print("✅ Model đã được tải thành công!")
            
            # Warmup model
            self.safe_print("🔥 Đang warmup model...")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_img, verbose=False)
            self.safe_print("✅ Warmup hoàn thành!")
            
            return True
            
        except Exception as e:
            self.safe_print(f"❌ Lỗi khi tải model: {e}")
            return False
    
    def khoi_tao_camera(self, camera_id=0):
        """Khởi tạo camera"""
        try:
            self.safe_print(f"📷 Đang khởi tạo camera {camera_id}...")
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                self.safe_print(f"❌ Không thể mở camera {camera_id}")
                return False
            
            # Cấu hình camera đơn giản
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.safe_print(f"✅ Camera khởi tạo thành công!")
            
            return True
            
        except Exception as e:
            self.safe_print(f"❌ Lỗi khởi tạo camera: {e}")
            return False
    
    def tinh_fps(self):
        """Tính FPS đơn giản"""
        current_time = time.time()
        time_diff = current_time - self.thoi_gian_frame_truoc
        if time_diff > 0:
            self.fps_hien_tai = 1.0 / time_diff
        self.thoi_gian_frame_truoc = current_time
    
    def phat_am_thanh_canh_bao(self):
        """Đã loại bỏ phát âm thanh ở đây, xử lý ở giao diện."""
        pass

    def cap_nhat_trang_thai_tong_quan(self, co_phat_hien, lop_tin_cay_nhat, confidence_cao_nhat):
        """Chỉ 2 trạng thái: có hoặc không có dây an toàn"""
        if lop_tin_cay_nhat == 0:  # Có dây an toàn
            self.trang_thai_hien_tai = "✅ ĐANG ĐEO DÂY AN TOÀN"
            self.mau_trang_thai = (0, 255, 0)  # Xanh lá
        else:  # Không có dây an toàn hoặc không phát hiện gì
            self.trang_thai_hien_tai = "❌ KHÔNG ĐEO DÂY AN TOÀN"
            self.mau_trang_thai = (0, 0, 255)  # Đỏ
        self.do_tin_cay_cao_nhat = confidence_cao_nhat
        self.thoi_gian_cap_nhat_trang_thai = time.time()
    
    def ve_ket_qua(self, frame, ket_qua):
        """Vẽ kết quả nhận diện lên frame (bỏ vẽ bounding box, không cập nhật thống kê số lần có/không dây an toàn)"""
        frame_hien_thi = frame.copy()
        co_khong_day_an_toan = False
        co_phat_hien = False
        confidence_cao_nhat = 0.0
        lop_tin_cay_nhat = None
        
        if ket_qua and len(ket_qua) > 0:
            for r in ket_qua:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        if conf < 0.:
                            continue
                        co_phat_hien = True
                        if conf > confidence_cao_nhat:
                            confidence_cao_nhat = conf
                            lop_tin_cay_nhat = cls
                        if cls == 1:
                            co_khong_day_an_toan = True
        
        # Cập nhật trạng thái tổng quan dựa trên detection có confidence cao nhất
        self.cap_nhat_trang_thai_tong_quan(co_phat_hien, lop_tin_cay_nhat, confidence_cao_nhat)
        
        # Không phát cảnh báo âm thanh ở đây nữa
        
        return frame_hien_thi, co_khong_day_an_toan
    
    def ve_thong_tin_overlay(self, frame, co_canh_bao=False):
        """Vẽ thông tin overlay đơn giản (bỏ thống kê nhật kí)"""
        height, width = frame.shape[:2]
        
        # === TRẠNG THÁI TỔNG QUAN (TO, NỔI BẬT) ===
        status_text = self.trang_thai_hien_tai
        confidence_text = f"({self.do_tin_cay_cao_nhat:.2f})" if self.do_tin_cay_cao_nhat > 0 else ""
        full_status = f"{status_text} {confidence_text}"
        
        # Tính kích thước text để căn giữa
        font_scale = 1.0
        thickness = 2
        (status_width, status_height), _ = cv2.getTextSize(
            full_status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Vị trí căn giữa phía trên
        status_x = (width - status_width) // 2
        status_y = 40
        
        # Vẽ nền cho trạng thái
        padding = 10
        cv2.rectangle(frame, 
                     (status_x - padding, status_y - status_height - padding),
                     (status_x + status_width + padding, status_y + padding),
                     (0, 0, 0), -1)
        
        # Vẽ viền màu theo trạng thái
        cv2.rectangle(frame,
                     (status_x - padding, status_y - status_height - padding),
                     (status_x + status_width + padding, status_y + padding),
                     self.mau_trang_thai, 2)
        
        # Vẽ text trạng thái
        cv2.putText(frame, full_status, (status_x, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.mau_trang_thai, thickness)
        
        # === CẢNH BÁO ĐƠN GIẢN ===
        if co_canh_bao:
            canh_bao_text = "CANH BAO: KHONG DEO DAY AN TOAN!"
            canh_bao_font = 0.7
            canh_bao_thickness = 2
            
            (canh_bao_width, canh_bao_height), _ = cv2.getTextSize(
                canh_bao_text, cv2.FONT_HERSHEY_SIMPLEX, canh_bao_font, canh_bao_thickness
            )
            
            canh_bao_x = (width - canh_bao_width) // 2
            canh_bao_y = height - 80
            
            # Nền đỏ nhấp nháy
            if int(time.time() * 3) % 2:  # Nhấp nháy nhanh hơn
                cv2.rectangle(frame, 
                             (canh_bao_x - 5, canh_bao_y - canh_bao_height - 5),
                             (canh_bao_x + canh_bao_width + 5, canh_bao_y + 5),
                             (0, 0, 255), -1)
                text_color = (255, 255, 255)
            else:
                text_color = (0, 0, 255)
            
            cv2.putText(frame, canh_bao_text, (canh_bao_x, canh_bao_y),
                       cv2.FONT_HERSHEY_SIMPLEX, canh_bao_font, text_color, canh_bao_thickness)
        
        # === HƯỚNG DẪN ĐƠN GIẢN (GÓC PHẢI) ===
        huong_dan = [
            "Q: Thoat",
            "S: Luu anh", 
            "R: Reset",
        ]
        
        for i, text in enumerate(huong_dan):
            cv2.putText(frame, text, (width - 120, height - 70 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def luu_anh_screenshot(self, frame):
        """Lưu ảnh screenshot"""
        try:
            thu_muc_screenshot = Path("screenshots")
            thu_muc_screenshot.mkdir(exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            ten_file = f"seatbelt_detection_{timestamp}.jpg"
            duong_dan = thu_muc_screenshot / ten_file
            
            cv2.imwrite(str(duong_dan), frame)
            self.safe_print(f"📸 Đã lưu ảnh: {duong_dan}")
        except Exception as e:
            self.safe_print(f"❌ Lỗi lưu ảnh: {e}")
    
    def chay_nhan_dien(self, camera_id=0):
        """Chạy nhận diện realtime"""
        
        # Khởi tạo model và camera
        if not self.khoi_tao_model():
            return False
            
        if not self.khoi_tao_camera(camera_id):
            return False
        
        self.safe_print("\n🚀 Bắt đầu nhận diện realtime...")
        self.safe_print("📋 Hướng dẫn điều khiển:")
        self.safe_print("   Q: Thoát")
        self.safe_print("   S: Lưu ảnh screenshot")
        self.safe_print("   R: Reset thống kê")
        self.safe_print("\n✨ Nhấn Q để dừng...\n")
        
        self.dang_chay = True
        
        try:
            while self.dang_chay:
                ret, frame = self.cap.read()
                if not ret:
                    self.safe_print("❌ Không thể đọc frame từ camera")
                    break
                
                # Tính FPS
                self.tinh_fps()
                
                # Nhận diện
                ket_qua = self.model(frame, verbose=False)
                
                # Vẽ kết quả
                frame_hien_thi, co_canh_bao = self.ve_ket_qua(frame, ket_qua)
                
                # Vẽ thông tin overlay
                frame_cuoi_cung = self.ve_thong_tin_overlay(frame_hien_thi, co_canh_bao)
                
                # Hiển thị
                cv2.imshow('Nhan Dien Day An Toan', frame_cuoi_cung)
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Q
                    break
                elif key == ord('s'):  # S - Screenshot
                    self.luu_anh_screenshot(frame_cuoi_cung)
                elif key == ord('r'):  # R - Reset thống kê
                    self.so_lan_co_day = 0
                    self.so_lan_khong_day = 0
                    self.trang_thai_hien_tai = "Không phát hiện"
                    self.mau_trang_thai = (128, 128, 128)
                    self.do_tin_cay_cao_nhat = 0.0
                    self.safe_print("🔄 Đã reset thống kê")
        except KeyboardInterrupt:
            self.safe_print("\n⚠️ Đã dừng bằng Ctrl+C")
        except Exception as e:
            self.safe_print(f"❌ Lỗi trong quá trình chạy: {e}")
        finally:
            self.dong_ung_dung()
    
    def dong_ung_dung(self):
        """Đóng ứng dụng và giải phóng tài nguyên (bỏ in thống kê nhật kí)"""
        self.dang_chay = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        self.safe_print("\n👋 Đã đóng ứng dụng!")
    
    def nhan_dien_day_an_toan(self, frame):
        """Nhận diện dây an toàn trên frame, trả về (có_dây, độ_tin_cậy)"""
        if self.model is None:
            self.khoi_tao_model()
        try:
            results = self.model(frame, verbose=False)
            co_day = True
            conf = 0.0
            if results and len(results) > 0:
                for r in results:
                    boxes = r.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf_box = float(box.conf[0])
                            cls = int(box.cls[0])
                            if conf_box > conf:
                                conf = conf_box
                                co_day = (cls == 0)  # 0: có dây, 1: không dây
            return co_day, conf
        except Exception as e:
            self.safe_print(f"Lỗi nhận diện dây an toàn: {e}")
            return True, 0.0

def main():
    try:
        print("=== WEBCAM NHẬN DIỆN DÂY AN TOÀN ===")
        print("Sử dụng YOLOv8n đã huấn luyện\n")
    except Exception:
        pass
    duong_dan_model = "modelantoan/best.pt"
    if not Path(duong_dan_model).exists():
        try:
            print(f"❌ Không tìm thấy model tại: {duong_dan_model}")
            print("💡 Vui lòng chạy huấn luyện trước hoặc kiểm tra đường dẫn model")
        except Exception:
            pass
        return
    detector = WebcamNhanDienDayAnToan(duong_dan_model)
    camera_id = 0
    detector.chay_nhan_dien(camera_id)

if __name__ == "__main__":
    main()
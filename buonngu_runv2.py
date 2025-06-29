import cv2
import numpy as np
from collections import deque
import time
import mediapipe as mp
import math
import os
from ultralytics import YOLO


class PhatHienBuonNgu:
    def __init__(self, yolo_model_path=None):
        # Khoi tao MediaPipe Face Mesh với cấu hình tối ưu
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.7,  # Tăng từ 0.5
            min_tracking_confidence=0.7    # Tăng từ 0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Dinh nghia chi so cho cac diem landmarks cua mat - mở rộng để chính xác hơn
        self.CHI_SO_MAT_TRAI = [263, 387, 385, 362, 380, 373, 374, 381, 382]  # Thêm điểm
        self.CHI_SO_MAT_PHAI = [33, 160, 158, 133, 144, 153, 154, 155, 145]   # Thêm điểm
        
        # === CẢI THIỆN THUẬT TOÁN ML ===
        # Khoi tao bien dem chop mat va thoi gian - cải thiện độ nhạy
        self.dem_mat_nham = 0
        self.thoi_gian_mat_nham = 0
        self.thoi_gian_canh_bao_cuoi = 0
        self.dem_mat_mo_lien_tuc = 0
        self.thoi_gian_tinh_tao = 0
        
        # Cai dat nguong canh bao - ĐIỀU CHỈNH ĐỂ NHẠY HỚN
        self.NGUONG_MAT_NHAM_FRAME = 5        # Giảm từ 8 xuống 5
        self.NGUONG_THOI_GIAN_BUON_NGU = 1.2  # Giảm từ 2.0 xuống 1.2 giây
        self.NGUONG_EAR = 0.30                # Tăng từ 0.27 lên 0.30
        self.NGUONG_EAR_NGHIEM_NGAT = 0.25    # Tăng từ 0.23 lên 0.25
        self.NGUONG_EAR_MO = 0.45             # Giảm từ 0.50 xuống 0.45
        self.SO_FRAME_XAC_NHAN_TINH_TAO = 8   # Giảm từ 10 xuống 8
        self.THOI_GIAN_CHO_GIUA_CANH_BAO = 2.0 # Giảm từ 3.0 xuống 2.0
        self.SO_LAN_CANH_BAO_TICH_LUY = 0
        
        # === THÊM CÁC CHỈ SỐ ML MỚI ===
        self.NGUONG_PERCLOS = 0.15      # Percentage of Eye Closure
        self.NGUONG_BLINK_DURATION = 0.5 # Thời gian chớp mắt bình thường
        self.NGUONG_SLOW_BLINK = 0.8    # Thời gian chớp mắt chậm
        self.NGUONG_HEAD_NOD = 15       # Góc gật đầu (pitch)
        self.NGUONG_HEAD_TILT = 20      # Góc nghiêng đầu (roll)
        
        # Buffer cho việc tính toán các chỉ số mới
        self.bo_dem_ear_chi_tiet = deque(maxlen=30)      # 1 giây ở 30fps
        self.bo_dem_blink_durations = deque(maxlen=10)
        self.bo_dem_head_pose = deque(maxlen=15)
        self.lich_su_trang_thai_mat = deque(maxlen=60)   # 2 giây lịch sử
        
        # Biến theo dõi trạng thái chớp mắt
        self.dang_chop_mat = False
        self.thoi_gian_bat_dau_chop = 0
        self.so_lan_chop_mat = 0
        self.thoi_gian_cuoi_chop = 0
        
        # === CÁC THAM SỐ FUSION ML+DL - TĂNG TRỌNG SỐ ML ===
        self.TRONG_SO_ML = 0.8  # Tăng từ 0.7 lên 0.8
        self.TRONG_SO_DL = 0.2  # Giảm từ 0.3 xuống 0.2
        self.NGUONG_FUSION_BUON_NGU = 0.45  # Giảm từ 0.53 xuống 0.45
        self.SO_FRAME_FUSION_XAC_NHAN = 6   # Giảm từ 8 xuống 6
        
        # Buffer lưu trữ kết quả fusion
        self.bo_dem_ket_qua_fusion = deque(maxlen=10)
        self.bo_dem_yolo_results = deque(maxlen=5)
        self.bo_dem_yolo_drowsy = deque(maxlen=10)
        self.dem_frame_buon_ngu_fusion = 0
        self.dem_frame_yolo_drowsy = 0
        
        # Khoi tao buffer luu tru cac dac trung theo chuoi thoi gian
        self.SO_BUOC_THOI_GIAN = 20
        self.bo_dem_dac_trung = deque(maxlen=self.SO_BUOC_THOI_GIAN)
        self.bo_dem_ear = deque(maxlen=20)  # Tăng từ 15 lên 20
        
        # Khởi tạo YOLOv8n model detection (DL)
        self.yolo_model = None
        self.yolo_model_path = yolo_model_path or 'savemodel/drowsiness_detection_yolov8s_tuned/weights/best.pt'
        try:
            print(f"🔄 Đang tải YOLOv8 model từ: {self.yolo_model_path}")
            self.yolo_model = YOLO(self.yolo_model_path)
            print("✅ Đã tải YOLOv8 model thành công!")
        except Exception as e:
            print(f"❌ Lỗi khi tải YOLOv8 model: {e}")
            self.yolo_model = None

    def tinh_toan_ear_cai_tien(self, cac_diem_mat):
        """Tinh toan Eye Aspect Ratio (EAR) cải tiến với nhiều điểm hơn"""
        if len(cac_diem_mat) < 6:
            return 0
        
        # Tính EAR cơ bản
        v1 = np.linalg.norm(cac_diem_mat[1] - cac_diem_mat[5])
        v2 = np.linalg.norm(cac_diem_mat[2] - cac_diem_mat[4])
        h = np.linalg.norm(cac_diem_mat[0] - cac_diem_mat[3])
        
        ear_co_ban = (v1 + v2) / (2.0 * h) if h > 0 else 0
        
        # Nếu có thêm điểm, tính EAR mở rộng
        if len(cac_diem_mat) >= 9:
            v3 = np.linalg.norm(cac_diem_mat[6] - cac_diem_mat[8])
            ear_mo_rong = (v1 + v2 + v3) / (3.0 * h) if h > 0 else 0
            # Trung bình có trọng số
            ear = 0.7 * ear_co_ban + 0.3 * ear_mo_rong
        else:
            ear = ear_co_ban
        
        return ear
    
    def tinh_toan_ear(self, cac_diem_mat):
        """Wrapper cho hàm tính EAR cải tiến"""
        return self.tinh_toan_ear_cai_tien(cac_diem_mat)
    
    def trich_xuat_diem_mat(self, face_landmarks, chi_so_mat):
        """Trích xuất điểm landmark của mắt với xử lý lỗi tốt hơn"""
        cac_diem = []
        for idx in chi_so_mat:
            try:
                if idx < len(face_landmarks.landmark):
                    cac_diem.append(np.array([
                        face_landmarks.landmark[idx].x,
                        face_landmarks.landmark[idx].y
                    ]))
                else:
                    cac_diem.append(np.array([0.0, 0.0]))
            except (IndexError, AttributeError):
                cac_diem.append(np.array([0.0, 0.0]))
        return np.array(cac_diem)
    
    def tinh_perclos(self, lich_su_ear, nguong_ear=None):
        """Tính PERCLOS (Percentage of Eye Closure) - chỉ số quan trọng"""
        if nguong_ear is None:
            nguong_ear = self.NGUONG_EAR
        
        if len(lich_su_ear) == 0:
            return 0.0
        
        so_frame_nham = sum(1 for ear in lich_su_ear if ear < nguong_ear)
        perclos = so_frame_nham / len(lich_su_ear)
        return perclos
    
    def phan_tich_nhip_chop_mat(self, ear_hien_tai, thoi_gian_hien_tai):
        """Phân tích nhịp độ và thời gian chớp mắt"""
        # Xác định trạng thái chớp mắt
        dang_chop_hien_tai = ear_hien_tai < self.NGUONG_EAR
        
        thoi_gian_chop = 0
        tan_so_chop = 0
        loai_chop = "binh_thuong"
        
        if dang_chop_hien_tai and not self.dang_chop_mat:
            # Bắt đầu chớp mắt
            self.dang_chop_mat = True
            self.thoi_gian_bat_dau_chop = thoi_gian_hien_tai
        elif not dang_chop_hien_tai and self.dang_chop_mat:
            # Kết thúc chớp mắt
            self.dang_chop_mat = False
            thoi_gian_chop = thoi_gian_hien_tai - self.thoi_gian_bat_dau_chop
            
            # Lưu thời gian chớp
            self.bo_dem_blink_durations.append(thoi_gian_chop)
            self.so_lan_chop_mat += 1
            
            # Phân loại loại chớp mắt
            if thoi_gian_chop > self.NGUONG_SLOW_BLINK:
                loai_chop = "cham"
            elif thoi_gian_chop < self.NGUONG_BLINK_DURATION:
                loai_chop = "nhanh"
            
            # Tính tần số chớp mắt (trong 1 phút)
            if thoi_gian_hien_tai - self.thoi_gian_cuoi_chop > 0:
                khoang_cach_giua_cac_lan_chop = thoi_gian_hien_tai - self.thoi_gian_cuoi_chop
                if khoang_cach_giua_cac_lan_chop > 0:
                    tan_so_chop = 60.0 / khoang_cach_giua_cac_lan_chop
            
            self.thoi_gian_cuoi_chop = thoi_gian_hien_tai
        
        return {
            'thoi_gian_chop': thoi_gian_chop,
            'tan_so_chop': tan_so_chop,
            'loai_chop': loai_chop,
            'dang_chop': self.dang_chop_mat
        }
    
    def tinh_diem_so_ml_buon_ngu_cai_tien(self, ear_trung_binh, goc_yaw, goc_pitch, goc_roll, 
                                         thoi_gian_mat_nham, perclos, thong_tin_chop_mat):
        """Tính điểm số buồn ngủ từ ML cải tiến với nhiều đặc trưng hơn"""
        diem_so = 0.0
        
        # 1. Điểm số từ EAR (25% trọng số)
        if ear_trung_binh <= self.NGUONG_EAR_NGHIEM_NGAT:
            diem_so += 0.25
        elif ear_trung_binh <= self.NGUONG_EAR:
            ti_le = (self.NGUONG_EAR - ear_trung_binh) / (self.NGUONG_EAR - self.NGUONG_EAR_NGHIEM_NGAT)
            diem_so += 0.25 * ti_le
        
        # 2. Điểm số từ PERCLOS (25% trọng số) - QUAN TRỌNG
        if perclos > self.NGUONG_PERCLOS:
            diem_so += min(0.25, perclos / 0.5 * 0.25)  # Tối đa 25% nếu PERCLOS = 50%
        
        # 3. Điểm số từ thời gian mắt nhắm (20% trọng số)
        if thoi_gian_mat_nham > 0:
            diem_so += min(0.20, thoi_gian_mat_nham / self.NGUONG_THOI_GIAN_BUON_NGU * 0.20)
        
        # 4. Điểm số từ tư thế đầu (15% trọng số)
        diem_so_tu_the = 0
        if abs(goc_yaw) > 20:  # Đầu xoay ngang
            diem_so_tu_the += min(0.05, abs(goc_yaw) / 45.0 * 0.05)
        if abs(goc_pitch) > self.NGUONG_HEAD_NOD:  # Đầu gật
            diem_so_tu_the += min(0.05, abs(goc_pitch) / 30.0 * 0.05)
        if abs(goc_roll) > self.NGUONG_HEAD_TILT:  # Đầu nghiêng
            diem_so_tu_the += min(0.05, abs(goc_roll) / 30.0 * 0.05)
        diem_so += diem_so_tu_the
        
        # 5. Điểm số từ nhịp chớp mắt (15% trọng số)
        diem_so_chop_mat = 0
        if thong_tin_chop_mat['loai_chop'] == 'cham':
            diem_so_chop_mat += 0.08
        elif thong_tin_chop_mat['tan_so_chop'] < 10:  # Chớp mắt chậm
            diem_so_chop_mat += 0.05
        elif thong_tin_chop_mat['tan_so_chop'] > 25:  # Chớp mắt quá nhanh
            diem_so_chop_mat += 0.07
        
        # Nếu đang chớp mắt quá lâu
        if thong_tin_chop_mat['dang_chop'] and thong_tin_chop_mat['thoi_gian_chop'] > self.NGUONG_SLOW_BLINK:
            diem_so_chop_mat += 0.10
        
        diem_so += min(diem_so_chop_mat, 0.15)
        
        return min(diem_so, 1.0)
    
    def tinh_diem_so_dl_buon_ngu(self, yolo_label, confidence=0.8):
        """Tính điểm số buồn ngủ từ DL YOLO với logic 10 frame liên tục (0.0 - 1.0)"""
        if yolo_label is None:
            self.dem_frame_yolo_drowsy = max(0, self.dem_frame_yolo_drowsy - 1)
            return 0.0
        
        # Lưu trạng thái vào buffer
        self.bo_dem_yolo_drowsy.append(yolo_label.lower() == 'drowsy')
        
        if yolo_label.lower() == 'drowsy':
            self.dem_frame_yolo_drowsy += 1
            # Cần ít nhất 10 frame liên tục mới xác nhận drowsy
            if self.dem_frame_yolo_drowsy >= 10:
                return confidence
            else:
                return confidence * 0.5  # Giảm độ tin cậy nếu chưa đủ 10 frame
        elif yolo_label.lower() == 'alert':
            self.dem_frame_yolo_drowsy = 0
            return 0.0
        else:
            self.dem_frame_yolo_drowsy = max(0, self.dem_frame_yolo_drowsy - 1)
            return 0.0
    
    def hop_nhat_ket_qua_ml_dl(self, diem_so_ml, diem_so_dl, yolo_label):
        """Hợp nhất kết quả từ ML và DL để đưa ra quyết định cuối cùng"""
        # Tính điểm số tổng hợp theo trọng số
        diem_so_tong_hop = (diem_so_ml * self.TRONG_SO_ML + 
                           diem_so_dl * self.TRONG_SO_DL)
        
        # Lưu vào buffer để làm mượt kết quả
        self.bo_dem_ket_qua_fusion.append(diem_so_tong_hop)
        self.bo_dem_yolo_results.append(yolo_label)
        
        # Tính điểm trung bình từ buffer
        diem_so_trung_binh = np.mean(list(self.bo_dem_ket_qua_fusion))
        
        # Xác định trạng thái cuối cùng
        trang_thai_fusion = "tinh_tao"
        do_tin_cay = diem_so_trung_binh
        mau_fusion = (0, 255, 0)  # Xanh lá = tỉnh táo
        
        # Logic quyết định phức tạp hơn
        if diem_so_trung_binh >= self.NGUONG_FUSION_BUON_NGU:
            # Kiểm tra xem có đủ frame liên tục không
            if len(self.bo_dem_ket_qua_fusion) >= self.SO_FRAME_FUSION_XAC_NHAN:
                cac_frame_gan_day = list(self.bo_dem_ket_qua_fusion)[-self.SO_FRAME_FUSION_XAC_NHAN:]
                if all(score >= self.NGUONG_FUSION_BUON_NGU * 0.7 for score in cac_frame_gan_day):  # Giảm từ 0.8 xuống 0.7
                    self.dem_frame_buon_ngu_fusion += 1
                    trang_thai_fusion = "buon_ngu"
                    mau_fusion = (0, 0, 255)  # Đỏ = buồn ngủ
                else:
                    trang_thai_fusion = "nghi_ngo"
                    mau_fusion = (0, 165, 255)  # Cam = nghi ngờ
            else:
                trang_thai_fusion = "nghi_ngo"
                mau_fusion = (0, 165, 255)
        elif diem_so_trung_binh >= self.NGUONG_FUSION_BUON_NGU * 0.4:  # Giảm từ 0.5 xuống 0.4
            trang_thai_fusion = "canh_bao"
            mau_fusion = (0, 255, 255)  # Vàng = cảnh báo
        else:
            self.dem_frame_buon_ngu_fusion = max(0, self.dem_frame_buon_ngu_fusion - 1)
        
        # Xác định mức độ nguy hiểm
        muc_do_nguy_hiem = "THẤP"
        if do_tin_cay >= 0.7:  # Giảm từ 0.8 xuống 0.7
            muc_do_nguy_hiem = "CAO"
        elif do_tin_cay >= 0.5:  # Giảm từ 0.6 xuống 0.5
            muc_do_nguy_hiem = "TRUNG BÌNH"
        
        return {
            'trang_thai_fusion': trang_thai_fusion,
            'diem_so_ml': diem_so_ml,
            'diem_so_dl': diem_so_dl,
            'diem_so_tong_hop': diem_so_tong_hop,
            'diem_so_trung_binh': diem_so_trung_binh,
            'do_tin_cay': do_tin_cay,
            'mau_fusion': mau_fusion,
            'muc_do_nguy_hiem': muc_do_nguy_hiem,
            'dem_frame_buon_ngu': self.dem_frame_buon_ngu_fusion
        }
    
    def trich_xuat_hinh_mat(self, frame, face_landmarks, chi_so_mat, kich_thuoc_mat=(24, 24)):
        """Trich xuat va chuan hoa anh mat tu frame"""
        h, w, _ = frame.shape
        diem_mat = []
        for idx in chi_so_mat:
            if idx < len(face_landmarks.landmark):
                x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                diem_mat.append((x, y))
        
        if len(diem_mat) < 4:
            return np.zeros(kich_thuoc_mat)
        
        min_x = max(0, min([x for x, y in diem_mat]) - int((max([x for x, y in diem_mat]) - min([x for x, y in diem_mat])) * 0.2))
        max_x = min(w, max([x for x, y in diem_mat]) + int((max([x for x, y in diem_mat]) - min([x for x, y in diem_mat])) * 0.2))
        min_y = max(0, min([y for x, y in diem_mat]) - int((max([y for x, y in diem_mat]) - min([y for x, y in diem_mat])) * 0.2))
        max_y = min(h, max([y for x, y in diem_mat]) + int((max([y for x, y in diem_mat]) - min([y for x, y in diem_mat])) * 0.2))
        
        hinh_mat = frame[min_y:max_y, min_x:max_x]
        if hinh_mat.size == 0:
            return np.zeros(kich_thuoc_mat)
        
        hinh_mat = cv2.cvtColor(hinh_mat, cv2.COLOR_BGR2GRAY)
        hinh_mat = cv2.resize(hinh_mat, kich_thuoc_mat)
        hinh_mat = hinh_mat / 255.0
        return hinh_mat
    
    def trich_xuat_dac_trung_cai_tien(self, frame, face_landmarks):
        """Trích xuất các đặc trưng cải tiến cho mô hình học sâu"""
        h, w, _ = frame.shape
        
        hinh_mat_trai = self.trich_xuat_hinh_mat(frame, face_landmarks, self.CHI_SO_MAT_TRAI)
        hinh_mat_phai = self.trich_xuat_hinh_mat(frame, face_landmarks, self.CHI_SO_MAT_PHAI)
        
        mat_trai = self.trich_xuat_diem_mat(face_landmarks, self.CHI_SO_MAT_TRAI)
        mat_phai = self.trich_xuat_diem_mat(face_landmarks, self.CHI_SO_MAT_PHAI)
        
        ear_trai = self.tinh_toan_ear(mat_trai)
        ear_phai = self.tinh_toan_ear(mat_phai)
        
        # Tính toán các góc tư thế đầu chính xác hơn
        try:
            mui = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y, face_landmarks.landmark[1].z])
            cam = np.array([face_landmarks.landmark[18].x, face_landmarks.landmark[18].y, face_landmarks.landmark[18].z])
            diem_tai_trai = np.array([face_landmarks.landmark[234].x, face_landmarks.landmark[234].y, face_landmarks.landmark[234].z])
            diem_tai_phai = np.array([face_landmarks.landmark[454].x, face_landmarks.landmark[454].y, face_landmarks.landmark[454].z])
            tran = np.array([face_landmarks.landmark[10].x, face_landmarks.landmark[10].y, face_landmarks.landmark[10].z])
            
            vector_tai = diem_tai_phai - diem_tai_trai
            vector_mat = mui - cam
            vector_thang_dung = tran - cam
            
            # Chuẩn hóa vector
            vector_tai = vector_tai / (np.linalg.norm(vector_tai) + 1e-6)
            vector_mat = vector_mat / (np.linalg.norm(vector_mat) + 1e-6)
            vector_thang_dung = vector_thang_dung / (np.linalg.norm(vector_thang_dung) + 1e-6)
            
            # Tính góc chính xác hơn
            goc_yaw = math.asin(np.clip(vector_tai[2], -1, 1)) * 180 / math.pi
            goc_pitch = math.asin(np.clip(vector_mat[2], -1, 1)) * 180 / math.pi
            goc_roll = math.atan2(vector_thang_dung[0], vector_thang_dung[1]) * 180 / math.pi
        except (IndexError, ValueError):
            goc_yaw = goc_pitch = goc_roll = 0
        
        # Tính thêm các đặc trưng mới
        ear_trung_binh = (ear_trai + ear_phai) / 2
        ear_asymmetry = abs(ear_trai - ear_phai)  # Độ bất đối xứng
        
        dac_trung = np.array([
            ear_trai, ear_phai, ear_trung_binh, ear_asymmetry,
            goc_yaw, goc_pitch, goc_roll,
            abs(goc_yaw), abs(goc_pitch), abs(goc_roll)  # Thêm giá trị tuyệt đối
        ])
        
        hinh_mat_trai = hinh_mat_trai.reshape(24, 24, 1)
        hinh_mat_phai = hinh_mat_phai.reshape(24, 24, 1)
        
        return hinh_mat_trai, hinh_mat_phai, dac_trung
    
    def trich_xuat_dac_trung(self, frame, face_landmarks):
        """Wrapper cho hàm trích xuất đặc trưng cải tiến"""
        return self.trich_xuat_dac_trung_cai_tien(frame, face_landmarks)
    
    def ve_diem_mat_an_toan(self, frame, face_landmarks):
        """Ve landmarks mat mot cach an toan"""
        try:
            h, w, _ = frame.shape
            for idx in self.CHI_SO_MAT_TRAI + self.CHI_SO_MAT_PHAI:
                if idx < len(face_landmarks.landmark):
                    x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        except Exception as e:
            pass
    
    def xu_ly_yolo_detection(self, frame):
        """Xu ly detection bang YOLOv8"""
        if self.yolo_model is None:
            return None, 0.0, "Khong co model"
        
        try:
            # Chay inference
            results = self.yolo_model(frame, verbose=False)
            
            # Xu ly ket qua
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    # Lay detection co confidence cao nhat
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    if len(confidences) > 0:
                        max_conf_idx = np.argmax(confidences)
                        confidence = confidences[max_conf_idx]
                        class_id = int(classes[max_conf_idx])
                        
                        # Map class ID to label (gia su: 0=alert, 1=drowsy)
                        class_names = ['alert', 'drowsy']
                        if class_id < len(class_names):
                            label = class_names[class_id]
                            return label, confidence, f"YOLO: {label} ({confidence:.2f})"
            
            return "alert", 0.5, "YOLO: Khong phat hien"
            
        except Exception as e:
            return None, 0.0, f"Loi YOLO: {str(e)[:50]}"
    
    def xu_ly_frame_chinh(self, frame):
        """Ham xu ly frame chinh - ket hop ML va DL"""
        thoi_gian_hien_tai = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ket_qua_phat_hien = self.face_mesh.process(frame_rgb)
        
        # Khoi tao cac bien ket qua
        trang_thai_cuoi_cung = "tinh_tao"
        mau_trang_thai = (0, 255, 0)
        thong_tin_hien_thi = []
        yolo_label = None
        yolo_confidence = 0.0
        thong_tin_ml = {}
        yolo_info = ""
        ket_qua_fusion = {}
        
        # === XU LY YOLO DETECTION (DL) ===
        yolo_label, yolo_confidence, yolo_info = self.xu_ly_yolo_detection(frame)
        diem_so_dl = self.tinh_diem_so_dl_buon_ngu(yolo_label, yolo_confidence)
        
        # === XU LY MEDIAPIPE DETECTION (ML) ===
        if ket_qua_phat_hien.multi_face_landmarks:
            for face_landmarks in ket_qua_phat_hien.multi_face_landmarks:
                # Ve landmarks
                self.ve_diem_mat_an_toan(frame, face_landmarks)
                
                # Trich xuat dac trung
                hinh_mat_trai, hinh_mat_phai, dac_trung = self.trich_xuat_dac_trung(frame, face_landmarks)
                
                # Tinh cac chi so ML
                ear_trai = dac_trung[0]
                ear_phai = dac_trung[1]
                ear_trung_binh = dac_trung[2]
                goc_yaw = dac_trung[4]
                goc_pitch = dac_trung[5]
                goc_roll = dac_trung[6]
                
                # Luu EAR vao buffer
                self.bo_dem_ear.append(ear_trung_binh)
                self.bo_dem_ear_chi_tiet.append(ear_trung_binh)
                self.lich_su_trang_thai_mat.append(ear_trung_binh < self.NGUONG_EAR)
                
                # Tinh PERCLOS
                perclos = self.tinh_perclos(list(self.bo_dem_ear_chi_tiet))
                
                # Phan tich chop mat
                thong_tin_chop_mat = self.phan_tich_nhip_chop_mat(ear_trung_binh, thoi_gian_hien_tai)
                
                # Cap nhat trang thai mat nham
                if ear_trung_binh < self.NGUONG_EAR:
                    self.dem_mat_nham += 1
                    if self.thoi_gian_mat_nham == 0:
                        self.thoi_gian_mat_nham = thoi_gian_hien_tai
                    self.dem_mat_mo_lien_tuc = 0
                else:
                    if self.dem_mat_nham > 0:
                        self.dem_mat_mo_lien_tuc += 1
                        if self.dem_mat_mo_lien_tuc >= self.SO_FRAME_XAC_NHAN_TINH_TAO:
                            self.dem_mat_nham = 0
                            self.thoi_gian_mat_nham = 0
                            self.thoi_gian_tinh_tao = thoi_gian_hien_tai
                
                # Tinh thoi gian mat nham
                thoi_gian_mat_nham_hien_tai = 0
                if self.thoi_gian_mat_nham > 0:
                    thoi_gian_mat_nham_hien_tai = thoi_gian_hien_tai - self.thoi_gian_mat_nham
                
                # Tinh diem so ML
                diem_so_ml = self.tinh_diem_so_ml_buon_ngu_cai_tien(
                    ear_trung_binh, goc_yaw, goc_pitch, goc_roll,
                    thoi_gian_mat_nham_hien_tai, perclos, thong_tin_chop_mat
                )
                
                # === HOP NHAT KET QUA ML + DL ===
                ket_qua_fusion = self.hop_nhat_ket_qua_ml_dl(diem_so_ml, diem_so_dl, yolo_label)
                
                trang_thai_cuoi_cung = ket_qua_fusion['trang_thai_fusion']
                mau_trang_thai = ket_qua_fusion['mau_fusion']
                
                # Chuan bi thong tin hien thi
                thong_tin_hien_thi = [
                    f"TRANG THAI: {trang_thai_cuoi_cung.upper().replace('_', ' ')}",
                    f"Muc do: {ket_qua_fusion['muc_do_nguy_hiem']}",
                    f"",
                    f"=== CHI SO ML ===",
                    f"EAR: L={ear_trai:.3f} R={ear_phai:.3f} AVG={ear_trung_binh:.3f}",
                    f"PERCLOS: {perclos:.1%} ({'NGUY HIEM' if perclos > self.NGUONG_PERCLOS else 'BINH THUONG'})",
                    f"Thoi gian nham mat: {thoi_gian_mat_nham_hien_tai:.1f}s",
                    f"Chop mat: {thong_tin_chop_mat['loai_chop']} (tan so: {thong_tin_chop_mat['tan_so_chop']:.1f}/phut)",
                    f"Tu the dau: Yaw={goc_yaw:.1f}° Pitch={goc_pitch:.1f}° Roll={goc_roll:.1f}°",
                    f"",
                    f"=== CHI SO DL (YOLO) ===",
                    f"{yolo_info}",
                    f"",
                    f"=== KET QUA FUSION ===",
                    f"Diem ML: {ket_qua_fusion['diem_so_ml']:.3f}",
                    f"Diem DL: {ket_qua_fusion['diem_so_dl']:.3f}",
                    f"Diem tong hop: {ket_qua_fusion['diem_so_tong_hop']:.3f}",
                    f"Do tin cay: {ket_qua_fusion['do_tin_cay']:.3f}",
                    f"Frame buon ngu: {ket_qua_fusion['dem_frame_buon_ngu']}",
                    f"So lan canh bao: {self.SO_LAN_CANH_BAO_TICH_LUY}"
                ]
                thong_tin_ml = {
                    'ear_trai': ear_trai,
                    'ear_phai': ear_phai,
                    'ear_trung_binh': ear_trung_binh,
                    'perclos': perclos,
                    'thoi_gian_mat_nham': thoi_gian_mat_nham_hien_tai,
                    'chop_mat': thong_tin_chop_mat,
                    'goc_yaw': goc_yaw,
                    'goc_pitch': goc_pitch,
                    'goc_roll': goc_roll
                }
                break  # Chi xu ly mat dau tien
        else:
            # Khong phat hien mat
            diem_so_ml = 0.0
            ket_qua_fusion = self.hop_nhat_ket_qua_ml_dl(diem_so_ml, diem_so_dl, yolo_label)
            trang_thai_cuoi_cung = "khong_phat_hien_mat"
            mau_trang_thai = (128, 128, 128)  # Xam
            thong_tin_hien_thi = [
                "KHONG PHAT HIEN MAT",
                f"Chi su dung YOLO Detection:",
                f"{yolo_info}",
                f"Diem DL: {diem_so_dl:.3f}"
            ]
            thong_tin_ml = {}
        # Return as dict for GUI compatibility
        return {
            'frame': frame,
            'trang_thai_cuoi_cung': trang_thai_cuoi_cung,
            'mau_trang_thai': mau_trang_thai,
            'thong_tin_hien_thi': thong_tin_hien_thi,
            'thong_tin_ml': thong_tin_ml,
            'ket_qua_fusion': ket_qua_fusion,
            'yolo_info': yolo_info
        }
    
    def chay_phat_hien_buon_ngu(self, nguon_video=0):
        """Ham chinh chay phat hien buon ngu"""
        print("🚀 Bắt đầu hệ thống phát hiện buồn ngủ cải tiến ML+DL")
        print("📊 Sử dụng: MediaPipe (ML) + YOLOv8 (DL) + Fusion Algorithm")
        print("⚙️  Các tham số đã được tối ưu cho độ nhạy cao")
        print("🎯 Nhấn 'q' để thoát, 'r' để reset thống kê")
        print("-" * 60)
        
        cap = cv2.VideoCapture(nguon_video)
        if not cap.isOpened():
            print("❌ Lỗi: Không thể mở camera!")
            return
        
        # Cai dat camera cho hieu suat toi uu
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps_counter = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Không thể đọc dữ liệu từ camera")
                    break
                
                # Lat nguoc frame de giong nhu nhin guong
                frame = cv2.flip(frame, 1)
                
                # Xu ly frame
                result = self.xu_ly_frame_chinh(frame)
                frame_ket_qua = result.get('frame', frame)
                trang_thai = result.get('trang_thai_cuoi_cung', '')
                mau_trang_thai = result.get('mau_trang_thai', (255,255,255))
                thong_tin = result.get('thong_tin_hien_thi', [])
                
                # Ve khung va thong tin len frame
                self.ve_giao_dien_hien_thi(frame_ket_qua, trang_thai, mau_trang_thai, thong_tin)
                
                # Tinh va hien thi FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = 30 / elapsed_time if elapsed_time > 0 else 0
                    cv2.putText(frame_ket_qua, f"FPS: {current_fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    start_time = time.time()
                
                # Hien thi frame
                cv2.imshow('Phat Hien Buon Ngu - ML+DL Fusion', frame_ket_qua)
                
                # Xu ly phim bam
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("👋 Đang thoát hệ thống...")
                    break
                elif key == ord('r'):
                    print("🔄 Reset thống kê...")
                    self.reset_thong_ke()
                elif key == ord('s'):
                    # Luu screenshot
                    timestamp = int(time.time())
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_ket_qua)
                    print(f"📸 Đã lưu ảnh: {filename}")
                
        except KeyboardInterrupt:
            print("\n⚠️  Ngừng chương trình bởi người dùng")
        except Exception as e:
            print(f"❌ Lỗi không xác định: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Đã đóng tất cả tài nguyên")
    
    def ve_giao_dien_hien_thi(self, frame, trang_thai, mau_trang_thai, thong_tin):
        """Ve giao dien hien thi thong tin"""
        h, w, _ = frame.shape
        
        # Ve trạng thái buồn ngủ nhỏ gọn ở góc trên bên trái
        text_trang_thai = trang_thai.upper().replace('_', ' ')
        if text_trang_thai == "BUON NGU":
            color = (0, 0, 255)  # Đỏ cho buồn ngủ
        else:
            color = (255, 255, 255)  # Trắng cho trạng thái khác
        font_scale = 0.9
        thickness = 2
        # Vẽ text ở góc trên bên trái
        cv2.putText(frame, text_trang_thai, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # Không vẽ khung lớn, không vẽ mức độ nguy hiểm, không vẽ số lần cảnh báo
        # Chỉ vẽ hướng dẫn phím tắt nhỏ gọn ở góc dưới
        huong_dan = ["'q': Thoat", "'r': Reset", "'s': Chup anh"]
        for i, cmd in enumerate(huong_dan):
            cv2.putText(frame, cmd, (w - 150, h - 60 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def reset_thong_ke(self):
        """Reset tat ca thong ke"""
        self.dem_mat_nham = 0
        self.thoi_gian_mat_nham = 0
        self.thoi_gian_canh_bao_cuoi = 0
        self.dem_mat_mo_lien_tuc = 0
        self.SO_LAN_CANH_BAO_TICH_LUY = 0
        self.bo_dem_ear.clear()
        self.bo_dem_ear_chi_tiet.clear()
        self.bo_dem_ket_qua_fusion.clear()
        self.bo_dem_yolo_results.clear()
        self.bo_dem_yolo_drowsy.clear()
        self.lich_su_trang_thai_mat.clear()
        self.dem_frame_buon_ngu_fusion = 0
        self.dem_frame_yolo_drowsy = 0
        print("✅ Đã reset tất cả thống kê!")


# === CHUONG TRINH CHINH ===
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 HỆ THỐNG PHÁT HIỆN BUỒN NGỦ THÔNG MINH")
    print("🔬 Sử dụng Machine Learning + Deep Learning Fusion")
    print("⚡ Phiên bản cải tiến với độ nhạy cao")
    print("=" * 60)
    
    # Khoi tao he thong
    try:
        # Duong dan den model YOLO (co the thay doi)
        yolo_path = "savemodel/drowsiness_detection_yolov8s_tuned/weights/best.pt"
        
        # Kiem tra xem file model co ton tai khong
        if not os.path.exists(yolo_path):
            print(f"⚠️  Không tìm thấy YOLO model tại: {yolo_path}")
            print("🔄 Sẽ chỉ sử dụng MediaPipe (ML) để phát hiện")
            yolo_path = None
        
        detector = PhatHienBuonNgu(yolo_model_path=yolo_path)
        
        # Chon nguon video
        print("\n📹 Chọn nguồn video:")
        print("0: Camera mặc định")
        print("1: Camera ngoài (nếu có)")
        print("Hoặc nhập đường dẫn file video")
        
        nguon = input("Nhập lựa chọn (mặc định 0): ").strip()
        
        if nguon == "" or nguon == "0":
            nguon_video = 0
        elif nguon == "1":
            nguon_video = 1
        else:
            try:
                nguon_video = int(nguon)
            except ValueError:
                # Gia su la duong dan file
                nguon_video = nguon
        
        print(f"\n🎬 Sử dụng nguồn video: {nguon_video}")
        print("🚀 Khởi động hệ thống...")
        
        # Bat dau phat hien
        detector.chay_phat_hien_buon_ngu(nguon_video)
        
    except Exception as e:
        print(f"❌ Lỗi khởi tạo hệ thống: {e}")
        print("💡 Hãy kiểm tra:")
        print("   - Camera đã được kết nối")
        print("   - Các thư viện đã được cài đặt đầy đủ")
        print("   - Đường dẫn YOLO model (nếu sử dụng)")
    
    print("\n👋 Cảm ơn bạn đã sử dụng hệ thống!")
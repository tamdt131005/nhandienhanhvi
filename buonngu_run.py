import cv2
import numpy as np
from collections import deque
import time
import pygame
import mediapipe as mp
import math
import os
from ultralytics import YOLO


class PhatHienBuonNgu:
    def __init__(self, yolo_model_path=None):
        # Khoi tao MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Dinh nghia chi so cho cac diem landmarks cua mat
        self.CHI_SO_MAT_TRAI = [263, 387, 385, 362, 380, 373]
        self.CHI_SO_MAT_PHAI = [33, 160, 158, 133, 144, 153]
        
        # Khoi tao bien dem chop mat va thoi gian
        self.dem_mat_nham = 0
        self.thoi_gian_mat_nham = 0
        self.thoi_gian_canh_bao_cuoi = 0
        self.dem_mat_mo_lien_tuc = 0
        self.thoi_gian_tinh_tao = 0
        
        # Cai dat nguong canh bao
        self.NGUONG_MAT_NHAM_FRAME = 8
        self.NGUONG_THOI_GIAN_BUON_NGU = 2
        self.NGUONG_EAR = 0.27
        self.NGUONG_EAR_NGHIEM_NGAT = 0.23
        self.NGUONG_EAR_MO = 0.50
        self.SO_FRAME_XAC_NHAN_TINH_TAO = 10
        self.THOI_GIAN_CHO_GIUA_CANH_BAO = 3.0
        self.SO_LAN_CANH_BAO_TICH_LUY = 0
        
        # === CÁC THAM SỐ FUSION ML+DL ===
        self.TRONG_SO_ML = 0.7  # Tăng trọng số cho ML (EAR)
        self.TRONG_SO_DL = 0.3  # Giảm trọng số cho DL (YOLO)
        self.NGUONG_FUSION_BUON_NGU = 0.53  # Giảm ngưỡng tổng hợp để tăng nhạy
        self.SO_FRAME_FUSION_XAC_NHAN = 8  # Giảm số frame xác nhận liên tục
        
        # Buffer lưu trữ kết quả fusion
        self.bo_dem_ket_qua_fusion = deque(maxlen=10)
        self.bo_dem_yolo_results = deque(maxlen=5)
        self.bo_dem_yolo_drowsy = deque(maxlen=10)  # Buffer riêng cho DL drowsy detection
        self.dem_frame_buon_ngu_fusion = 0
        self.dem_frame_yolo_drowsy = 0  # Đếm frame drowsy liên tục cho DL
        
        # Khoi tao buffer luu tru cac dac trung theo chuoi thoi gian
        self.SO_BUOC_THOI_GIAN = 20
        self.bo_dem_dac_trung = deque(maxlen=self.SO_BUOC_THOI_GIAN)
        self.bo_dem_ear = deque(maxlen=15)
        
        # Khoi tao pygame cho canh bao am thanh
        try:
            pygame.mixer.init()
            self.co_am_thanh = True
        except:
            self.co_am_thanh = False
            print("Khong the khoi tao am thanh")
            
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

    def tinh_toan_ear(self, cac_diem_mat):
        """Tinh toan Eye Aspect Ratio (EAR) tu cac diem landmarks cua mat"""
        v1 = np.linalg.norm(cac_diem_mat[1] - cac_diem_mat[5])
        v2 = np.linalg.norm(cac_diem_mat[2] - cac_diem_mat[4])
        h = np.linalg.norm(cac_diem_mat[0] - cac_diem_mat[3])
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        return ear
    
    def trich_xuat_diem_mat(self, face_landmarks, chi_so_mat):
        """Trích xuất đúng 6 điểm landmark của mắt để tính EAR"""
        cac_diem = []
        for idx in chi_so_mat:
            try:
                cac_diem.append(np.array([
                    face_landmarks.landmark[idx].x,
                    face_landmarks.landmark[idx].y
                ]))
            except IndexError:
                cac_diem.append(np.array([0.0, 0.0]))
        return np.array(cac_diem)
    
    def tinh_diem_so_ml_buon_ngu(self, ear_trung_binh, goc_yaw, thoi_gian_mat_nham):
        """Tính điểm số buồn ngủ từ ML (0.0 - 1.0)"""
        diem_so = 0.0
        
        # Điểm số từ EAR (40% trọng số)
        if ear_trung_binh <= self.NGUONG_EAR_NGHIEM_NGAT:
            diem_so += 0.4  # EAR rất thấp
        elif ear_trung_binh <= self.NGUONG_EAR:
            diem_so += 0.3 * (self.NGUONG_EAR - ear_trung_binh) / (self.NGUONG_EAR - self.NGUONG_EAR_NGHIEM_NGAT)
        
        # Điểm số từ thời gian mắt nhắm (35% trọng số)
        if thoi_gian_mat_nham > 0:
            diem_so += min(0.35, thoi_gian_mat_nham / self.NGUONG_THOI_GIAN_BUON_NGU * 0.35)
        
        # Điểm số từ góc yaw - đầu nghiêng (25% trọng số)
        if abs(goc_yaw) > 20:
            diem_so += min(0.25, abs(goc_yaw) / 45.0 * 0.25)
        
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
                if all(score >= self.NGUONG_FUSION_BUON_NGU * 0.8 for score in cac_frame_gan_day):
                    self.dem_frame_buon_ngu_fusion += 1
                    trang_thai_fusion = "buon_ngu"
                    mau_fusion = (0, 0, 255)  # Đỏ = buồn ngủ
                else:
                    trang_thai_fusion = "nghi_ngo"
                    mau_fusion = (0, 165, 255)  # Cam = nghi ngờ
            else:
                trang_thai_fusion = "nghi_ngo"
                mau_fusion = (0, 165, 255)
        elif diem_so_trung_binh >= self.NGUONG_FUSION_BUON_NGU * 0.5:
            trang_thai_fusion = "canh_bao"
            mau_fusion = (0, 255, 255)  # Vàng = cảnh báo
        else:
            self.dem_frame_buon_ngu_fusion = max(0, self.dem_frame_buon_ngu_fusion - 1)
        
        # Xác định mức độ nguy hiểm
        muc_do_nguy_hiem = "THẤP"
        if do_tin_cay >= 0.8:
            muc_do_nguy_hiem = "CAO"
        elif do_tin_cay >= 0.6:
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
            x, y = int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)
            diem_mat.append((x, y))
        
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
    
    def trich_xuat_dac_trung(self, frame, face_landmarks):
        """Trich xuat cac dac trung cho mo hinh hoc sau"""
        h, w, _ = frame.shape
        
        hinh_mat_trai = self.trich_xuat_hinh_mat(frame, face_landmarks, self.CHI_SO_MAT_TRAI)
        hinh_mat_phai = self.trich_xuat_hinh_mat(frame, face_landmarks, self.CHI_SO_MAT_PHAI)
        
        mat_trai = self.trich_xuat_diem_mat(face_landmarks, self.CHI_SO_MAT_TRAI)
        mat_phai = self.trich_xuat_diem_mat(face_landmarks, self.CHI_SO_MAT_PHAI)
        
        ear_trai = self.tinh_toan_ear(mat_trai)
        ear_phai = self.tinh_toan_ear(mat_phai)
        
        mui = np.array([face_landmarks.landmark[4].x, face_landmarks.landmark[4].y, face_landmarks.landmark[4].z])
        cam = np.array([face_landmarks.landmark[152].x, face_landmarks.landmark[152].y, face_landmarks.landmark[152].z])
        diem_tai_trai = np.array([face_landmarks.landmark[234].x, face_landmarks.landmark[234].y, face_landmarks.landmark[234].z])
        diem_tai_phai = np.array([face_landmarks.landmark[454].x, face_landmarks.landmark[454].y, face_landmarks.landmark[454].z])
        
        vector_tai = diem_tai_phai - diem_tai_trai
        vector_mat = mui - cam
        
        vector_tai = vector_tai / (np.linalg.norm(vector_tai) + 1e-6)
        vector_mat = vector_mat / (np.linalg.norm(vector_mat) + 1e-6)
        
        goc_yaw = math.asin(max(min(vector_tai[2], 1), -1)) * 180 / math.pi
        goc_pitch = math.asin(max(min(vector_mat[2], 1), -1)) * 180 / math.pi
        goc_roll = math.atan2(vector_mat[0], vector_mat[1]) * 180 / math.pi
        
        dac_trung = np.array([ear_trai, ear_phai, goc_yaw, goc_pitch, goc_roll])
        
        hinh_mat_trai = hinh_mat_trai.reshape(24, 24, 1)
        hinh_mat_phai = hinh_mat_phai.reshape(24, 24, 1)
        
        return hinh_mat_trai, hinh_mat_phai, dac_trung
    
    def ve_diem_mat_an_toan(self, frame, face_landmarks):
        """Ve landmarks mat mot cach an toan"""
        h, w, _ = frame.shape
        
        for idx in self.CHI_SO_MAT_TRAI:
            if idx < len(face_landmarks.landmark):
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        
        for idx in self.CHI_SO_MAT_PHAI:
            if idx < len(face_landmarks.landmark):
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    
    def xac_nhan_trang_thai_buon_ngu(self, ear_trung_binh, goc_yaw, thoi_gian_hien_tai):
        """Logic xac nhan trang thai buon ngu cai tien - chỉ cho ML"""
        self.bo_dem_ear.append(ear_trung_binh)
        ear_trung_binh_bo_dem = np.mean(list(self.bo_dem_ear)) if len(self.bo_dem_ear) > 0 else ear_trung_binh
        
        nguong_dieu_chinh = self.NGUONG_EAR
        nguong_nghiem_ngat_dieu_chinh = self.NGUONG_EAR_NGHIEM_NGAT
        
        if abs(goc_yaw) > 15:
            nguong_dieu_chinh *= 1.15
            nguong_nghiem_ngat_dieu_chinh *= 1.1
        
        trang_thai_mat = "mo"
        mau_trang_thai = (0, 255, 0)
        
        if ear_trung_binh < nguong_nghiem_ngat_dieu_chinh:
            trang_thai_mat = "nham_hoan_toan"
            mau_trang_thai = (0, 0, 255)
            self.dem_mat_nham += 1
            self.dem_mat_mo_lien_tuc = 0
        elif ear_trung_binh < nguong_dieu_chinh:
            trang_thai_mat = "dang_nham"
            mau_trang_thai = (0, 165, 255)
            self.dem_mat_nham += 1
            self.dem_mat_mo_lien_tuc = 0
        else:
            trang_thai_mat = "mo"
            mau_trang_thai = (0, 255, 0)
            self.dem_mat_mo_lien_tuc += 1
            if self.dem_mat_mo_lien_tuc >= self.SO_FRAME_XAC_NHAN_TINH_TAO:
                self.dem_mat_nham = 0
                self.thoi_gian_mat_nham = 0
                self.thoi_gian_tinh_tao = thoi_gian_hien_tai
        
        co_buon_ngu = False
        van_ban_trang_thai = "ML: Tinh tao"
        mau_trang_thai_chinh = (0, 255, 0)
        
        if self.dem_mat_nham >= self.NGUONG_MAT_NHAM_FRAME:
            if self.thoi_gian_mat_nham == 0:
                self.thoi_gian_mat_nham = thoi_gian_hien_tai
            
            thoi_gian_mat_nham_lien_tuc = thoi_gian_hien_tai - self.thoi_gian_mat_nham
            
            if thoi_gian_mat_nham_lien_tuc >= self.NGUONG_THOI_GIAN_BUON_NGU:
                co_buon_ngu = True
                van_ban_trang_thai = f"ML: BUON NGU! ({thoi_gian_mat_nham_lien_tuc:.1f}s)"
                mau_trang_thai_chinh = (0, 0, 255)
            else:
                van_ban_trang_thai = f"ML: Nghi ngo ({thoi_gian_mat_nham_lien_tuc:.1f}s)"
                mau_trang_thai_chinh = (0, 165, 255)
        
        return {
            'co_buon_ngu': co_buon_ngu,
            'trang_thai_mat': trang_thai_mat,
            'van_ban_trang_thai': van_ban_trang_thai,
            'mau_trang_thai': mau_trang_thai,
            'mau_trang_thai_chinh': mau_trang_thai_chinh,
            'ear_trung_binh': ear_trung_binh,
            'ear_trung_binh_bo_dem': ear_trung_binh_bo_dem,
            'dem_mat_nham': self.dem_mat_nham,
            'thoi_gian_mat_nham': thoi_gian_hien_tai - self.thoi_gian_mat_nham if self.thoi_gian_mat_nham > 0 else 0
        }
    
    def xu_ly_frame(self, frame):
        """Xu ly frame video chinh: ML (EAR) + DL (YOLOv8n) + FUSION"""
        thoi_gian_hien_tai = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ket_qua = self.face_mesh.process(frame_rgb)
        
        # Khởi tạo thông tin trạng thái ML
        thong_tin_ml = {
            'co_buon_ngu': False,
            'trang_thai_mat': 'khong_phat_hien',
            'van_ban_trang_thai': 'ML: Khong phat hien khuon mat',
            'mau_trang_thai': (0, 0, 255),
            'mau_trang_thai_chinh': (0, 0, 255),
            'ear_trung_binh': 0,
            'ear_trung_binh_bo_dem': 0,
            'dem_mat_nham': 0,
            'thoi_gian_mat_nham': 0
        }
        
        du_doan_mo_hinh = None
        diem_so_ml = 0.0
        ear_trung_binh = 0.0
        goc_yaw = 0.0
        
        # === ML PIPELINE (EAR) ===
        if ket_qua.multi_face_landmarks:
            for face_landmarks in ket_qua.multi_face_landmarks:
                self.ve_diem_mat_an_toan(frame, face_landmarks)
                hinh_mat_trai, hinh_mat_phai, dac_trung = self.trich_xuat_dac_trung(frame, face_landmarks)
                ear_trung_binh = (dac_trung[0] + dac_trung[1]) / 2
                goc_yaw = dac_trung[2]
                
                self.bo_dem_dac_trung.append((hinh_mat_trai, hinh_mat_phai, dac_trung))
                thong_tin_ml = self.xac_nhan_trang_thai_buon_ngu(ear_trung_binh, goc_yaw, thoi_gian_hien_tai)
                
                # Tính điểm số ML
                diem_so_ml = self.tinh_diem_so_ml_buon_ngu(
                    ear_trung_binh, goc_yaw, thong_tin_ml['thoi_gian_mat_nham']
                )
                break
        
        # === DL PIPELINE (YOLOv8n detection) ===
        yolo_results = None
        yolo_label = None
        yolo_confidence = 0.0
        diem_so_dl = 0.0
        
        if self.yolo_model is not None:
            try:
                yolo_results = self.yolo_model.predict(frame, verbose=False)
                for r in yolo_results:
                    boxes = r.boxes.cpu().numpy() if hasattr(r, 'boxes') else []
                    names = r.names if hasattr(r, 'names') else {0: 'alert', 1: 'drowsy'}
                    for box in boxes:
                        cls = int(box.cls[0])
                        yolo_confidence = float(box.conf[0])
                        label = names.get(cls, str(cls))
                        yolo_label = label
                        break
                
                # Tính điểm số DL (có logic 4 frame)
                diem_so_dl = self.tinh_diem_so_dl_buon_ngu(yolo_label, yolo_confidence)
                
            except Exception as e:
                print(f"[YOLOv8] Lỗi khi detect: {e}")
        
        # === FUSION ML + DL ===
        ket_qua_fusion = self.hop_nhat_ket_qua_ml_dl(diem_so_ml, diem_so_dl, yolo_label)
        
        # Xử lý cảnh báo dựa trên kết quả fusion
        if ket_qua_fusion['trang_thai_fusion'] == 'buon_ngu':
            if thoi_gian_hien_tai - self.thoi_gian_canh_bao_cuoi >= self.THOI_GIAN_CHO_GIUA_CANH_BAO:
                self.thoi_gian_canh_bao_cuoi = thoi_gian_hien_tai
                self.SO_LAN_CANH_BAO_TICH_LUY += 1
                
        # Hiển thị kết quả
        self.hien_thi_thong_tin_frame(frame, thong_tin_ml, du_doan_mo_hinh, yolo_label, ket_qua_fusion)
        
        return frame, thong_tin_ml, du_doan_mo_hinh, yolo_label, ket_qua_fusion

    def hien_thi_thong_tin_frame(self, frame, thong_tin_ml=None, du_doan_mo_hinh=None, yolo_label=None, ket_qua_fusion=None):
        """Hiển thị thông tin tối giản trên frame"""
        h, w, _ = frame.shape
        
        # === HIỂN THỊ TRẠNG THÁI CHÍNH (FUSION) ===
        if ket_qua_fusion:
            # Trạng thái chính từ fusion - chỉ hiển thị trạng thái và mức độ
            trang_thai_text = f"TRANG THAI: {ket_qua_fusion['trang_thai_fusion'].upper()}"
            if ket_qua_fusion['trang_thai_fusion'] == 'buon_ngu':
                trang_thai_text = f"CANH BAO: BUON NGU!"
            
            # Hiển thị trạng thái chính
            cv2.putText(frame, trang_thai_text, (10, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, ket_qua_fusion['mau_fusion'], 3)
            
            # Chỉ hiển thị điểm tin cậy khi buồn ngủ
            if ket_qua_fusion['trang_thai_fusion'] in ['buon_ngu', 'canh_bao']:
                cv2.putText(frame, f"Do tin cay: {ket_qua_fusion['do_tin_cay']:.2f}", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ket_qua_fusion['mau_fusion'], 2)
        
        # === HIỂN THỊ THÔNG TIN BỔ SUNG TỐI GIẢN ===
        # Chỉ hiển thị EAR khi cần thiết
        if thong_tin_ml and thong_tin_ml['ear_trung_binh'] < 0.3:
            cv2.putText(frame, f"EAR: {thong_tin_ml['ear_trung_binh']:.3f}", (10, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Hiển thị trạng thái DL đơn giản
        if yolo_label is not None:
            dl_color = (0, 255, 0) if yolo_label == 'alert' else (0, 0, 255)
            cv2.putText(frame, f"DL: {yolo_label.upper()}", (w - 150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, dl_color, 2)
        # Chỉ hiển thị số lần cảnh báo khi có cảnh báo
        if self.SO_LAN_CANH_BAO_TICH_LUY > 0:
            cv2.putText(frame, f"Canh bao: {self.SO_LAN_CANH_BAO_TICH_LUY}", (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def phat_am_thanh_canh_bao(self):
        """Phát âm thanh cảnh báo buồn ngủ"""
        if self.co_am_thanh:
            try:
                # Tạo âm thanh cảnh báo đơn giản
                frequency = 800  # Hz
                duration = 500   # milliseconds
                sample_rate = 22050
                frames = int(duration * sample_rate / 1000)
                
                arr = np.zeros(frames)
                for i in range(frames):
                    arr[i] = np.sin(2 * np.pi * frequency * i / sample_rate)
                
                arr = (arr * 32767).astype(np.int16)
                sound = pygame.sndarray.make_sound(arr)
                sound.play()
            except Exception as e:
                print(f"Lỗi phát âm thanh: {e}")

    def luu_bao_cao_session(self):
        """Lưu báo cáo phiên làm việc"""
        thoi_gian_hien_tai = time.time()
        bao_cao = {
            'thoi_gian_ket_thuc': thoi_gian_hien_tai,
            'so_lan_canh_bao': self.SO_LAN_CANH_BAO_TICH_LUY,
            'tong_frame_buon_ngu': self.dem_frame_buon_ngu_fusion,
            'ty_le_fusion_scores': list(self.bo_dem_ket_qua_fusion) if self.bo_dem_ket_qua_fusion else [],
            'yolo_results': list(self.bo_dem_yolo_results) if self.bo_dem_yolo_results else []
        }
        
        try:
            import json
            with open(f'drowsiness_report_{int(thoi_gian_hien_tai)}.json', 'w', encoding='utf-8') as f:
                json.dump(bao_cao, f, indent=2, ensure_ascii=False)
            print(f"✅ Đã lưu báo cáo: drowsiness_report_{int(thoi_gian_hien_tai)}.json")
        except Exception as e:
            print(f"❌ Lỗi lưu báo cáo: {e}")

    def chay_ung_dung(self):
        """Chay ung dung chinh với fusion ML+DL"""
        print("=== HE THONG PHAT HIEN BUON NGU (ML + DL FUSION) ===")
        print("🔹 ML: Sử dụng EAR (Eye Aspect Ratio) và MediaPipe")
        print("🔹 DL: Sử dụng YOLOv8n detection")  
        print("🔹 FUSION: Kết hợp cả hai phương pháp")
        print("📝 Bấm 'Q' để thoát, 'S' để lưu báo cáo")
        print("="*50)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Không thể mở camera!")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("📸 Camera đã sẵn sàng. Nhấn 'Q' để thoát.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Không thể đọc frame từ camera!")
                    break
                
                frame = cv2.flip(frame, 1)
                frame_da_xu_ly, thong_tin_ml, du_doan_mo_hinh, yolo_label, ket_qua_fusion = self.xu_ly_frame(frame)
                
                # Hiển thị cửa sổ
                cv2.imshow('🚗 Phat Hien Buon Ngu - ML+DL Fusion', frame_da_xu_ly)
                
                # Phát cảnh báo âm thanh nếu cần
                if ket_qua_fusion and ket_qua_fusion['trang_thai_fusion'] == 'buon_ngu':
                    if ket_qua_fusion['dem_frame_buon_ngu'] % 30 == 0:  # Mỗi giây phát một lần
                        self.phat_am_thanh_canh_bao()
                
                # Xử lý phím bấm
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("🔄 Đang thoát ứng dụng...")
                    break
                elif key == ord('s') or key == ord('S'):
                    print("💾 Đang lưu báo cáo...")
                    self.luu_bao_cao_session()
        
        except KeyboardInterrupt:
            print("⚠️ Nhận Ctrl+C để dừng chương trình")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("🏁 Đã đóng ứng dụng")
            
            # Tự động lưu báo cáo khi thoát
            if self.SO_LAN_CANH_BAO_TICH_LUY > 0 or self.dem_frame_buon_ngu_fusion > 0:
                print("💾 Tự động lưu báo cáo cuối session...")
                self.luu_bao_cao_session()

    def in_thong_ke_hien_tai(self):
        """In thống kê hiện tại của hệ thống"""
        print("\n" + "="*50)
        print("📊 THỐNG KÊ HỆ THỐNG HIỆN TẠI")
        print("="*50)
        print(f"🔢 Số lần cảnh báo tích lũy: {self.SO_LAN_CANH_BAO_TICH_LUY}")
        print(f"📊 Số frame buồn ngủ fusion: {self.dem_frame_buon_ngu_fusion}")
        print(f"⚙️ Trọng số ML: {self.TRONG_SO_ML}")
        print(f"⚙️ Trọng số DL: {self.TRONG_SO_DL}")
        print(f"🎯 Ngưỡng fusion: {self.NGUONG_FUSION_BUON_NGU}")
        
        if len(self.bo_dem_ket_qua_fusion) > 0:
            diem_trung_binh = np.mean(list(self.bo_dem_ket_qua_fusion))
            print(f"📈 Điểm fusion trung bình: {diem_trung_binh:.3f}")
        
        if len(self.bo_dem_yolo_results) > 0:
            yolo_results = list(self.bo_dem_yolo_results)
            drowsy_count = sum(1 for r in yolo_results if r == 'drowsy')
            alert_count = sum(1 for r in yolo_results if r == 'alert')
            print(f"🤖 YOLO - Drowsy: {drowsy_count}, Alert: {alert_count}")
        
        print("="*50)

def main():
    """Hàm main với xử lý lỗi cải tiến"""
    try:
        print("🚀 Khởi động hệ thống phát hiện buồn ngủ...")
        
        # Khởi tạo hệ thống
        he_thong = PhatHienBuonNgu()
        
        # In thông tin cấu hình
        print(f"⚙️ Trọng số ML: {he_thong.TRONG_SO_ML}")
        print(f"⚙️ Trọng số DL: {he_thong.TRONG_SO_DL}")
        print(f"🎯 Ngưỡng fusion: {he_thong.NGUONG_FUSION_BUON_NGU}")
        
        # Chạy ứng dụng
        he_thong.chay_ung_dung()
        
        # In thống kê cuối
        he_thong.in_thong_ke_hien_tai()
        
    except Exception as e:
        print(f"❌ Lỗi trong quá trình chạy ứng dụng: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("🔚 Chương trình kết thúc")

if __name__ == "__main__":
    main()
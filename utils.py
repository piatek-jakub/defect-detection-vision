import cv2
import numpy as np
from PIL import Image
import random
def rotate_patch(patch, mask, angle):
    h, w = patch.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_patch = cv2.warpAffine(
        patch, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    rotated_mask = cv2.warpAffine(
        mask, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return rotated_patch, rotated_mask
def detect_hazelnut_mask(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    return binary

def apply_cut_and_paste(src_pil, target_pil, anomaly_map, threshold=30, dilation=3):
    """
    Wycina anomalię, rotuje ją, skaluje jeśli trzeba i wkleja w głąb orzecha.
    """
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2BGR)
    dst = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)
    
    # przygotowujemy maske anomalii
    amap_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask_anomaly = cv2.threshold(amap_norm, threshold, 255, cv2.THRESH_BINARY)
    
    if dilation > 0:
        kernel = np.ones((dilation, dilation), np.uint8)
        mask_anomaly = cv2.dilate(mask_anomaly, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask_anomaly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return target_pil
        
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    patch = src[y:y+h, x:x+w]
    patch_mask = mask_anomaly[y:y+h, x:x+w]

    # robimy losowa rotacje
    angle = random.uniform(0, 360)
    patch, patch_mask = rotate_patch(patch, patch_mask, angle)
    h, w = patch.shape[:2]

    # szukamy na orzechu bezpiecznego miejsca
    target_mask = detect_hazelnut_mask(dst)
    
    # robimy erozje zeby sprobowac znalezc bezpieczna strefe
    kernel_size_y = h // 2 + 5
    kernel_size_x = w // 2 + 5
    safe_zone_mask = cv2.erode(target_mask, np.ones((kernel_size_y, kernel_size_x), np.uint8))
    
    y_idx, x_idx = np.where(safe_zone_mask > 0)
    
    # jesli anomalia jest nadal za duza
    if len(y_idx) == 0:
        # zmniejszamy patch do 70% pierwotnej wielkosci
        scale = 0.7
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 5 and new_h > 5:
            patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            patch_mask = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            h, w = new_h, new_w
            
            # znowu probujemy wyznaczyc safe zone po zmniejszeniu
            safe_zone_mask = cv2.erode(target_mask, np.ones((h // 2 + 2, w // 2 + 2), np.uint8))
            y_idx, x_idx = np.where(safe_zone_mask > 0)

    # jak nadal sie nie miesci to robimy fallback do zwyklej maski
    if len(y_idx) == 0:
        y_idx, x_idx = np.where(target_mask > 0)
    
    if len(y_idx) == 0: return target_pil
    
    # wybieramy punkt i wklejamy
    rand_idx = random.randint(0, len(y_idx) - 1)
    t_y = y_idx[rand_idx] - h // 2
    t_x = x_idx[rand_idx] - w // 2
    
    # finalne dopasowanie do granic
    th, tw = dst.shape[:2]
    t_y = max(0, min(t_y, th - h))
    t_x = max(0, min(t_x, tw - w))
    
    # wyciecie i blending z maska
    roi = dst[t_y:t_y+h, t_x:t_x+w]
    mask_3d = cv2.cvtColor(patch_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
    
    blended_patch = (patch.astype(np.float32) * mask_3d + roi.astype(np.float32) * (1 - mask_3d))
    dst[t_y:t_y+h, t_x:t_x+w] = blended_patch.astype(np.uint8)
    
    return Image.fromarray(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))


class SmartConfig(dict):
    def get(self, key, default=None):
        if key not in self:
            print(f"DEBUG: Próbowano uzyskać parametr '{key}', ale go nie znaleziono. Używam domyślnej wartości: {default}")
        return super().get(key, default)
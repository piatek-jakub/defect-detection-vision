import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from PIL import Image, ImageTk
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading


try:
    plt.switch_backend('TkAgg') 
except ImportError:
    pass 

DEFAULT_CONFIG = {
    'IMG_SIZE': 512,
    'BATCH_SIZE': 8,
    'CROP_SIZE': 96,
    'ANOMALY_THRESHOLD': 0.75, 
    'SEGMENTATION_THRESHOLD_RATIO': 0.75,
    'DILATION_KERNEL_SIZE': 5 
}
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

def get_hazelnut_bbox(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(largest)  # x, y, w, h

class AnomalyDataset(Dataset):
    """ Ładuje próbki dla PatchCore (normalne) i testowe. """
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_path, label

class ResNet50_Extractor(nn.Module):
    """ Ekstraktor cech dla PatchCore. """
    def __init__(self):
        super().__init__()
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        self.layer1 = nn.Sequential(*list(m.children())[:5]) 
        self.layer2 = list(m.children())[5]
        self.layer3 = list(m.children())[6]
        
    def forward(self, x):
        _ = self.layer1(x) 
        f2 = self.layer2(_)
        f3 = self.layer3(f2)
        return f2, f3

class PatchCore:
    """ Implementacja PatchCore. """
    def __init__(self, sampling_ratio=0.2, device="cpu"):
        self.sampling_ratio = sampling_ratio
        self.memory_bank = None
        self.device = device

    @torch.no_grad()
    def build_memory(self, dataloader, extractor):
        all_features = []
        for imgs, _, _ in dataloader: 
            imgs = imgs.to(self.device)
            f2, f3 = extractor(imgs) 
            
            f3_upsampled = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
            feats_aggregated = torch.cat((f2, f3_upsampled), dim=1)
            feats = F.normalize(feats_aggregated, dim=1)
            feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.size(1))
            all_features.append(feats.cpu())

        all_features = torch.cat(all_features, dim=0)
        num_samples = int(len(all_features) * self.sampling_ratio)
        indices = torch.randperm(len(all_features))[:num_samples]
        self.memory_bank = all_features[indices].to(self.device)

    @torch.no_grad()
    def predict(self, img, extractor, img_size):
        f2, f3 = extractor(img.unsqueeze(0).to(self.device))

        f3_upsampled = F.interpolate(f3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        feats_aggregated = torch.cat((f2, f3_upsampled), dim=1)
        feats = F.normalize(feats_aggregated, dim=1)
        
        feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.size(1))
        d = torch.cdist(feats, self.memory_bank)
        min_dist, _ = torch.min(d, dim=1)
        
        anomaly_map = min_dist.reshape(img_size // 8, img_size // 8) 
        anomaly_map = cv2.resize(anomaly_map.cpu().numpy(), (img_size, img_size))

        score = anomaly_map.max()
        return anomaly_map, float(score)

class AnomalyClassifierCNN(nn.Module):
    """ Prosty Klasyfikator CNN dla wyciętych łatek. """
    def __init__(self, num_classes, crop_size):
        super().__init__()
        self.crop_size = int(crop_size)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2)  
        )
        final_dim = 64 * (self.crop_size // 8) * (self.crop_size // 8) 
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(final_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AugmentableTensorDataset(Dataset):
    """ 
    Dataset opakowujący tensory (wycięte łatki) i aplikujący transformacje (augmentację)
    w locie, konwertując tymczasowo do PIL Image.
    """
    def __init__(self, data_tensor, labels_tensor, transform=None):
        self.data_tensor = data_tensor 
        self.labels_tensor = labels_tensor
        self.transform = transform
        
    def __len__(self):
        return len(self.labels_tensor)
        
    def __getitem__(self, idx):
        img_tensor = self.data_tensor[idx]
        label = self.labels_tensor[idx]
        

        img_np = img_tensor.numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        
        if self.transform:
            img_pil = self.transform(img_pil)
            
        return img_pil.to(img_tensor.dtype), label


class AnomalyDetectionApp:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = device
        self.anomaly_classes = ["crack", "cut", "hole", "print"]
        self.num_classes = len(self.anomaly_classes)
        
        self.extractor = None
        self.patchcore = None
        self.classifier = None
        
        self.test_samples = []
        self.test_class_to_idx = {name: i for i, name in enumerate(self.anomaly_classes)}
        

        self.transform = self._create_base_transform(self.config['IMG_SIZE'])
        self.synthetic_samples = []

    def get_synthetic_sample(self, index):
        """ Zwraca buforowaną syntetyczną próbkę C&P (oryginalny płat, tło, finalna łatka). """
        if 0 <= index < len(self.synthetic_samples):
            return self.synthetic_samples[index]
        return None
    
    def _create_base_transform(self, img_size):
        """ Tworzy podstawową transformację do skalowania dla PatchCore (512x512). """
        return transforms.Compose([
            transforms.Resize((int(img_size), int(img_size))),
            transforms.ToTensor(),
        ])

    def _create_train_transform(self, is_augmenting, crop_size):
        """ Tworzy transformację treningową dla CNN (wycięte łatki). """
        crop_size = int(crop_size)
        
        if is_augmenting:
            train_transforms = [

                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=10),
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
            ]
        else:

            train_transforms = [
                transforms.Resize((crop_size, crop_size)),
                transforms.ToTensor(),
            ]
            
        return transforms.Compose(train_transforms)
    def _load_good_samples(self):
        """ Pomocnicza funkcja do ładowania ścieżek tylko do obrazów 'good'. """
        return [path for path, _ in self._load_all_samples("./train", ["good"])]
    @property
    def total_synthetic_samples(self):
        return len(self.synthetic_samples)
    def generate_cut_paste_sample(self, anomaly_path, anomaly_map,
                              target_img_path, target_label_idx):

        img_size = int(self.config['IMG_SIZE'])
        crop_size = int(self.config['CROP_SIZE'])
        segment_ratio = self.config['SEGMENTATION_THRESHOLD_RATIO']
        dilation_size = int(self.config['DILATION_KERNEL_SIZE'])
    
        margin = 10

        img_anomaly = cv2.imread(anomaly_path)
        if img_anomaly is None:
            return None
    
        img_anomaly = cv2.resize(img_anomaly, (img_size, img_size))
    
        max_val = anomaly_map.max()
        if max_val <= 0:
            return None
    
        threshold = max_val * segment_ratio
        binary_map = (anomaly_map >= threshold).astype(np.uint8) * 255
    
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated_map = cv2.dilate(binary_map, kernel, iterations=1)
    
        contours, _ = cv2.findContours(
            dilated_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
    
        if not contours:
            return None
    
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
    
        if w < 5 or h < 5:
            return None
    
        patch = img_anomaly[y:y+h, x:x+w].copy()
        patch_mask = dilated_map[y:y+h, x:x+w].copy()
    

        scale = np.random.uniform(0.6, 1.0)
        patch = cv2.resize(patch, None, fx=scale, fy=scale)
        patch_mask = cv2.resize(
            patch_mask, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_NEAREST
        )
    
        angle = np.random.uniform(-25, 25)
        patch, patch_mask = rotate_patch(patch, patch_mask, angle)
    
        h, w = patch.shape[:2]

        img_target = cv2.imread(target_img_path)
        if img_target is None:
            return None
    
        img_target = cv2.resize(img_target, (img_size, img_size))
    
        hazelnut_mask = detect_hazelnut_mask(img_target)
        bbox = get_hazelnut_bbox(hazelnut_mask)
    
        if bbox is None:
            return None
    
        hx, hy, hw, hh = bbox

        x_min = hx + margin
        x_max = hx + hw - w - margin
        y_min = hy + margin
        y_max = hy + hh - h - margin
    
        if x_min >= x_max or y_min >= y_max:
            return None
    
        for _ in range(10):
            x_paste = np.random.randint(x_min, x_max)
            y_paste = np.random.randint(y_min, y_max)
    
            roi_mask = hazelnut_mask[y_paste:y_paste+h, x_paste:x_paste+w]
            if roi_mask.shape[:2] == (h, w) and np.mean(roi_mask) > 200:
                break
        else:
            return None

        img_target_clone = img_target.copy()
    
        alpha = patch_mask.astype(np.float32) / 255.0
        alpha = alpha[..., None]
    
        roi = img_target_clone[y_paste:y_paste+h, x_paste:x_paste+w]
    
        blended = roi * (1 - alpha) + patch * alpha
        img_target_clone[y_paste:y_paste+h, x_paste:x_paste+w] = blended.astype(np.uint8)

        center_x = x_paste + w // 2
        center_y = y_paste + h // 2
    
        half_crop = crop_size // 2
    
        cx1 = max(0, center_x - half_crop)
        cy1 = max(0, center_y - half_crop)
        cx2 = min(img_size, cx1 + crop_size)
        cy2 = min(img_size, cy1 + crop_size)
    
        cropped = img_target_clone[cy1:cy2, cx1:cx2]
        cropped = cv2.resize(cropped, (crop_size, crop_size))
    
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        cropped_tensor = transforms.ToTensor()(cropped).float()
    
        return cropped_tensor, target_label_idx
    def _load_all_samples(self, root_dir, classes):
        """ Pomocnicza funkcja do ładowania listy ścieżek i etykiet. """
        samples = []
        for class_name in classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): continue
            label = self.test_class_to_idx.get(class_name, -1)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(class_path, img_name)
                    samples.append((img_path, label))
        return samples

    def load_test_data(self, test_classes):
        """ Ładuje wszystkie obrazki testowe. """
        self.test_samples = []
        for class_name in test_classes:
            self.test_samples.extend(self._load_all_samples("./test", [class_name]))
        self.test_samples.sort(key=lambda x: (x[1], x[0]))
        return len(self.test_samples)

    def crop_anomaly_region(self, image_path, anomaly_map):
        """ Dynamiczne wycinanie na podstawie największego konturu. """
        img = cv2.imread(image_path)
        img_size = int(self.config['IMG_SIZE'])
        crop_size = int(self.config['CROP_SIZE'])
        segment_ratio = self.config['SEGMENTATION_THRESHOLD_RATIO']
        dilation_size = int(self.config['DILATION_KERNEL_SIZE'])

        img_resized = cv2.resize(img, (img_size, img_size))
        
        max_val = anomaly_map.max()
        threshold = max_val * segment_ratio
        
        binary_map = np.zeros_like(anomaly_map, dtype=np.uint8)
        binary_map[anomaly_map >= threshold] = 255
        binary_map = binary_map.astype(np.uint8)
        
        kernel = np.ones((dilation_size, dilation_size), np.uint8)
        dilated_map = cv2.dilate(binary_map, kernel, iterations=1)
        
        contours, _ = cv2.findContours(dilated_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:

            max_idx = np.unravel_index(anomaly_map.argmax(), anomaly_map.shape)
            center_y, center_x = max_idx[0], max_idx[1] 
            half_crop = crop_size // 2
            x_min = max(0, center_x - half_crop)
            y_min = max(0, center_y - half_crop)
            x_max = min(img_size, center_x + half_crop)
            y_max = min(img_size, center_y + half_crop)
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            margin = 10 
            x_min = max(0, x - margin)
            y_min = max(0, y - margin)
            x_max = min(img_size, x + w + margin)
            y_max = min(img_size, y + h + margin)

        cropped_img = img_resized[y_min:y_max, x_min:x_max]

        if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
            cropped_img = img_resized[:crop_size, :crop_size] 

        cropped_img = cv2.resize(cropped_img, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
        return cropped_img

    def train_full_model(self, num_epochs, status_callback, is_augmenting, is_cut_paste):
        """ Główne trenowanie: PatchCore + CNN """
        self.synthetic_samples = []

        batch_size_int = int(self.config['BATCH_SIZE'])
        
        status_callback("Inicjalizacja i budowa PatchCore...")
        self.extractor = ResNet50_Extractor().to(self.device).eval()
        self.patchcore = PatchCore(sampling_ratio=0.2, device=self.device)
        

        normal_samples = self._load_all_samples("./train", ["good"])
        normal_ds = AnomalyDataset(normal_samples, self.transform)
        normal_loader = DataLoader(normal_ds, batch_size=batch_size_int, shuffle=False)
        self.patchcore.build_memory(normal_loader, self.extractor)


        status_callback("Generowanie danych CNN z PatchCore...")
        all_anomaly_tensors = []
        all_anomaly_labels = []
        
        all_anomaly_samples = self._load_all_samples("./test", self.anomaly_classes)
        anomaly_ds = AnomalyDataset(all_anomaly_samples, self.transform)
        good_img_paths = self._load_good_samples()
        for img_tensor, path, label_idx in anomaly_ds:
            anomaly_map, score = self.patchcore.predict(img_tensor, self.extractor, int(self.config['IMG_SIZE']))
            
            if score > self.config['ANOMALY_THRESHOLD']: 
                cropped_img_bgr = self.crop_anomaly_region(path, anomaly_map)
                cropped_img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
                cropped_tensor = transforms.ToTensor()(cropped_img_rgb).float()
                
                all_anomaly_tensors.append(cropped_tensor)

                class_name = os.path.basename(os.path.dirname(path))
                all_anomaly_labels.append(self.test_class_to_idx[class_name]) 
                current_label_idx = self.test_class_to_idx[class_name]

                if is_cut_paste and good_img_paths:

                    target_img_path = np.random.choice(good_img_paths)

                    synthetic_result = self.generate_cut_paste_sample(
                        anomaly_path=path, 
                        anomaly_map=anomaly_map, 
                        target_img_path=target_img_path,
                        target_label_idx=current_label_idx
                    )

                    if synthetic_result:
                        synthetic_tensor, synthetic_label = synthetic_result
                        all_anomaly_tensors.append(synthetic_tensor)
                        all_anomaly_labels.append(synthetic_label)
                        self.synthetic_samples.append({
                            'original_crop_bgr': cropped_img_bgr,
                            'good_bg_path': target_img_path, 
                            'final_synthetic_tensor': synthetic_tensor, 
                            'label_idx': synthetic_label
                        })

        if not all_anomaly_tensors:
            status_callback("Błąd: Nie wykryto anomalii powyżej progu PatchCore. Anulowanie treningu CNN.", is_error=True)
            return None

        X = all_anomaly_tensors
        y = all_anomaly_labels
        

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

        test_tensors = torch.stack(X_test).to(self.device)
        test_labels = torch.LongTensor(y_test).to(self.device)
        

        train_tensors = torch.stack(X_train).to("cpu")
        train_labels = torch.LongTensor(y_train).to("cpu")


        train_transform_cnn = self._create_train_transform(is_augmenting, self.config['CROP_SIZE'])
        train_data_cnn = AugmentableTensorDataset(train_tensors, train_labels, transform=train_transform_cnn)
        train_loader_cnn = DataLoader(train_data_cnn, batch_size=16, shuffle=True)


        self.classifier = AnomalyClassifierCNN(self.num_classes, self.config['CROP_SIZE']).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001, weight_decay=1e-4) 
        
        train_losses, test_losses, test_accuracies = [], [], []

        for epoch in range(num_epochs):
            self.classifier.train()
            running_loss = 0.0
            
            for inputs, labels in train_loader_cnn: 
                
                inputs, labels = inputs.to(self.device), labels.to(self.device) 
                
                optimizer.zero_grad()
                outputs = self.classifier(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            

            avg_train_loss = running_loss / len(train_loader_cnn)
            test_acc, test_loss = self._evaluate_classifier(test_tensors, test_labels, criterion)
            
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
            test_accuracies.append(test_acc)
            
            status_callback(f"Epoka {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Test Acc: {test_acc*100:.2f}%")
        
        status_callback("Trening CNN zakończony. Model gotowy.")
        
        return train_losses, test_losses, test_accuracies, test_tensors, test_labels


    def _evaluate_classifier(self, data_tensors, data_labels, criterion):
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(data_tensors)
            loss = criterion(outputs, data_labels).item()
            _, predicted = torch.max(outputs.data, 1)
            y_true = data_labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            accuracy = accuracy_score(y_true, y_pred)
        return accuracy, loss

    def get_test_sample(self, index):
        if index < 0 or index >= len(self.test_samples): return None, None, None
        img_path, label_idx = self.test_samples[index]
        real_label = self.anomaly_classes[label_idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, img_path, real_label

    def predict_sample(self, img_tensor, img_path):
        if not self.extractor or not self.patchcore or not self.classifier:
            raise ValueError("Modele nie zostały załadowane lub wytrenowane.")

        anomaly_map, score = self.patchcore.predict(img_tensor, self.extractor, int(self.config['IMG_SIZE']))
        
        predicted_label = "N/A"
        cropped_img_bgr = None
        
        if score > self.config['ANOMALY_THRESHOLD']:
            cropped_img_bgr = self.crop_anomaly_region(img_path, anomaly_map)
            cropped_tensor = transforms.ToTensor()(cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)).float()
            self.classifier.eval() 
            cropped_tensor = cropped_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.classifier(cropped_tensor)
                _, pred_idx = torch.max(output.data, 1)
            
            predicted_label = self.anomaly_classes[pred_idx.item()] 

        return anomaly_map, score, predicted_label, cropped_img_bgr

    def save_classifier(self, file_path):
        if self.classifier:
            torch.save(self.classifier.state_dict(), file_path)
            return True
        return False
        
    def load_classifier(self, file_path):

        self.classifier = AnomalyClassifierCNN(self.num_classes, int(self.config['CROP_SIZE'])).to(self.device)
        self.classifier.load_state_dict(torch.load(file_path, map_location=self.device))
        self.classifier.eval()
        

        self.extractor = ResNet50_Extractor().to(self.device).eval()
        self.patchcore = PatchCore(sampling_ratio=0.2, device=self.device)
        normal_samples = self._load_all_samples("./train", ["good"])
        normal_ds = AnomalyDataset(normal_samples, self.transform)
        normal_loader = DataLoader(normal_ds, batch_size=int(self.config['BATCH_SIZE']), shuffle=False) 
        self.patchcore.build_memory(normal_loader, self.extractor)
        
        return True

class AnomalyGUI:
    def __init__(self, master):
        self.master = master
        master.title("PatchCore + CNN Anomaly Classifier")

        self.config = DEFAULT_CONFIG.copy()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.app_logic = AnomalyDetectionApp(self.config, self.device)
        
        self.current_index = 0
        self.total_samples = 0
        self.total_synthetic_samples = 0
        self.current_synthetic_index = 0
        self.test_tensors_for_cm = None
        self.test_labels_for_cm = None


        self.param_vars = {}
        self.epoch_var = tk.IntVar(value=10)
        self.augment_var = tk.BooleanVar(value=False)
        self.test_classes_var = {c: tk.BooleanVar(value=True) for c in self.app_logic.anomaly_classes}
        self.status_var = tk.StringVar(value=f"Gotowy. DEVICE: {self.device}")
        self.cut_paste_var = tk.BooleanVar(value=False) 
        self.create_widgets()
        self.load_test_samples()
        

        master.protocol("WM_DELETE_WINDOW", self.on_closing) 

    def on_closing(self):
        self.master.destroy()

    def after_idle_call(self, func, *args):
        """ Bezpiecznie wykonuje funkcję w głównym wątku Tkinter. """
        self.master.after_idle(lambda: func(*args))

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        self.frame_train = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_train, text='1. Trening & Parametry')
        self.create_param_section(self.frame_train)
        self.create_train_section(self.frame_train)
        
        self.frame_test = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_test, text='2. Testowanie & Wyniki')
        self.create_test_section(self.frame_test)
        self.create_image_display(self.frame_test)

        self.frame_cut_paste = ttk.Frame(self.notebook)
        self.notebook.add(self.frame_cut_paste, text='3. Syntetyczne Dane C&P')
        self.create_cut_paste_display(self.frame_cut_paste)

        status_bar = ttk.Label(self.master, textvariable=self.status_var, relief=tk.SUNKEN, anchor='w')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    def create_cut_paste_display(self, parent):
        cp_frame = ttk.LabelFrame(parent, text="Wizualizacja Generowania Syntetycznych Próbek")
        cp_frame.pack(padx=10, pady=10, fill="both", expand=True)


        nav_frame = ttk.Frame(cp_frame)
        nav_frame.pack(pady=5)
        ttk.Button(nav_frame, text="< Poprzedni C&P", command=lambda: self.navigate_synthetic(-1)).pack(side=tk.LEFT, padx=5)
        self.cp_index_label = ttk.Label(nav_frame, text="0/0")
        self.cp_index_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Następny C&P >", command=lambda: self.navigate_synthetic(1)).pack(side=tk.LEFT, padx=5)


        self.cp_canvas = tk.Canvas(cp_frame, height=250, bg='white')
        self.cp_canvas.pack(padx=10, pady=10, fill="both", expand=True)


        self.cp_status_label = ttk.Label(cp_frame, text="---")
        self.cp_status_label.pack(pady=5)    
    def navigate_synthetic(self, direction):
        self.total_synthetic_samples = self.app_logic.total_synthetic_samples
        if self.total_synthetic_samples == 0: 
            self.cp_status_label.config(text="Brak wygenerowanych próbek C&P. Uruchom trening z włączoną opcją C&P.")
            return

        new_index = self.current_synthetic_index + direction
        if 0 <= new_index < self.total_synthetic_samples:
            self.current_synthetic_index = new_index
            self.update_cut_paste_display()    
    def update_cut_paste_display(self):
        self.total_synthetic_samples = self.app_logic.total_synthetic_samples
        if self.total_synthetic_samples == 0:
            self.cp_index_label.config(text="0/0")
            self.cp_canvas.delete("all")
            return

        self.cp_index_label.config(text=f"{self.current_synthetic_index + 1}/{self.total_synthetic_samples}")

        sample_data = self.app_logic.get_synthetic_sample(self.current_synthetic_index)

        if not sample_data:
            self.cp_status_label.config(text="Błąd ładowania próbki.")
            return


        orig_crop_bgr = sample_data['original_crop_bgr']
        good_bg_path = sample_data['good_bg_path']
        final_synthetic_tensor = sample_data['final_synthetic_tensor']
        label = self.app_logic.anomaly_classes[sample_data['label_idx']]

        img_bg = cv2.imread(good_bg_path)
        img_bg = cv2.resize(img_bg, (200, 200))

        img_crop = cv2.resize(orig_crop_bgr, (200, 200))

        tensor_np = final_synthetic_tensor.permute(1, 2, 0).numpy() # C,H,W -> H,W,C
        img_synthetic = (tensor_np * 255).astype(np.uint8)
        img_synthetic = cv2.cvtColor(img_synthetic, cv2.COLOR_RGB2BGR)
        img_synthetic = cv2.resize(img_synthetic, (200, 200))

        self._display_cut_paste_images(
            cv2.cvtColor(img_bg, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(img_synthetic, cv2.COLOR_BGR2RGB)
        )
        self.cp_status_label.config(text=f"Klasa Anomalii: {label} | Generowane z tła: {os.path.basename(good_bg_path)}")
    def create_param_section(self, parent):
        param_frame = ttk.LabelFrame(parent, text="Parametry Modeli")
        param_frame.pack(padx=10, pady=10, fill="x")

        row = 0
        for name, default in DEFAULT_CONFIG.items():
            ttk.Label(param_frame, text=f"{name}:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            var = tk.DoubleVar(value=default)
            entry = ttk.Entry(param_frame, textvariable=var, width=10)
            entry.grid(row=row, column=1, sticky="w", padx=5, pady=2)
            self.param_vars[name] = var
            row += 1

        ttk.Label(param_frame, text="Liczba Epok CNN:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(param_frame, textvariable=self.epoch_var, width=10).grid(row=row, column=1, sticky="w", padx=5, pady=2)
    def _display_cut_paste_images(self, bg_img_rgb, crop_img_rgb, final_img_rgb):
        """ Rysuje 3 obrazy (Tło, Płat, Wynik) na canvas C&P. """
        canvas = self.cp_canvas
        canvas.delete("all")

        images = {
            "1. Tło 'Good'": bg_img_rgb,
            "2. Oryginalny Płat (Anomalia)": crop_img_rgb,
            "3. Syntetyczny Rezultat (do CNN)": final_img_rgb
        }

        self.cp_tk_images = []
        x_offset = 10
        img_display_size = 200

        for title, img_data in images.items():
            img_pil = Image.fromarray(img_data)
            tk_img = ImageTk.PhotoImage(img_pil)
            self.cp_tk_images.append(tk_img)

            canvas.create_image(x_offset, 10, anchor=tk.NW, image=tk_img)
            canvas.create_text(x_offset + img_display_size / 2, 10 + img_display_size + 10, 
                               text=title, font=('Arial', 8, 'bold'))

            x_offset += img_display_size + 20
    def create_train_section(self, parent):
        train_frame = ttk.LabelFrame(parent, text="Kontrola Modelu & Trening")
        train_frame.pack(padx=10, pady=10, fill="x")
        

        ttk.Checkbutton(train_frame, 
                        text="Włącz Augmentację CNN (Flips, Rotacje)", 
                        variable=self.augment_var).pack(fill="x", padx=5, pady=5)

        ttk.Checkbutton(train_frame, 
                        text="Włącz Cut-and-Paste (Wklejanie wad na tło 'good')", 
                        variable=self.cut_paste_var).pack(fill="x", padx=5, pady=5)

        ttk.Button(train_frame, text="1. Uruchom Trening CNN", command=self.start_training_thread).pack(fill="x", padx=5, pady=5)
        ttk.Button(train_frame, text="2. Zapisz Wytrenowany Model", command=self.save_model).pack(fill="x", padx=5, pady=5)
        ttk.Button(train_frame, text="3. Wczytaj Model CNN (.pth)", command=self.load_model).pack(fill="x", padx=5, pady=5)
        ttk.Button(train_frame, text="4. Pokaż Macierz Pomyłek", command=self.show_confusion_matrix_safe).pack(fill="x", padx=5, pady=5)

    def update_config(self):
        """ Aktualizuje konfigurację w logice aplikacji (konwertując typy). """
        try:
            INT_KEYS = ['IMG_SIZE', 'BATCH_SIZE', 'CROP_SIZE', 'DILATION_KERNEL_SIZE']
            
            for name, var in self.param_vars.items():
                value = var.get()
                
                if name in INT_KEYS:
                    self.config[name] = int(value) 
                else:
                    self.config[name] = value 
                    
            self.app_logic.config = self.config
            

            self.app_logic.transform = self.app_logic._create_base_transform(self.config['IMG_SIZE'])
            
            return True
        except Exception as e:
            messagebox.showerror("Błąd Parametrów", f"Niepoprawne dane wejściowe: {e}")
            return False

    def start_training_thread(self):
        if not self.update_config(): return
        
        self.status_var.set("Przygotowanie do treningu...")
        is_augmenting = self.augment_var.get()
        is_cut_paste = self.cut_paste_var.get()

        threading.Thread(target=self._run_training, args=(is_augmenting,is_cut_paste)).start()

    def _update_status_threadsafe(self, message, is_error=False):
        if is_error:
            messagebox.showerror("Błąd", message)
            self.status_var.set("Gotowy. Błąd Treningu.")
        else:
            self.status_var.set(message)
            self.master.update_idletasks()

    def _run_training(self, is_augmenting, is_cut_paste):
        try:
            results = self.app_logic.train_full_model(
                self.epoch_var.get(), 
                self._update_status_threadsafe,
                is_augmenting=is_augmenting,
                is_cut_paste=is_cut_paste
            )
            
            if results:
                train_losses, test_losses, test_accuracies, test_tensors, test_labels = results
                self.test_tensors_for_cm = test_tensors
                self.test_labels_for_cm = test_labels
                

                self.after_idle_call(self.show_learning_curve, train_losses, test_losses, test_accuracies)
                

            self.status_var.set("Trening zakończony pomyślnie. (PatchCore + CNN)")
            self.update_image_display()
            if is_cut_paste and self.app_logic.total_synthetic_samples > 0:
                self.current_synthetic_index = 0
                self.after_idle_call(self.update_cut_paste_display)
        except Exception as e:
            self._update_status_threadsafe(f"Krytyczny błąd podczas treningu: {e}", is_error=True)

    def save_model(self):
        if not self.app_logic.classifier:
            messagebox.showwarning("Ostrzeżenie", "Brak wytrenowanego klasyfikatora CNN.")
            return
        
        file_path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
        if file_path and self.app_logic.save_classifier(file_path):
            self.status_var.set(f"Model zapisano do: {file_path}")

    def load_model(self):
        file_path = filedialog.askopenfilename(defaultextension=".pth", filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            try:
                self.update_config() 
                self.app_logic.load_classifier(file_path)
                self.status_var.set(f"Model CNN wczytano pomyślnie z: {file_path}")
                self.update_image_display()
            except Exception as e:
                messagebox.showerror("Błąd Ładowania", f"Nie można wczytać modelu. Sprawdź CROP_SIZE i BATCH_SIZE lub plik. Błąd: {e}")
                self.status_var.set("Wczytywanie modelu nieudane.")

    
    def show_confusion_matrix_safe(self):
        """ Bezpieczne wywołanie macierzy pomyłek w głównym wątku. """
        self.after_idle_call(self._show_confusion_matrix)

    def _show_confusion_matrix(self):
        """ Wyświetla macierz pomyłek. """
        if self.test_tensors_for_cm is None or not self.app_logic.classifier:
            messagebox.showwarning("Ostrzeżenie", "Najpierw przeprowadź trening CNN, aby wygenerować dane testowe (Macierz Pomyłek).")
            return

        self.app_logic.classifier.eval()
        
        with torch.no_grad():
            outputs = self.app_logic.classifier(self.test_tensors_for_cm)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true = self.test_labels_for_cm.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            
            cm = confusion_matrix(y_true, y_pred, labels=np.arange(self.app_logic.num_classes)) 
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=self.app_logic.anomaly_classes, 
                yticklabels=self.app_logic.anomaly_classes
            )
            plt.title('Macierz Pomyłek (Confusion Matrix)')
            plt.ylabel('Rzeczywista Klasa')
            plt.xlabel('Przewidywana Klasa')
            plt.show()

    def show_learning_curve(self, train_losses, test_losses, test_accuracies):
        """ Wyświetla wykres krzywej uczenia. Wywoływane z głównego wątku. """
        epochs_range = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, test_losses, label='Test Loss')
        plt.title('Krzywa Straty (Loss) - Diagnostyka Overfittingu')
        plt.xlabel('Epoka')
        plt.ylabel('Loss (Cross Entropy)')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
        plt.title('Krzywa Dokładności (Accuracy)')
        plt.xlabel('Epoka')
        plt.ylabel('Dokładność')
        plt.legend()
        plt.grid(True)

        plt.show()

    def create_test_section(self, parent):
        test_frame = ttk.LabelFrame(parent, text="Wybór Klas Testowych & Nawigacja")
        test_frame.pack(padx=10, pady=10, fill="x")

        class_frame = ttk.Frame(test_frame)
        class_frame.pack(padx=5, pady=5)
        ttk.Label(class_frame, text="Testuj klasy:").pack(side=tk.LEFT)
        for class_name in self.app_logic.anomaly_classes:
            ttk.Checkbutton(class_frame, text=class_name, variable=self.test_classes_var[class_name], command=self.load_test_samples).pack(side=tk.LEFT, padx=5)

        nav_frame = ttk.Frame(test_frame)
        nav_frame.pack(pady=5)
        
        ttk.Button(nav_frame, text="< Poprzedni", command=lambda: self.navigate(-1)).pack(side=tk.LEFT, padx=5)
        self.index_label = ttk.Label(nav_frame, text="0/0")
        self.index_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Następny >", command=lambda: self.navigate(1)).pack(side=tk.LEFT, padx=5)
        
    def load_test_samples(self):
        selected_classes = [c for c, var in self.test_classes_var.items() if var.get()]
        self.total_samples = self.app_logic.load_test_data(selected_classes)
        self.current_index = 0
        self.update_image_display()
        
    def navigate(self, direction):
        if self.total_samples == 0: return
        
        new_index = self.current_index + direction
        if 0 <= new_index < self.total_samples:
            self.current_index = new_index
            self.update_image_display()
        
    def create_image_display(self, parent):
        self.image_canvas = tk.Canvas(parent, height=200, bg='light gray')
        self.image_canvas.pack(padx=10, pady=10, fill="x")
        self.result_labels = {}
        
        labels_frame = ttk.Frame(parent)
        labels_frame.pack(padx=10, pady=5)

        labels = ["Score:", "Real Class:", "Predicted Class:"]
        for i, label_text in enumerate(labels):
            ttk.Label(labels_frame, text=label_text).grid(row=0, column=i*2, padx=10, sticky="w")
            self.result_labels[label_text] = ttk.Label(labels_frame, text="---", font=('Arial', 10, 'bold'))
            self.result_labels[label_text].grid(row=0, column=i*2 + 1, padx=5, sticky="w")

    def update_image_display(self):
        self.index_label.config(text=f"{self.current_index + 1}/{self.total_samples}")
        
        if self.total_samples == 0:
            self.image_canvas.delete("all")
            return

        img_tensor, img_path, real_label = self.app_logic.get_test_sample(self.current_index)
        
        if not self.app_logic.classifier:
            self._display_single_image(img_path, self.image_canvas)
            self.result_labels["Predicted Class:"].config(text="Model Niezainicjowany", foreground='red')
            self.result_labels["Score:"].config(text="---", foreground='black')
            self.result_labels["Real Class:"].config(text=real_label, foreground='black')
            return

        try:
            anomaly_map, score, predicted_label, cropped_img_bgr = self.app_logic.predict_sample(img_tensor, img_path)
            
            self._display_four_images(img_path, anomaly_map, cropped_img_bgr)
            
            self.result_labels["Score:"].config(text=f"{score:.4f}", foreground='blue')
            self.result_labels["Real Class:"].config(text=real_label, foreground='black')
            
            if predicted_label == real_label:
                 self.result_labels["Predicted Class:"].config(text=predicted_label, foreground='green')
            else:
                 self.result_labels["Predicted Class:"].config(text=predicted_label, foreground='red')
            
        except ValueError as e:
            self._display_single_image(img_path, self.image_canvas)
            self.result_labels["Predicted Class:"].config(text=str(e), foreground='red')

    def _display_single_image(self, img_path, canvas):
        canvas.delete("all")
        img = Image.open(img_path).convert("RGB")
        img = img.resize((200, 200))
        self.tk_img = ImageTk.PhotoImage(img)
        canvas.create_image(10, 10, anchor=tk.NW, image=self.tk_img)
        canvas.config(height=220)

    def _display_four_images(self, img_path, anomaly_map, cropped_img_bgr):
        canvas = self.image_canvas
        canvas.delete("all")
        canvas_height = 200
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 800
        img_display_size = 180 
        
        img_original = cv2.imread(img_path)
        img_resized = cv2.resize(img_original, (int(self.config['IMG_SIZE']), int(self.config['IMG_SIZE']))) 
        img_resized_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        exponent = 4.0 
        scaled_map = anomaly_map ** exponent
        scaled_map = (scaled_map - scaled_map.min()) / (scaled_map.max() - scaled_map.min() + 1e-5) 

        heatmap = cv2.applyColorMap((scaled_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(img_resized_rgb, 0.6, heatmap_rgb, 0.4, 0)
        
        cropped_anomaly_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB) if cropped_img_bgr is not None else np.zeros((int(self.config['CROP_SIZE']), int(self.config['CROP_SIZE']), 3), dtype=np.uint8)


        images_to_display = {
            "Overlay": blended,
            "Heatmap": heatmap_rgb,
            "Crop": cropped_anomaly_rgb,
            "Original": img_resized_rgb
        }
        
        self.tk_images = []
        x_offset = 10
        
        for title, img_data in images_to_display.items():
            img_pil = Image.fromarray(img_data)
            img_pil = img_pil.resize((img_display_size, img_display_size))
            
            tk_img = ImageTk.PhotoImage(img_pil)
            self.tk_images.append(tk_img)
            
            canvas.create_image(x_offset, 10, anchor=tk.NW, image=tk_img)
            canvas.create_text(x_offset + img_display_size / 2, 10 + img_display_size + 10, 
                               text=title, font=('Arial', 8, 'bold'))
            
            x_offset += img_display_size + 20 

        canvas.config(height=canvas_height + 30)


if __name__ == '__main__':
    root = tk.Tk()
    app = AnomalyGUI(root)
    root.mainloop()
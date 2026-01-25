import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from patchcore_logic import PatchCore
from models import ResNet50_Extractor, AnomalyClassifierCNN
from PIL import Image
import torch.nn as nn
import numpy as np
import cv2
import random
from utils import apply_cut_and_paste


#datasety 
class SimpleCNNDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # Lista krotek (PIL Image, label_idx)
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_pil, label = self.samples[idx]
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label

class SimplePatchCoreDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_path, label

#główna logika backendu
class AnomalyDetectionApp:
    def __init__(self, device="cpu"):
        self.device = device
        self.extractor = None
        self.patchcore = None
        self.classifier = None
        self.patchcore_ready = False
        self.test_samples = []
        self.model_trained_with_crops = False
        
        self.anomaly_classes = ["crack", "cut", "hole", "print"]
        self.num_classes = len(self.anomaly_classes)

    def get_transform(self, size):
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def train_pipeline(self, config, status_callback):
        if not self.patchcore_ready:
            raise Exception("Najpierw załaduj PatchCore!")

        # Pobieranie parametrów z konfiguracji
        aug_factor = int(config.get('aug_factor', 1))
        use_cutpaste = config.get('aug_cutpaste', False)
        use_flips = config.get('aug_flips', False)
        use_cropping = config.get('use_cropping', False)
        
        crop_size = int(config.get('crop_size', 128))
        image_size = int(config.get('img_size', 224))
        threshold = config.get('aug_seg_ratio',190)
        dilation = config.get('aug_dilation',190)
        # Rozmiar docelowy zależy od wybranego trybu (czy chcemy wycinać anomalie)
        target_size = crop_size if use_cropping else image_size

        # Transformacje
        pc_trans = self.get_transform(512)
        cnn_trans = self.get_transform(target_size)

        # ścieżki do datasetu
        base_test_path = "./dataset/test"
        good_dir = "./dataset/train/good"
        good_files = [os.path.join(good_dir, f) for f in os.listdir(good_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # zbieramy wszystkie pliki z datasetu z anomaliami
        all_file_paths = [] 
        for idx, class_name in enumerate(self.anomaly_classes):
            class_dir = os.path.join(base_test_path, class_name)
            if not os.path.exists(class_dir): continue
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg'))]
            for f in files:
                all_file_paths.append((f, idx))

        if not all_file_paths:
            raise Exception("Brak zdjęć w dataset/test!")

        # dzielimy train/test
        random.shuffle(all_file_paths)
        split_idx = int(len(all_file_paths) * (1 - config['test_size']))
        train_paths = all_file_paths[:split_idx]
        test_paths = all_file_paths[split_idx:]

        # przygotowanie danych treningowych
        train_samples = []
        for img_path, idx in train_paths:
            img_pil = Image.open(img_path).convert("RGB").resize((512, 512))
            
            # generujemy mape anomalii jesli potrzeba (wycinanie anomalii do CNN lub  augmentacja cut-paste)
            amap = None
            if use_cropping or use_cutpaste:
                img_t_512 = pc_trans(img_pil)
                amap, _ = self.patchcore.predict(img_t_512, self.extractor, 512)

            # petla augmentacji 
            for i in range(aug_factor):
                current_img = img_pil.copy()
                current_amap = amap.copy() if amap is not None else None
                
                # augmentacja podstawowa
                if use_flips and i > 0:
                    if random.random() > 0.5:
                        current_img = current_img.transpose(Image.FLIP_LEFT_RIGHT)
                        if current_amap is not None:
                            current_amap = np.fliplr(current_amap)
                            
                    if random.random() > 0.5:
                        current_img = current_img.transpose(Image.FLIP_TOP_BOTTOM)
                        if current_amap is not None:
                            current_amap = np.flipud(current_amap)
                    
                # augmentacja cut-paste
                if use_cutpaste and good_files and i > 0:
                    target_pil = Image.open(random.choice(good_files)).convert("RGB").resize((512, 512))
                  
                    current_img = apply_cut_and_paste(current_img, target_pil, amap, threshold, dilation)
                    
                    # odswiezamy mape dla nwoego obrazu
                    tmp_t = pc_trans(current_img)
                    current_amap, _ = self.patchcore.predict(tmp_t, self.extractor, 512)

                # przetwarzanie finalne 
                if use_cropping:
                    processed_img, _ = self.get_anomaly_crop(current_img, current_amap, crop_size)
                else:
                    processed_img = current_img.resize((target_size, target_size), Image.LANCZOS)
                
                train_samples.append((processed_img, idx))

        # przygotowanie danych testowych
        val_samples = []
        for img_path, idx in test_paths:
            img_pil = Image.open(img_path).convert("RGB").resize((512, 512))
            
            if use_cropping:
                img_t_512 = pc_trans(img_pil)
                amap, _ = self.patchcore.predict(img_t_512, self.extractor, 512)
                processed_img, _ = self.get_anomaly_crop(img_pil, amap, crop_size)
            else:
                processed_img = img_pil.resize((target_size, target_size), Image.LANCZOS)
            
            val_samples.append((processed_img, idx))

        self.test_samples = val_samples
        self.model_trained_with_crops = use_cropping

        # loadujemy dataset w dataloader
        train_ds = SimpleCNNDataset(train_samples, cnn_trans)
        val_ds = SimpleCNNDataset(val_samples, cnn_trans)
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
        
        return self.train_cnn(train_loader, val_loader, config['epochs'], status_callback=status_callback)

    def train_cnn(self, train_loader, val_loader, epochs, lr=0.001, status_callback=None):
        self.classifier = AnomalyClassifierCNN(num_classes=self.num_classes).to(self.device)
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            self.classifier.train()
            running_loss, train_correct, train_total = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.classifier(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # walidacja
            self.classifier.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = self.classifier(imgs)
                    val_loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            metrics = {
                'train_loss': running_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'train_acc': 100 * train_correct / train_total,
                'val_acc': 100 * val_correct / val_total
            }
            for k, v in metrics.items(): history[k].append(v)

            if status_callback:
                status_callback(epoch + 1, epochs, round(metrics['train_loss'], 4), f"V_Acc: {metrics['val_acc']:.1f}%")

        return history

    def get_anomaly_crop(self, img_pil, anomaly_map, crop_size=128):
        img_np = np.array(img_pil)
        h, w = anomaly_map.shape
        _, _, _, max_loc = cv2.minMaxLoc(anomaly_map)
        x_c, y_c = max_loc

        x1 = max(0, x_c - crop_size // 2)
        y1 = max(0, y_c - crop_size // 2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        if x2 == w: x1 = max(0, w - crop_size)
        if y2 == h: y1 = max(0, h - crop_size)

        crop = img_np[y1:y2, x1:x2]
        return Image.fromarray(crop), (x1, y1, x2, y2)

    def load_patchcore_to_memory(self, sampling_ratio, status_callback):
        try:
            self.patchcore_ready = False
            self.extractor = ResNet50_Extractor().to(self.device).eval()
            self.patchcore = PatchCore(sampling_ratio=sampling_ratio, device=self.device)
            
            good_dir = "./dataset/train/good"
            good_samples = [(os.path.join(good_dir, f), 0) for f in os.listdir(good_dir) if f.lower().endswith(('.png', '.jpg'))]
            
            ds = SimplePatchCoreDataset(good_samples, self.get_transform(512))
            loader = DataLoader(ds, batch_size=8, shuffle=False)
            
            status_callback("Budowanie banku pamięci...")
            self.patchcore.build_memory(loader, self.extractor)
            self.patchcore_ready = True
            return True
        except Exception as e:
            status_callback(f"Błąd: {str(e)}", is_error=True)
            return False
        
    def get_patchcore_prediction(self, img_path):
        """ Metoda pomocnicza używana przez GUI do pobierania mapy anomalii """
        if not self.patchcore_ready:
            return None, 0.0
            
        # ladowanie i preprocessing obrazu do rozmiaru wymaganego do patchcore (512)
        img_pil = Image.open(img_path).convert("RGB")
        # transformujemy do 512x512
        img_tensor = self.get_transform(512)(img_pil)
        
        # predykcja mapy i wynik anomalii (score)
        # anomaly_map to heatmapa, score to najwyzsza wartosc anomalii
        anomaly_map, score = self.patchcore.predict(img_tensor, self.extractor, 512)
        return anomaly_map, score
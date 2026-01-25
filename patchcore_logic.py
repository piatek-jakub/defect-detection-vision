import torch
import torch.nn.functional as F
import cv2
import numpy as np


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
        feats = feats.permute(0, 2, 3, 1).reshape(-1, feats.size(1)) # rozmiar [N, D]
        
        batch_size = 1024
        min_dists = []
        
        for i in range(0, feats.size(0), batch_size):
            batch_feats = feats[i:i + batch_size].to(self.device)
            # dystans euklidesowy do banku pamięci
            dist_batch = torch.cdist(batch_feats, self.memory_bank)
            min_dist_batch, _ = torch.min(dist_batch, dim=1)
            min_dists.append(min_dist_batch.cpu())
            
        min_dist = torch.cat(min_dists)

        anomaly_map = min_dist.reshape(img_size // 8, img_size // 8) 
        anomaly_map = cv2.resize(anomaly_map.numpy(), (img_size, img_size))

        score = anomaly_map.max()
        return anomaly_map, float(score)

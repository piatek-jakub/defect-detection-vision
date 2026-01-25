import torch
import torch.nn as nn
from torchvision import models

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

class AnomalyClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # resnet18 z transfer learningiem
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # podmieniamy ostatnia warstwe na liczbe klas w datasecie hazelnuta
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)
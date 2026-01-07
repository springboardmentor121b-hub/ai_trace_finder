import torch
import torch.nn as nn
import cv2
import numpy as np

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 64 * 64, 128)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def extract_feature(model, img_path, device):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not readable")

    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        feat = model(img)

    return feat.cpu().numpy().flatten()

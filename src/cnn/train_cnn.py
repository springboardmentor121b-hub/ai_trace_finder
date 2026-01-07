import os
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from cnn_model import CNNModel

DATA_PATH = "Data_Sets"
MODEL_PATH = "models/cnn_model.pth"

transform = T.Compose([
    T.Grayscale(),
    T.Resize((256, 256)),
    T.ToTensor()
])

dataset = ImageFolder(DATA_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = CNNModel(num_classes=len(dataset.classes))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    total_loss = 0
    for imgs, labels in loader:
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("CNN model saved")
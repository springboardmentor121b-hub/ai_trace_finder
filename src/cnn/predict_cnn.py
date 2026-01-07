import torch
import torchvision.transforms as T
from PIL import Image
from cnn_model import CNNModel
from torchvision.datasets import ImageFolder

DATA_PATH = "Data_Sets"

transform = T.Compose([
    T.Grayscale(),
    T.Resize((256, 256)),
    T.ToTensor()
])

dataset = ImageFolder(DATA_PATH, transform=transform)
model = CNNModel(num_classes=len(dataset.classes))
model.load_state_dict(torch.load("models/cnn_model.pth"))
model.eval()

img_path = input("Enter image path: ")

img = Image.open(img_path)
x = transform(img).unsqueeze(0)

pred = model(x).argmax(1).item()
print("Predicted Class:", dataset.classes[pred])
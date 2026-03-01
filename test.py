import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys

# Load the model we trained
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("papillon_model.pth"))
model.eval()

# Prepare the photo
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load your photo (drag any dog photo to your papillon-classifier folder)
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0)

# Predict!
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    classes = ["Not a Papillon", "Papillon!"]
    confidence = torch.nn.functional.softmax(output, dim=1)[0]
    print(f"\nResult: {classes[predicted.item()]}")
    print(f"Confidence: {confidence[predicted.item()]*100:.1f}%")

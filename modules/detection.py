import torch
import torch.nn as nn
from torchvision import models, transforms

PATHOLOGIES = [
    "pneumonia",
    "cardiomegaly",
    "pleural_effusion",
    "pneumothorax",
    "atelectasis",
    "lung_mass"
]

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def build_model(num_classes=6):
    model = models.densenet121(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze denseblock3 + denseblock4 + norm5
    for param in model.features.denseblock3.parameters():
        param.requires_grad = True
    for param in model.features.denseblock4.parameters():
        param.requires_grad = True
    for param in model.features.norm5.parameters():
        param.requires_grad = True

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.25),
        nn.Linear(in_features, num_classes)
    )
    return model

def load_model(weights_path, device="cuda"):
    model = build_model()
    model.load_state_dict(torch.load(weights_path,
                                      map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict(model, image_path, device="cuda", threshold=0.3):
    from PIL import Image
    img    = Image.open(image_path).convert("RGB")
    tensor = TRANSFORM(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.sigmoid(logits).cpu().numpy()[0]

    results = {}
    for i, path in enumerate(PATHOLOGIES):
        results[path] = {
            "probability": float(probs[i]),
            "detected":    bool(probs[i] >= threshold)
        }
    return results
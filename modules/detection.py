import torch.nn as nn
from torchvision import models
from torchvision import transforms

MODEL_NAMES = ["resnet50", "densenet121", "efficientnet_b0", "mobilenet_v3"]

PATHOLOGIES = [
    "pneumonia", 
    "cardiomegaly", 
    "pleural_effusion",
    "pneumothorax", 
    "atelectasis", 
    "lung_mass"
]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2, 
        contrast=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def build_model(model_name="densenet121", num_classes=6):
    if model_name == "resnet50":
        model = models.resnet50(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(2048, num_classes)
        )

    elif model_name == "densenet121":
        model = models.densenet121(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features.denseblock3.parameters():
            param.requires_grad = True
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        for param in model.features.norm5.parameters():
            param.requires_grad = True
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1024, num_classes)
        )
    
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features[-3:].parameters():
            param.requires_grad = True
        
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, num_classes)
        )
    
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.features[-3:].parameters():
            param.requires_grad = True

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(960, num_classes)
        )

    return model
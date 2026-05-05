import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models

# EfficientNet-B0 expects 224x224, normalized to ImageNet stats
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# ImageFolder reads folder names as class labels, so ensure data/processed has subfolders for each class (e.g., pan/, cheque/)
dataset = datasets.ImageFolder("data/processed", transform=transform)
print(dataset.class_to_idx) 
# 80% train, 20% validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

# Load pre-trained EfficientNet-B0 and modify the classifier for 2 classes (pan vs cheque)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze all layers — only train the classifier head
for param in model.parameters():                                                                                                                                               
    param.requires_grad = False

# Replace final layer: 1000 ImageNet classes → 2 document classes
model.classifier[1] = nn.Linear(1280, 2)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()                                                                                                                                                  
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # Zero gradients from previous step
        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels)
        loss.backward() # Backward pass and update weights
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # No gradient computation during validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1) # Get predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    
    print(f"Validation Accuracy: {correct/total:.4f}")

# Save trained model weights
torch.save(model.state_dict(), "app/models/efficientnet_b0.pth")
print("Model saved")
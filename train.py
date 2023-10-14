# Imports
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse

# Parse command line arguments for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='flowers', help='Path to the dataset directory')
parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Path to save checkpoints')
parser.add_argument('--arch', type=str, default='vgg11', help='Pretrained model architecture (e.g., vgg11, resnet18)')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
args = parser.parse_args()

# Data preprocessing and augmentation
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

data_dir = args.data_dir
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Load a pretrained model
model = getattr(models, args.arch)(pretrained=True)

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Define a new classifier
classifier = nn.Sequential(
    nn.Linear(25088, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(args.hidden_units, len(train_data.class_to_idx)),
    nn.LogSoftmax(dim=1)
)

model.classifier = classifier

# Loss function and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Use GPU if available
device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(args.epochs):
    model.train()
    running_loss = 0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{args.epochs} - Loss: {running_loss/len(trainloader):.4f}")

# Save checkpoint
checkpoint = {
    'state_dict': model.state_dict(),
    'class_to_idx': train_data.class_to_idx,
    'classifier': model.classifier,
    'optimizer_state_dict': optimizer.state_dict(),
    'epochs': args.epochs
}
torch.save(checkpoint, args.save_dir)

print(f"Model checkpoint saved as '{args.save_dir}' with class_to_idx mapping attached to the model.")

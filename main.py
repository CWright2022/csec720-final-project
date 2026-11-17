# CSEC720 final project
#Fall 2026
#Cayden Wright, Eric Antonecchia

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define transformations (Resize is required since GTSRB image sizes vary)
baseline_resize_normalize = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))
])

# Download and load datasets
full_train_data = datasets.GTSRB(root='./data', split='train', download=True, transform=baseline_resize_normalize)
test_data = datasets.GTSRB(root='./data', split='test', download=True, transform=baseline_resize_normalize)

# Split training set into 80% train and 20% validation
train_size = int(0.8 * len(full_train_data))
val_size = len(full_train_data) - train_size
train_data, val_data = random_split(full_train_data, [train_size, val_size])

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
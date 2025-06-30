import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Paths
TRAIN_DIR = "Task_B/train"

# -----------------------------------
# 1. Dataset Pair Creation Function
# -----------------------------------
def get_image_paths(identity_path):
    """
    Returns a list of clean image paths from the identity folder.
    Ignores the distortion/ subfolder.
    """
    return [
        os.path.join(identity_path, f)
        for f in os.listdir(identity_path)
        if os.path.isfile(os.path.join(identity_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

def create_pairs(root_dir):
    print("🔄 Preparing dataset...")
    identities = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"🔍 Found {len(identities)} identity folders.")

    pairs = []
    labels = []

    for i in tqdm(range(len(identities))):
        id1 = identities[i]
        id1_images = get_image_paths(id1)
        if len(id1_images) < 2:
            continue

        # Positive pair
        img1, img2 = random.sample(id1_images, 2)
        pairs.append((img1, img2))
        labels.append(1)

        # Negative pair
        j = random.choice([x for x in range(len(identities)) if x != i])
        id2 = identities[j]
        id2_images = get_image_paths(id2)
        if len(id2_images) < 1:
            continue

        img3 = random.choice(id2_images)
        pairs.append((img1, img3))
        labels.append(0)

    print(f"✅ Created {len(pairs)} image pairs.")
    return pairs, labels

# -----------------------------------
# 2. Custom Pair Dataset
# -----------------------------------
class FacePairDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)

# -----------------------------------
# 3. Siamese Network
# -----------------------------------
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.embedding = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 256)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        out1 = self.embedding(x1)
        out2 = self.embedding(x2)
        diff = torch.abs(out1 - out2)
        return self.classifier(diff)

# -----------------------------------
# 4. Training
# -----------------------------------
if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    pairs, labels = create_pairs(TRAIN_DIR)
    dataset = FacePairDataset(pairs, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SiameseNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print("🚀 Starting training...")
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for (img1, img2), label in dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(img1, img2).squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (output > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}, Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "face_model.pth")
    print("✅ Model training complete and saved as face_model.pth")

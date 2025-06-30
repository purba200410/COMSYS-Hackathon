import torch.nn as nn

class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Flatten(),                             # Flattens to (batch, 3*224*224) = (batch, 150528)
            nn.Linear(150528, 256),                   # Match: 150528 → 256
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 1),                    # Match: 512 → 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.embedding(x)

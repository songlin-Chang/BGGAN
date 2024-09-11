import torch
import torch.nn as nn

class AttentionFusion(nn.Module):
    def __init__(self, input_size):
        super(AttentionFusion, self).__init__()

        # Linear layers for computing attention scores
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.fc3 = nn.Linear(input_size, 1)

        # Softmax to compute attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature1, feature2):
        # Compute attention scores
        scores1 = torch.tanh(self.fc1(feature1))
        scores2 = torch.tanh(self.fc2(feature2))

        # Bilinear Pooling to capture non-linear relationships
        bilinear_pooling = torch.matmul(scores1.unsqueeze(2), scores2.unsqueeze(1))

        # Flatten and pass through a linear layer to get attention weights
        scores = self.fc3(bilinear_pooling.view(bilinear_pooling.size(0), -1))

        # Apply softmax to get attention weights
        weights = self.softmax(scores)

        # Weighted sum of features
        fused_feature = weights.unsqueeze(2) * feature1 + weights.unsqueeze(1) * feature2

        return fused_feature

# Example usage:
# Assuming feature1 and feature2 are torch tensors with the same size
# feature_size = 256  # Adjust based on the actual feature size
# attention_fusion = AttentionFusion(input_size=feature_size)

# # Forward pass
# result = attention_fusion(feature1, feature2)

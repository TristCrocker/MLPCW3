import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50

class Detr(nn.Module):
    def __init__(self, num_queries, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Detr, self).__init__()

        #Backbone
        self.backbone = resnet50(pretrained=True)
        self.conv1x1 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        #Transformer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)

        #Embeddings
        self.embeddings = nn.Parameter(torch.rand(num_queries, hidden_dim))

        #Bounding Box & Class Prediction
        self.class_emb = nn.Linear(hidden_dim, 2)  # 2 classes, boat and background
        self.bbox_emb = nn.Linear(hidden_dim, 4)  # (cx, cy, w, h)

    def forward(self, images):
        #Extract
        features = self.backbone(images)

        features = features if isinstance(features, torch.Tensor) else features['layer4']

        features = self.conv1x1(features['layer4'])
        features = features.flatten(2).permute(2, 0, 1)

        #Pos Encoding
        pos_enc = torch.zeros_like(features)

        #Trans
        memory = self.transformer(features, self.embeddings.unsqueeze(1).repeat(1, features.size(1), 1) + pos_enc)

        #Preds
        class_logits = self.class_emb(memory)
        bbox = self.bbox_emb(memory).sigmoid()

        return {'Class (Logits)' : class_logits, 'Bbox' : bbox}


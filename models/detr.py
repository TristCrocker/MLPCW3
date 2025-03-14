import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor




class Detr(nn.Module):
    def __init__(self, num_queries, hidden_dim=256, num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Detr, self).__init__()

        #Backbone
        self.backbone = create_feature_extractor(resnet50(weights=ResNet50_Weights.IMAGENET1K_V1), return_nodes={'layer4': 'feature_map'}) 
        self.conv1x1 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        #Transformer
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, batch_first=True)

        #Embeddings
        self.embeddings = nn.Parameter(torch.rand(num_queries, hidden_dim))

        #Bounding Box & Class Prediction
        self.class_emb = nn.Linear(hidden_dim, 2)  # 2 classes, boat and background
        self.bbox_emb = nn.Linear(hidden_dim, 4)  # (cx, cy, w, h)

    def forward(self, images):
        #Extract
        features = self.backbone(images)

        features = self.conv1x1(features['feature_map'])
        features = features.flatten(2).permute(2, 0, 1)

        #Pos Encoding
        pos_enc = torch.zeros_like(features)

        #Trans
        batch_size, seq_len, _ = features.shape  # Extract correct shape
        query_embeds = self.embeddings[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1) 

        memory = self.transformer(features, query_embeds + pos_enc)

        #Preds
        class_logits = self.class_emb(memory)
        bbox = self.bbox_emb(memory).sigmoid()

        return {'pred_logits' : class_logits, 'Bbox' : bbox}


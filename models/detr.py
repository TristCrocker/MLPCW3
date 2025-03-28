import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class Detr(nn.Module):
    def __init__(self, num_classes=1, hidden_dim=256, num_queries=10, num_heads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(Detr, self).__init__()

        # Backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # Decrease size to iamge size
        self.conv1x1 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True 
        )

        self.class_head = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_head = nn.Linear(hidden_dim, 4)

        # Embeddings
        self.query_pos = nn.Parameter(torch.rand(num_queries, hidden_dim))

        # Positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, images):
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        h = self.conv1x1(x)

        H, W = h.shape[-2:]
        pos_enc = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(0) 

        h = h.flatten(2).permute(0, 2, 1)

        batch_size = images.shape[0]
        query_pos = self.query_pos.unsqueeze(0).repeat(batch_size, 1, 1)

        transformer_output = self.transformer(
            src=pos_enc + 0.1 * h,
            tgt=query_pos
        ) 

        pred_logits = self.class_head(transformer_output)
        pred_boxes = self.bbox_head(transformer_output).sigmoid()

        return {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}


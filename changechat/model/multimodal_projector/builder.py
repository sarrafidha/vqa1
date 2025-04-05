import torch
import torch.nn as nn
import re
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class SingleStreamAttention(nn.Module):
    def __init__(self, mm_hidden_size, hidden_size):
        super(SingleStreamAttention, self).__init__()
        self.d_model = mm_hidden_size

        encoder_self_layer_classifier = nn.TransformerEncoderLayer(
            2 * self.d_model, nhead=8, dim_feedforward=int(4 * self.d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_self_layer_classifier, num_layers=3
        )
        img_feature_w, img_feature_h = 18, 18
        self.w_embedding = nn.Embedding(img_feature_w, int(self.d_model / 2))
        self.h_embedding = nn.Embedding(img_feature_h, int(self.d_model / 2))
        self.classifier_head = nn.Linear(2 * self.d_model, 2)
        self.project_out = nn.Linear(2 * self.d_model, hidden_size)

        # cls_token
        scale = self.d_model**-0.5
        self.cls_changeflag = nn.Parameter(scale * torch.randn(1, 2 * self.d_model))

    def position_embedding_2D_func(self, img_feat_A, img_feat_B):
        device = img_feat_A.device
        batch = img_feat_B.shape[0]
        Len_feat = img_feat_B.shape[1]
        h = int(math.sqrt(Len_feat))
        w = h
        pos_w = torch.arange(w, device=device).to(device)
        pos_h = torch.arange(h, device=device).to(device)
        embed_w = self.w_embedding(pos_w)
        embed_h = self.h_embedding(pos_h)
        position_embedding = torch.cat(
            [
                embed_w.unsqueeze(0).repeat(h, 1, 1),
                embed_h.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        )
        position_embedding = position_embedding.unsqueeze(0).repeat(
            batch, 1, 1, 1
        )  # (batch, h, w, d_model)
        position_embedding = position_embedding.view(batch, -1, self.d_model)
        img_feat_A = img_feat_A + position_embedding  # NLD
        img_feat_B = img_feat_B + position_embedding  # NLD

        return img_feat_A, img_feat_B

    def forward(self, img_feat):
        img_feat_A = img_feat[:, 0, ...]  # (N,L,768)
        img_feat_B = img_feat[:, 1, ...]  # (N,L,768)

        # 2D image position_embedding
        img_feat_A, img_feat_B = self.position_embedding_2D_func(
            img_feat_A, img_feat_B
        )  # (N, L, D)

        img_feat = torch.cat([img_feat_A, img_feat_B], dim=-1)  # (N, L, 2D)
        img_feat_with_cls = torch.cat(
            [
                self.cls_changeflag.unsqueeze(0).expand(
                    img_feat.shape[0], *self.cls_changeflag.shape
                ),
                img_feat,
            ],
            dim=1,
        )

        img_feat_with_cls = self.transformer_encoder(
            img_feat_with_cls.permute(1, 0, 2)
        ).permute(
            1, 0, 2
        )  # (N, L, 2D)
        change_pred = self.classifier_head(img_feat_with_cls[:, 0, :])
        img_feat = img_feat_with_cls
        img_feat = self.project_out(img_feat)  # (N, L, D)
        return change_pred, img_feat


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    if projector_type == "single_stream_attention":
        return SingleStreamAttention(config.mm_hidden_size, config.hidden_size)

    raise ValueError(f"Unknown projector type: {projector_type}")

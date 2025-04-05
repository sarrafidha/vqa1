import torch
import torch.nn as nn
import math


class ChangeClassifier(nn.Module):
    def __init__(
        self, mm_hidden_size, hidden_size=None, img_feature_w=18, img_feature_h=18
    ):
        super(ChangeClassifier, self).__init__()
        self.d_model = mm_hidden_size

        encoder_self_layer_classifier = nn.TransformerEncoderLayer(
            2 * self.d_model, nhead=8, dim_feedforward=int(4 * self.d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_self_layer_classifier, num_layers=3
        )
        self.w_embedding = nn.Embedding(img_feature_w, int(self.d_model / 2))
        self.h_embedding = nn.Embedding(img_feature_h, int(self.d_model / 2))
        self.classifier_head = nn.Linear(2 * self.d_model, 2)
        if hidden_size:
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
        if hasattr(self, "project_out"):
            img_feat = img_feat_with_cls
            img_feat = self.project_out(img_feat)  # (N, L, D)
            return change_pred, img_feat
        return change_pred

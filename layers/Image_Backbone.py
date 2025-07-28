import timm
import torch
import torch.nn as nn

class EfficientNet(nn.Module):
    def __init__(self, image_size=67, num_feature=512, configs=None):
        super().__init__()
        self.configs = configs
        self.input_channels = [
            configs.seq_len,
            configs.seq_len // 2,
            configs.seq_len // 4,
            configs.seq_len // 8
        ]

        self.backbones = nn.ModuleList()
        for in_ch in self.input_channels:
            model = timm.create_model(
                'efficientnet_b2', pretrained=False,
                features_only=True, out_indices=[-1]
            )
            stem = model.conv_stem
            model.conv_stem = nn.Conv2d(
                in_ch, stem.out_channels,
                kernel_size=stem.kernel_size,
                stride=stem.stride,
                padding=stem.padding,
                bias=False
            )
            self.backbones.append(model)

        with torch.no_grad():
            dummy = torch.zeros(
                1, self.input_channels[0],
                image_size, image_size
            )
            feat = self.backbones[0](dummy)[0]  # [1, C, h, w]
            _, _, h, w = feat.shape
            spatial_dim = h * w

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(spatial_dim, in_ch),
                nn.ReLU()
            ) for in_ch in self.input_channels
        ])

        backbone_out_ch = feat.shape[1]  # 1280
        self.out_feature = (
            nn.Linear(backbone_out_ch, num_feature)
            if num_feature != backbone_out_ch
            else nn.Identity()
        )

    def forward(self, x_list):
        out_list = []
        for i, x in enumerate(x_list):
            feats = self.backbones[i](x)[0]  # [B, 1280, h, w]
            B, C, h, w = feats.shape
            flat = feats.view(B, C, h * w)  # [B,1280,spatial_dim]
            proj = self.projections[i](flat)  # [B,1280,input_channels[i]]
            proj = proj.permute(0, 2, 1)  # [B,input_channels[i],1280]
            out = self.out_feature(proj)  # [B,input_channels[i],num_feature]
            out_list.append(out)
        return out_list

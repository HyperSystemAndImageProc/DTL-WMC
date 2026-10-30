import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# MSCAttention（无门控版）
# -------------------------
class MSCAttention(nn.Module):
    def __init__(self, in_channels):
        super(MSCAttention, self).__init__()
        # ---- 通道依赖建模 (CCB) ----
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, 1, 1)
        self.qkv_proj = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)
        self.Relu = nn.ReLU()
        # ---- 多尺度空间上下文 (MSC) ----
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels)
        self.conv7d = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=6, dilation=2, groups=in_channels)
        self.conv_fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()

        # ---- 通道依赖 ----
        pooled = self.avg_pool(x).view(B, C)  # (B, C)
        q = self.qkv_proj(pooled)
        k = self.qkv_proj(pooled)
        attn = self.softmax(torch.bmm(q.unsqueeze(2), k.unsqueeze(1)))  # (B, C, C)
        channel_out = torch.matmul(attn, pooled.unsqueeze(-1)).view(B, C, 1, 1)
        channel_out = x * channel_out.expand_as(x)

        # ---- 多尺度空间上下文 ----
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7d = self.conv7d(x)

        # 对齐尺寸，防止concat报错
        Hf, Wf = f3.size(2), f3.size(3)
        f5 = F.interpolate(f5, size=(Hf, Wf), mode='bilinear', align_corners=False)
        f7d = F.interpolate(f7d, size=(Hf, Wf), mode='bilinear', align_corners=False)

        spatial_out = torch.cat([f3, f5, f7d], dim=1)
        spatial_out = self.conv_fuse(spatial_out)

        # ---- 融合通道依赖 + 空间上下文 ----
        out = channel_out + spatial_out   # 简单相加，不用门控
        out = self.Relu(out)+ x
        return out


# -------------------------
# GatedBottleneck
# -------------------------
class GatedBottleneck(nn.Module):
    def __init__(self, dim=256, reduction=16):
        super(GatedBottleneck, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.gate_bin = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
        self.gate_multi = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        shared_feat = self.shared(x)
        bin_feat = shared_feat * self.gate_bin(shared_feat)
        multi_feat = shared_feat * self.gate_multi(shared_feat)
        return bin_feat, multi_feat


# -------------------------
# 主网络
# -------------------------
class WaferCNN_Attn_MultiTask1(nn.Module):
    def __init__(self, input_channels=1, num_faults=8):
        super(WaferCNN_Attn_MultiTask1, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.attn0 = MSCAttention(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.attn1 = MSCAttention(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.attn2 = MSCAttention(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(2)
        self.attn3 = MSCAttention(256)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.shared_bottleneck = GatedBottleneck(256)

        self.binary_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

        self.multi_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_faults)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.attn0(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.attn1(x)

        x = self.pool2(F.relu(self.bn3(self.conv3(x))))
        x = self.attn2(x)

        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = self.attn3(x)

        x = self.gap(x).view(x.size(0), -1)

        bin_feat, multi_feat = self.shared_bottleneck(x)

        bin_logit = self.binary_head(bin_feat).squeeze(1)
        multi_logits = self.multi_head(multi_feat)
        return bin_logit, multi_logits


# -------------------------
# 测试
# -------------------------
if __name__ == '__main__':
    model = WaferCNN_Attn_MultiTask1(input_channels=1, num_faults=8)
    dummy_input = torch.randn(2, 1, 64, 64)
    bin_out, multi_out = model(dummy_input)
    print("二分类输出:", bin_out.shape)
    print("多分类输出:", multi_out.shape)

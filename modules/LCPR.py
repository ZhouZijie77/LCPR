import math
import copy
import torch
from torch import nn
from modules.netvlad import NetVLADLoupe
from torchvision.models.resnet import resnet18


class LCPR(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv1_i = pretrained.conv1
        self.bn1_i = pretrained.bn1
        self.relu = pretrained.relu
        self.maxpool = pretrained.maxpool
        self.conv_l = nn.Conv2d(1, 64, kernel_size=1)
        self.layer1_i = pretrained.layer1
        self.layer2_i = pretrained.layer2
        self.layer2_l = copy.deepcopy(pretrained.layer2)
        self.layer3_i = pretrained.layer3
        self.layer3_l = copy.deepcopy(pretrained.layer3)
        self.layer4_i = pretrained.layer4
        self.layer4_l = copy.deepcopy(pretrained.layer4)
        self.fusion1 = fusion_block(in_channels=64, mid_channels=32, out_channels=64, num_cam=6)
        self.fusion2 = fusion_block(in_channels=128, mid_channels=64, out_channels=128, num_cam=6)
        self.fusion3 = fusion_block(in_channels=256, mid_channels=128, out_channels=256, num_cam=6)
        self.fusion4 = fusion_block(in_channels=512, mid_channels=256, out_channels=512, num_cam=6)
        self.v_conv_i = VertConv(in_channels=512, mid_channels=256, out_channels=256)
        self.v_conv_l = VertConv(in_channels=512, mid_channels=256, out_channels=256)
        self.netvlad_i = NetVLADLoupe(feature_size=256, max_samples=132, cluster_size=32, output_dim=128)
        self.netvlad_l = NetVLADLoupe(feature_size=256, max_samples=132, cluster_size=32, output_dim=128)
        self.eca = eca_layer(256)

    def forward(self, x_i, x_l):
        # x_i: B x N x C x Hi x Wi
        # x_l: B x C x Hl x Wl
        B, N, C, Hi, Wi = x_i.shape
        x_i = x_i.view(B * N, C, Hi, Wi)
        x_i = self.conv1_i(x_i)
        x_i = self.bn1_i(x_i)
        x_i = self.relu(x_i)
        x_i = self.maxpool(x_i)
        x_i = self.layer1_i(x_i)

        x_l = self.conv_l(x_l)
        x_i_1, x_l_1 = self.fusion1(x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = self.layer2_i(x_i)
        x_l = self.layer2_l(x_l)
        x_i_1, x_l_1 = self.fusion2(x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = self.layer3_i(x_i)
        x_l = self.layer3_l(x_l)
        x_i_1, x_l_1 = self.fusion3(x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = self.layer4_i(x_i)
        x_l = self.layer4_l(x_l)
        x_i_1, x_l_1 = self.fusion4(x_i, x_l)
        x_i = x_i + x_i_1
        x_l = x_l + x_l_1

        x_i = paronamic_concat(x_i, N=6)  # B x C x Hi X NWi

        x_i = self.v_conv_i(x_i)  # B x C x NWi
        x_i = x_i.unsqueeze(2)  # B x C x 1 x NWi
        x_l = self.v_conv_l(x_l)  # B x C x Wl
        x_l = x_l.unsqueeze(2)  # B x C x 1 x Wl

        x_i = self.netvlad_i(x_i)
        x_l = self.netvlad_l(x_l)
        x = torch.cat((x_i, x_l), dim=-1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.eca(x)
        x = x.squeeze(-1).squeeze(-1)
        x = nn.functional.normalize(x, dim=-1)

        return x

    @classmethod
    def create(cls, weights=None):
        if weights is not None:
            pretrained = resnet18(weights=weights)
        else:
            pretrained = resnet18()
        model = cls(pretrained)
        return model


class eca_layer(nn.Module):

    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def paronamic_concat(x, N):
    # x: BN x C x H x W
    BN, C, H, W = x.shape
    B = int(BN / N)
    x = x.view(B, N, C, H, W)
    x = x.permute(0, 2, 3, 1, 4)  # B x C x H x N x W
    x = x.reshape(B, C, H, N * W)  # B x C x H x NW
    return x


def paronamic_concat_inv(x, N):
    # x: B x C x H x NW
    B, C, H, NW = x.shape
    W = int(NW / N)
    x = x.view(B, C, H, N, W)
    x = x.permute(0, 3, 1, 2, 4)  # B x N x C x H x W
    x = x.reshape(B * N, C, H, W)
    return x


class fusion_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_cam):
        super().__init__()
        self.num_cam = num_cam
        self.v_conv_i = VertConv(in_channels, mid_channels, out_channels)
        self.v_conv_l = VertConv(in_channels, mid_channels, out_channels)
        self.atten = MultiHeadAttention(d_model=out_channels, n_head=4)

    def forward(self, x_i, x_l):
        # x_i: BN x C x H x Wi
        _, _, Hi, Wi = x_i.shape
        _, _, Hl, Wl = x_l.shape
        x_i = paronamic_concat(x_i, self.num_cam)
        x_i = self.v_conv_i(x_i)  # B x C x NWi
        x_l = self.v_conv_l(x_l)  # B x C x Wl
        x = torch.cat((x_i, x_l), dim=-1)  # B x C x (NWi+Wl)
        x = x.transpose(1, 2)  # B x (NWi+Wl) x C
        x = x + self.atten(x, x, x)
        x = x.transpose(1, 2)  # B x C x (NWi+Wl)
        x_i, x_l = torch.split(x, [self.num_cam * Wi, Wl], dim=-1)

        x_i = x_i.unsqueeze(2)
        x_i = x_i.expand(-1, -1, Hi, -1)
        x_i = paronamic_concat_inv(x_i, self.num_cam)

        x_l = x_l.unsqueeze(2)
        x_l = x_l.expand(-1, -1, Hl, -1)
        return x_i, x_l


class VertConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)
        )

        self.reduce_conv = nn.Sequential(
            nn.Conv1d(
                in_channels,
                mid_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(
                mid_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=True,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = x.max(2)[0]
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):

        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class ScaleDotProductAttention(nn.Module):

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        k_t = k.transpose(2, 3)
        score = (q @ k_t) / math.sqrt(d_tensor)

        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        score = self.softmax(score)

        v = score @ v

        return v, score

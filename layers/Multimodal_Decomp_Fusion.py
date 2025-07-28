import numpy as np
from layers.Autoformer_EncDec import series_decomp
from layers.SoftmaxFusion import *
from layers.SelfAttention_Family import SelfAttention

class SlidingPCA(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: [B, T, C]
        Returns:
            season: [B, T, C]
            trend:  [B, T, C]
        """
        B, T, C = x.shape
        trend = torch.zeros_like(x)

        window_size = T

        for i in range(0, T - window_size + 1, self.stride):
            x_slice = x[:, i:i + window_size, :]  # [B, W, C]
            x_mean = x_slice.mean(dim=1, keepdim=True)  # [B, 1, C]
            x_centered = x_slice - x_mean

            u, s, v = torch.svd(x_centered)
            v1 = v[:, :, :1]  # [B, C, 1]
            proj = torch.matmul(x_centered, v1)  # [B, W, 1]
            x_recon = proj @ v1.transpose(-1, -2) + x_mean  # [B, W, C]

            trend[:, i:i + window_size, :] += x_recon

        count = torch.zeros_like(x[:, :, 0])  # [B, T]
        for i in range(0, T - window_size + 1, self.stride):
            count[:, i:i + window_size] += 1

        trend = trend / count.unsqueeze(-1).clamp(min=1.0)
        season = x - trend
        return season, trend


class SlidingKernelPCA(nn.Module):
    def __init__(self, window=24, stride=1, n_basis=4, kernel='cosine'):
        super().__init__()
        self.window = window
        self.stride = stride
        self.n_basis = n_basis
        self.kernel = kernel
        self.register_buffer('centroids', torch.randn(n_basis, 1, 1))  # init later with data

    def compute_kernel(self, x_centered):  # [B, W, C]
        # Compute kernel similarity between [B, W, C] and self.centroids: [n_basis, 1, 1]
        if self.kernel == 'cosine':
            x_norm = F.normalize(x_centered, dim=-1)  # [B, W, C]
            c_norm = F.normalize(self.centroids, dim=0)  # [n_basis, 1, 1]
            sim = torch.einsum('bwc,noc->bnw', x_norm, c_norm.expand(-1, x_centered.shape[1], x_centered.shape[2]))
        else:  # default to Gaussian
            dist = ((x_centered.unsqueeze(1) - self.centroids.view(self.n_basis, 1, 1)) ** 2).sum(dim=-1)
            sim = torch.exp(-dist / 2.0)
        return sim  # [B, n_basis, W]

    def forward(self, x):
        B, T, C = x.shape
        trend = torch.zeros_like(x)
        count = torch.zeros(B, T, device=x.device)

        for i in range(0, T - self.window + 1, self.stride):
            x_slice = x[:, i:i + self.window, :]  # [B, W, C]
            x_mean = x_slice.mean(dim=1, keepdim=True)
            x_centered = x_slice - x_mean

            sim = self.compute_kernel(x_centered)  # [B, n_basis, W]
            weights = F.softmax(sim, dim=1)  # soft attention over basis

            # trend = weighted avg over time using basis
            weighted = torch.einsum('bnw,bwc->bwc', weights, x_centered)
            trend_fit = weighted + x_mean  # [B, W, C]

            trend[:, i:i + self.window, :] += trend_fit
            count[:, i:i + self.window] += 1

        trend = trend / count.unsqueeze(-1).clamp(min=1.0)
        season = x - trend
        return season, trend


class SlidingMTE(nn.Module):
    def __init__(self, stride=1):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        B, T, C = x.shape

        window_size = T // 2

        pad_len = window_size // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        x = torch.cat([front, x, end], dim=1)

        B, T, C = x.shape

        trend = torch.zeros_like(x)
        count = torch.zeros_like(x[:, :, 0])  # [B, T]

        center = (window_size - 1) / 2
        distances = torch.abs(torch.arange(window_size) - center)  # [W]
        weights = (center + 1) - distances
        weights = weights.clamp(min=0.0).unsqueeze(0).unsqueeze(-1).to(x.device)  # [1, W, 1]

        for i in range(0, T - window_size + 1, self.stride):
            x_slice = x[:, i:i + window_size, :]  # [B, W, C]
            weighted = x_slice * weights  # [B, W, C]
            trend[:, i:i + window_size, :] += weighted
            count[:, i:i + window_size] += weights.squeeze(-1)

        trend = trend / count.unsqueeze(-1).clamp(min=1.0)
        season = x - trend

        trend = trend[:, pad_len:-pad_len, :]
        season = season[:, pad_len:-pad_len, :]

        return season, trend


class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiWaveletDecomposition(nn.Module):
    def __init__(self, in_channels, k=8, hidden_dim=64, base='legendre',
                 fusion_weights=None, fusion_learnable=True):
        super().__init__()
        self.in_channels = in_channels
        self.k = k

        self.proj_in = nn.Linear(in_channels, hidden_dim * k)
        self.proj_out = nn.Linear(hidden_dim, in_channels)

        self.filter = self._init_wavelet_basis(base, k)
        self.register_buffer("wavelet_basis", self.filter)  # [2k, k]

        self.fusion_season = SoftmaxFusion(num_inputs=k, weights=fusion_weights, learnable=fusion_learnable)
        self.fusion_trend = SoftmaxFusion(num_inputs=k, weights=fusion_weights, learnable=fusion_learnable)

    def _init_wavelet_basis(self, base, k):
        from scipy.special import eval_legendre
        if base == 'legendre':
            grid = np.linspace(-1, 1, k)
            basis = [eval_legendre(i, grid) for i in range(2 * k)]
            basis = np.stack(basis, axis=0)  # [2k, k]
        else:
            raise NotImplementedError(f"wavelet base '{base}' not supported.")
        return torch.tensor(basis, dtype=torch.float32)

    def forward(self, x):
        B, T, C = x.shape
        x_proj = self.proj_in(x).view(B, T, -1, self.k)  # [B, T, H, k]
        x_wavelet = torch.matmul(x_proj, self.wavelet_basis.T)  # [B, T, H, 2k]

        if x_wavelet.shape[-1] < 2 * self.k:
            raise ValueError(f"x_wavelet shape error: expected last dim 2k={2 * self.k}, got {x_wavelet.shape[-1]}")

        season_part = [x_wavelet[..., i] for i in range(self.k)]
        trend_part = [x_wavelet[..., i] for i in range(self.k, 2 * self.k)]

        season = self.fusion_season(season_part)  # [B, T, H]
        trend = self.fusion_trend(trend_part)  # [B, T, H]

        season = self.proj_out(season)  # [B, T, C]
        trend = self.proj_out(trend)  # [B, T, C]

        return season, trend


def check_decreasing_tensor_size(tensor_list):
    previous_numel = None

    for i, tensor in enumerate(tensor_list):
        current_numel = tensor.numel()

        if previous_numel is not None and current_numel > previous_numel:
            return False

        previous_numel = current_numel

    return True


class MultiScaleUpMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, configs):
        super(MultiScaleUpMixing, self).__init__()

        self.configs = configs

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        ),

                )
                for i in range(configs.down_sampling_layers)
            ]
        )
        self.fusion = SoftmaxFusion(num_inputs=2, learnable=True)

    def forward(self, season_list):
        is_decreasing = check_decreasing_tensor_size(season_list)

        down_sampling_layers_conv = nn.ModuleList([
            nn.Sequential(
                MultiScaleConvBlock(self.configs.d_model, kernel_sizes=[3, 5, 7],
                                    stride=self.configs.down_sampling_window),
                nn.GELU(),
                nn.Conv1d(self.configs.d_model, self.configs.d_model, kernel_size=1)
            ) for _ in range(self.configs.down_sampling_layers)
        ]).to("cuda:0")

        if is_decreasing:
            # mixing high->low
            out_high = season_list[0]
            out_low = season_list[1]
            out_season_list = [out_high.permute(0, 2, 1)]

            for i in range(len(season_list) - 1):
                out_low_res1 = down_sampling_layers_conv[i](out_high)
                out_low_res2 = self.down_sampling_layers[i](out_high)
                out_low_res = self.fusion([out_low_res1, out_low_res2])

                out_low = out_low + out_low_res
                out_high = out_low

                if i + 2 <= len(season_list) - 1:
                    out_low = season_list[i + 2]

                out_season_list.append(out_high.permute(0, 2, 1))

            return out_season_list
        else:
            season_list_slice = [
                season_list[:len(season_list) // 3],
                season_list[len(season_list) // 3:2 * len(season_list) // 3],
                season_list[2 * len(season_list) // 3:]
            ]

            out_season_list = []

            for season_list_part in season_list_slice:
                # mixing high->low
                out_high = season_list_part[0]
                out_low = season_list_part[1]
                out_season_list.append(out_high.permute(0, 2, 1))

                for i in range(len(season_list_part) - 1):
                    out_low_res1 = down_sampling_layers_conv[i](out_high)
                    out_low_res2 = self.down_sampling_layers[i](out_high)
                    out_low_res = self.fusion([out_low_res1, out_low_res2])

                    out_low = out_low + out_low_res
                    out_high = out_low
                    if i + 2 <= len(season_list_part) - 1:
                        out_low = season_list_part[i + 2]
                    out_season_list.append(out_high.permute(0, 2, 1))

            return out_season_list


class MultiScaleDownMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, configs):
        super(MultiScaleDownMixing, self).__init__()

        self.configs = configs

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** (i + 1)),
                        configs.seq_len // (configs.down_sampling_window ** i),
                        ),
                    nn.GELU(),
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.seq_len // (configs.down_sampling_window ** i),
                        ),
                )
                for i in reversed(range(configs.down_sampling_layers))
            ])
        self.fusion = SoftmaxFusion(num_inputs=2, learnable=True)

    def forward(self, trend_list):
        is_increasing = check_decreasing_tensor_size(trend_list)

        up_sampling_layers_conv = nn.ModuleList([
            nn.Sequential(
                MultiScaleUpConvBlock(self.configs.d_model, [3, 5, 7], self.configs.down_sampling_window),
                nn.GELU(),
                nn.Conv1d(self.configs.d_model, self.configs.d_model, 1)
            ) for _ in range(self.configs.down_sampling_layers)
        ]).to("cuda:0")

        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()

        if is_increasing:
            out_low = trend_list_reverse[0]
            out_high = trend_list_reverse[1]
            out_trend_list = [out_low.permute(0, 2, 1)]

            for i in range(len(trend_list_reverse) - 1):
                out_high_res1 = up_sampling_layers_conv[i](out_low)
                out_high_res2 = self.up_sampling_layers[i](out_low)
                out_high_res = self.fusion([out_high_res1, out_high_res2])

                out_high = out_high + out_high_res
                out_low = out_high
                if i + 2 <= len(trend_list_reverse) - 1:
                    out_high = trend_list_reverse[i + 2]
                out_trend_list.append(out_low.permute(0, 2, 1))

            out_trend_list.reverse()
            return out_trend_list
        else:
            trend_list_slice = [
                trend_list_reverse[:len(trend_list_reverse) // 3],
                trend_list_reverse[len(trend_list_reverse) // 3:2 * len(trend_list_reverse) // 3],
                trend_list_reverse[2 * len(trend_list_reverse) // 3:]
            ]

            out_trend_list = []

            for trend_list_part in trend_list_slice:
                # mixing low->high
                out_low = trend_list_part[0]
                out_high = trend_list_part[1]
                out_trend_list.append(out_low.permute(0, 2, 1))

                for i in range(len(trend_list_part) - 1):
                    out_high_res1 = up_sampling_layers_conv[i](out_low)
                    out_high_res2 = self.up_sampling_layers[i](out_low)
                    out_high_res = self.fusion([out_high_res1, out_high_res2])

                    out_high = out_high + out_high_res
                    out_low = out_high
                    if i + 2 <= len(trend_list_part) - 1:
                        out_high = trend_list_part[i + 2]
                    out_trend_list.append(out_low.permute(0, 2, 1))

            out_trend_list.reverse()

            return out_trend_list


class MultiScaleConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list, stride: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(channels, channels, k, stride=stride, padding=k // 2)
            for k in kernel_sizes
        ])
        self.fusion = SoftmaxFusion(num_inputs=len(kernel_sizes), learnable=True)

    def forward(self, x):
        outputs = [conv(x) for conv in self.branches]  # list of [B, C, L]
        return self.fusion(outputs)


class MultiScaleUpConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_sizes: list, stride: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.ConvTranspose1d(
                channels, channels, kernel_size=k,
                stride=stride, padding=k // 2,
                output_padding=stride - 1
            ) for k in kernel_sizes
        ])
        self.fusion = SoftmaxFusion(num_inputs=len(kernel_sizes), learnable=True)

    def forward(self, x):
        # x: [B, C, L_low]
        outputs = [branch(x) for branch in self.branches]  # list of [B, C, L_high]
        return self.fusion(outputs)  # [B, C, L_high]


class XGateFusion(nn.Module):
    def __init__(self,
                 dim: int,
                 output_dim: int,
                 cda_heads: int = 2,
                 gate_heads: int = 4,
                 alpha: float = 0.6,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0):

        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.alpha = alpha
        head_dim = dim // cda_heads
        self.scale_cda = head_dim ** -0.5
        self.cda_heads = cda_heads

        self.qkv_a = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_b = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_cda = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Gated Fusion
        self.attn_gated = nn.MultiheadAttention(embed_dim=dim, num_heads=gate_heads)
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
        self.feature_net = nn.Sequential(
            nn.Linear(dim * 2, output_dim),
            nn.ReLU()
        )
        self.res_proj = (nn.Linear(dim, output_dim)
                         if dim != output_dim else nn.Identity())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if x1.dim() == 2:
            x1 = x1.unsqueeze(0)
            x2 = x2.unsqueeze(0)
            squeeze = True
        B, T, C = x1.shape

        # Cross-Diffusion Attention with FFT
        qkv_a = self.qkv_a(x1).reshape(B, T, 3, self.cda_heads, C // self.cda_heads)
        qkv_a = qkv_a.permute(2, 0, 3, 1, 4)
        q_a, k_a, v_a = qkv_a[0], qkv_a[1], qkv_a[2]

        qkv_b = self.qkv_b(x2).reshape(B, T, 3, self.cda_heads, C // self.cda_heads)
        qkv_b = qkv_b.permute(2, 0, 3, 1, 4)
        q_b, k_b, v_b = qkv_b[0], qkv_b[1], qkv_b[2]

        # FFT-based Hadamard product (frequency domain)
        fft = torch.fft.rfft
        ifft = torch.fft.irfft

        freq_q_a = fft(q_a.float(), dim=-2)
        freq_k_b = fft(k_b.float(), dim=-2)
        freq_q_b = fft(q_b.float(), dim=-2)
        freq_k_a = fft(k_a.float(), dim=-2)

        # Hadamard product and inverse FFT
        prod_ab = freq_q_a * torch.conj(freq_k_b)
        prod_ba = freq_q_b * torch.conj(freq_k_a)

        corr_ab = ifft(prod_ab, n=T, dim=-2)
        corr_ba = ifft(prod_ba, n=T, dim=-2)

        # Normalize and dropout
        norm_ab = torch.tanh(corr_ab) * self.scale_cda
        norm_ba = torch.tanh(corr_ba) * self.scale_cda

        norm_ab = self.attn_drop(norm_ab)
        norm_ba = self.attn_drop(norm_ba)

        # Hadamard weighting of v
        out_ab = norm_ab * v_b
        out_ba = norm_ba * v_a

        # Rearrange to [B,T,C]
        out_ab = out_ab.permute(0, 2, 1, 3).reshape(B, T, C)
        out_ba = out_ba.permute(0, 2, 1, 3).reshape(B, T, C)

        # Weighted fusion
        out1 = self.alpha * out_ab + (1 - self.alpha) * x1
        out2 = self.alpha * out_ba + (1 - self.alpha) * x2

        out1 = self.proj_drop(self.proj_cda(out1))
        out2 = self.proj_drop(self.proj_cda(out2))

        # Dynamic Gated Fusion
        combined = torch.cat([out1, out2], dim=-1)
        gate = self.gate_net(combined)
        fused = gate * out1 + (1 - gate) * out2

        # Self-attention
        q = fused.permute(1, 0, 2)
        attn_out, _ = self.attn_gated(q, q, q)
        attn_out = attn_out.permute(1, 0, 2)

        # Final mapping + residual
        mapped = self.feature_net(combined)
        res = self.res_proj(attn_out)
        out = mapped + res

        if squeeze:
            out = out.squeeze(0)
        return out


class SequenceDecompositionFusion(nn.Module):
    def __init__(self, configs):
        super(SequenceDecompositionFusion, self).__init__()
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window

        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.channel_independence = configs.channel_independence

        if configs.decomp_method == 'moving_avg':
            self.decompsition = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(configs.top_k)
        elif configs.decomp_method == "wavelet":
            self.decompsition = MultiWaveletDecomposition(configs.d_model)
        elif configs.decomp_method == "sliding_pca":
            self.decompsition = SlidingPCA()
        elif configs.decomp_method == "sliding_kpca":
            self.decompsition = SlidingKernelPCA(window=configs.seq_len - 1)
        elif configs.decomp_method == "sliding_mte":
            self.decompsition = SlidingMTE()
        else:
            raise ValueError('decompsition is error')

        if not configs.channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
                nn.GELU(),
                nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
            )

        self.mixing_multi_scale_season = MultiScaleUpMixing(configs)
        self.mixing_multi_scale_trend = MultiScaleDownMixing(configs)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=configs.d_model, out_features=configs.d_ff),
            nn.GELU(),
            nn.Linear(in_features=configs.d_ff, out_features=configs.d_model),
        )
        self.self_attention = SelfAttention(embed_dim=configs.d_model, num_heads=2)
        self.fusion = XGateFusion(dim=self.d_model, output_dim=self.d_model)

    def process_out_list(self, out_list):
        tmp_out_list = []
        n_out = len(out_list)
        split_size = n_out // 3
        for i in range(0, split_size):
            out1 = out_list[i]
            out2 = out_list[i + split_size]
            out3 = out_list[i + 2 * split_size]
            out1_att = self.self_attention(out1)
            out2_att = self.self_attention(out2)
            out3_att = self.self_attention(out3)
            out_fused = self.fusion(out1_att, out2_att)
            out_fused = out_fused + out3_att
            tmp_out_list.append(out_fused)
        return tmp_out_list

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        # cross fusion
        is_decreasing = check_decreasing_tensor_size(out_season_list)
        if not is_decreasing:
            out_season_list = self.process_out_list(out_season_list)
            out_trend_list = self.process_out_list(out_trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend

            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TwoWayFusionSelector(nn.Module):
    def __init__(self, d_model):
        super(TwoWayFusionSelector, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, out1, out2):
        pooled = (out1 + out2) / 2
        pooled = pooled.mean(dim=1)
        scores = self.selector(pooled)
        weights = F.softmax(scores, dim=-1)

        avg_weights = weights.mean(dim=0)
        selected_idx = torch.argmax(avg_weights)

        if selected_idx == 0:
            return out1
        else:
            return out2

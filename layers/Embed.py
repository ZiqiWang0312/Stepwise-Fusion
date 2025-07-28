import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils import weight_norm
from positional_encodings.torch_encodings import PositionalEncoding2D
from utils.magnitude_max_pooling import magnitude_max_pooling_1d
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  #  [seq_len, batch_size, embed_dim]
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)
        return attn_output


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        # h: [batch_size, num_nodes, in_features]
        # adj: [batch_size, num_nodes, num_nodes]
        Wh = self.W(h)
        a_input = self._prepare_attention_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        return F.elu(h_prime)

    def _prepare_attention_input(self, Wh):
        # 计算注意力系数的输入
        batch_size, num_nodes, out_features = Wh.size()
        Wh_repeated = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        Wh_repeated_transposed = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        a_input = torch.cat([Wh_repeated, Wh_repeated_transposed], dim=-1)
        return a_input


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.1, alpha=0.2, nheads=4):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = nn.ModuleList([
            GATLayer(nfeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nheads)
        ])
        self.out_att = GATLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha)

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, nfeat]
        # adj: [batch_size, num_nodes, num_nodes]
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return F.log_softmax(x, dim=-1)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, num_heads=4):
        super(TokenEmbedding, self).__init__()
        self.gat = GAT(c_in, d_model, d_model, dropout=0.1, alpha=0.2, nheads=num_heads)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()

        adj = torch.ones(batch_size, seq_len, seq_len).to(x.device)  # [batch_size, seq_len, seq_len]

        x = self.gat(x, adj)  # [batch_size, seq_len, d_model]

        x = x.mean(dim=1)  # [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        return x


class TokenEmbeddingSeries(nn.Module):
    def __init__(self, c_in, d_model, hidden_dim=64, num_layers=3, dropout=0.1):
        super(TokenEmbeddingSeries, self).__init__()

        layers = []
        for i in range(num_layers):
            in_dim = c_in if i == 0 else hidden_dim
            out_dim = d_model if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.mlp(x)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataTimeEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataTimeEmbedding, self).__init__()

        self.value_embedding = TokenEmbeddingSeries(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_FreqComplex(nn.Module):
    def __init__(self, seq_len, d_model):
        super(DataEmbedding_FreqComplex, self).__init__()
        self.linear_real = nn.Linear(2 * seq_len, d_model)
        self.linear_imag = nn.Linear(2 * seq_len, d_model)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        B, N, L = x.size()
        x_fft = torch.fft.fft(x, n= 2 * L)
        x_real = self.linear_real(x_fft.real)
        x_imag = self.linear_imag(x_fft.imag)
        x = torch.complex(x_real, x_imag)
        return x


class DataEmbedding_FreqInterpolate(nn.Module):
    def __init__(self, seq_len, d_model):
        super(DataEmbedding_FreqInterpolate, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        B, N, L = x.size()
        x_fft = torch.fft.fft(x, n=2 * L)
        x_fft_resampled = self.resample_fft(x_fft, self.d_model)
        return x_fft_resampled

    def resample_fft(self, x_fft, new_length):
        real_part = x_fft.real
        imag_part = x_fft.imag
        real_interpolated = F.interpolate(real_part, size=new_length, mode='linear', align_corners=False)
        imag_interpolated = F.interpolate(imag_part, size=new_length, mode='linear', align_corners=False)
        x_fft_resampled = torch.complex(real_interpolated, imag_interpolated)
        return x_fft_resampled


class DataEmbedding_Freq_FourierInterpolate(nn.Module):
    def __init__(self, seq_len, d_model, c_in):
        super(DataEmbedding_Freq_FourierInterpolate, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.scalars = nn.Parameter(torch.ones(c_in, d_model), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(c_in, d_model, dtype=torch.cfloat), requires_grad=True)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        B, N, L = x.size()
        x_fft = torch.fft.rfft(x, n = 2 * L)
        x_fft_resampled = self.fourier_interpolate(x_fft, self.d_model)
        x_fft_resampled = x_fft_resampled * self.scalars + self.bias
        return x_fft_resampled

    def fourier_interpolate(self, x_fft, new_length):
        B, N, L = x_fft.shape
        if new_length > L:
            # Upsampling: We keep all the original data and pad with zeros in high frequencies
            resampled_data = torch.zeros(B, N, new_length, dtype=torch.cfloat,
                                         device=x_fft.device) + 0.0001  # Prepare the new data array

            resampled_data[:, :, :L] = x_fft
        else:
            # Downsampling or keeping the length the same
            resampled_data = x_fft[:, :, :new_length]

        return resampled_data


class DataEmbedding_FeaturePatching(nn.Module):
    def __init__(self, seq_len,  patch_size,  embed_dim = 512, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_FeaturePatching, self).__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_of_patches = (seq_len - patch_size)//(patch_size//2) + 1
        self.inner_dim = patch_size * 10
        self.embed_dim = embed_dim

        self.conv1 = nn.Conv1d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv1d(1, 3, kernel_size=9)
        self.conv3 = nn.Conv1d(1, 3, kernel_size=15)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.fc1 = nn.Linear(self.inner_dim, embed_dim*4)
        self.fc2 = nn.Linear(embed_dim*4, embed_dim)
        self.pe  = PositionalEncoding2D(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.sigm = nn.GELU()

    def forward(self, x, x_mark):
        B, L, N = x.shape
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]

        if x_mark is not None:
            N += x_mark.shape[2]
            x = torch.cat([x, x_mark.permute(0, 2, 1)], 1)

        x = x.reshape(-1, 1, L)
        x_1 = F.pad(x, (4, 0), mode = 'replicate')
        x_1 = self.conv1(x_1)
        x_2 = F.pad(x, (8, 0), mode = 'replicate')
        x_2 = self.conv2(x_2)
        x_3 = F.pad(x, (14, 0), mode = 'replicate')
        x_3 = self.conv3(x_3)
        x_1 = F.pad(x_1, (2, 0), mode = 'constant', value = 0)
        x_2 = F.pad(x_2, (4, 0), mode = 'constant', value = 0)
        x_3 = F.pad(x_3, (6, 0), mode = 'constant', value = 0)

        x_1 = magnitude_max_pooling_1d(x_1, 3, 1)
        x_2 = magnitude_max_pooling_1d(x_2, 5, 1)
        x_3 = magnitude_max_pooling_1d(x_3, 7, 1)



        x_1 = x_1.reshape(B, N, 3, L)
        x_2 = x_2.reshape(B, N, 3, L)
        x_3 = x_3.reshape(B, N, 3, L)
        x = x.reshape(B, N, 1, L)


        x_1 = x_1.unfold(3, self.patch_size, self.patch_size//2)
        x_2 = x_2.unfold(3, self.patch_size, self.patch_size//2)
        x_3 = x_3.unfold(3, self.patch_size, self.patch_size//2)
        x = x.unfold(3, self.patch_size, self.patch_size//2)


        x_1 = x_1.permute(0, 1, 3, 2, 4)
        x_2 = x_2.permute(0, 1, 3, 2, 4)
        x_3 = x_3.permute(0, 1, 3, 2, 4)
        x = x.permute(0, 1, 3, 2, 4)


        x = torch.cat([x, x_1, x_2, x_3], dim = 3)


        x = x.reshape(B, N, self.n_of_patches, -1)
        x = self.gelu1(self.fc1(x))
        x = self.fc2(x)
        x = self.pe(x) + x #apply 2D positional encodings

        x = x.reshape(B, -1, self.embed_dim)

        return self.dropout(x)

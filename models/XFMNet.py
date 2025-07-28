import torch
import torch.nn as nn
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos, DataEmbedding
from layers.Image_Backbone import EfficientNet
from layers.Multimodal_Decomp_Fusion import SequenceDecompositionFusion, TwoWayFusionSelector
from layers.SelfAttention_Family import SelfAttention

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.seq_decomp_fusion = SequenceDecompositionFusion(configs)

        self.preprocess = series_decomp(configs.moving_avg)
        self.enc_in = configs.enc_in

        if self.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        self.tif_enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)

        self.layer = configs.e_layers

        self.predict_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.pred_len,
                    )
                for i in range(configs.down_sampling_layers + 1)
            ]
        )
        if self.channel_independence:
            self.projection_layer = nn.Linear(
                configs.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(
                configs.d_model, configs.c_out, bias=True)

            self.out_res_layers = torch.nn.ModuleList([
                torch.nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len // (configs.down_sampling_window ** i),
                    )
                for i in range(configs.down_sampling_layers + 1)
            ])

            self.regression_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        configs.seq_len // (configs.down_sampling_window ** i),
                        configs.pred_len,
                        )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
        if configs.use_image_proj:
            self.img_proj_layers = nn.ModuleList([
                nn.Linear(
                    configs.seq_len // (configs.down_sampling_window ** i),
                    configs.seq_len * configs.enc_in // (configs.down_sampling_window ** i)
                )
                for i in range(configs.down_sampling_layers + 1)
            ])

        self.img_model = EfficientNet(num_feature=configs.enc_in, configs=configs)
        self.self_attention = SelfAttention(embed_dim=configs.d_model, num_heads=2)
        self.recursive_fuse_time = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.ReLU(),
            nn.Linear(configs.d_model * 2, configs.d_model)
        )
        self.recursive_fuse_image = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.ReLU(),
            nn.Linear(configs.d_model * 2, configs.d_model)
        )
        self.recursive_fuse_concat = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model * 2),
            nn.ReLU(),
            nn.Linear(configs.d_model * 2, configs.d_model)
        )
        self.fusion_weights = nn.Parameter(torch.randn(4))
        self.selector = TwoWayFusionSelector(d_model=configs.c_out)
        self.fusion_conv = nn.Conv2d(
            in_channels=2 * self.configs.d_model,
            out_channels=self.configs.d_model,
            kernel_size=(3, 1),
            padding=(1, 0)
        )

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.configs.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'lp':
            p = 2
            down_pool = torch.nn.LPPool1d(norm_type=p, kernel_size=self.configs.down_sampling_window,
                                          stride=self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = torch.nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                        kernel_size=3, padding=padding,
                                        stride=self.configs.down_sampling_window,
                                        padding_mode='circular',
                                        bias=False).to("cuda:0")
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def __tif_multi_scale_process_inputs(self, x):
        if self.configs.down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'lp':
            p = 2
            down_pool = nn.LPPool1d(norm_type=p, kernel_size=self.configs.down_sampling_window,
                                    stride=self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=67 * 67, out_channels=67 * 67,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False).to("cuda:0")
        else:
            return x

        B, C, H, W = x.shape
        x_ori = x.flatten(2, 3).permute(0, 2, 1)  # [B, C, H*W]
        x_list = [x]
        for i in range(self.configs.down_sampling_layers):
            x_ori = down_pool(x_ori)  # [B, C, L] → [B, C, L']
            x_sample = x_ori.permute(0, 2, 1)
            x_list.append(x_sample.view(*x_sample.shape[:-1], H, W))

        return x_list

    def preprocess_and_align_sample(self, x_enc, x_mark_enc, x_tif):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)
        x_tif_list = self.__tif_multi_scale_process_inputs(x_tif)
        x_tif_enc = self.img_model(x_tif_list)

        for i in range(len(x_tif_enc)):
            x_tif_means = x_tif_enc[i].mean(1, keepdim=True).detach()
            x_tif_enc[i] = x_tif_enc[i] - x_tif_means
            x_tif_stdev = torch.sqrt(torch.var(x_tif_enc[i], dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_tif_enc[i] = x_tif_enc[i] / x_tif_stdev

        x_list = []
        x_mark_list = []
        x_tif_mark_list = x_mark_enc

        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)

        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_att = self.self_attention(enc_out)
                enc_out_list.append(enc_out_att)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_att = self.self_attention(enc_out)
                enc_out_list.append(enc_out_att)

        x_tif_list = []
        if x_tif_mark_list is not None:
            for i, x, x_mark in zip(range(len(x_tif_enc)), x_tif_enc, x_tif_mark_list):
                enc_out = self.tif_enc_embedding(x, x_mark)
                if self.channel_independence:
                    if self.configs.use_image_proj:
                        enc_out = self.img_proj_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                        enc_out = enc_out.reshape(enc_out.size(0) * self.enc_in, enc_out.size(1) // self.enc_in,
                                                  enc_out.size(2))
                    else:
                        enc_out = enc_out.repeat(self.configs.enc_in, 1, 1)
                x_tif_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_tif_enc)), x_tif_enc):
                enc_out = self.tif_enc_embedding(x, None)
                if self.channel_independence:
                    if self.configs.use_image_proj:
                        enc_out = self.img_proj_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                        enc_out = enc_out.reshape(enc_out.size(0) * self.enc_in, enc_out.size(1) // self.enc_in,
                                                  enc_out.size(2))
                    else:
                        enc_out = enc_out.repeat(self.configs.enc_in, 1, 1)
                x_tif_list.append(enc_out)

        for i, image_feature in enumerate(x_tif_list):
            image_feature = self.self_attention(image_feature)
            enc_out_list.append(image_feature)

        for i, image_feature in enumerate(x_tif_list):
            fused = torch.cat([enc_out_list[i], image_feature], dim=-1)  # [B, T, 2C]
            fused = fused.permute(0, 2, 1).unsqueeze(-1)  # [B, 2C, T, 1]

            fused = fused.to(self.fusion_conv.weight.dtype)
            concatenated_features = self.fusion_conv(fused)  # [B, C, T, 1]
            concatenated_features = concatenated_features.squeeze(-1).permute(0, 2, 1)
            enc_out_list.append(concatenated_features)

        return x_list, enc_out_list

    def long_term_cross_site_predict(self, B, enc_out_list, x_list):
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        sum_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        stacked_out = torch.stack(dec_out_list, dim=-1)
        weights = torch.softmax(self.fusion_weights, dim=0)
        weighted_out = stacked_out * weights.view(1, 1, 1, -1)
        softmax_out = weighted_out.sum(-1)

        dec_out = self.selector(sum_out, softmax_out)

        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_tif, adj):
        B, T, L = x_enc.shape
        x_list, enc_out_list = self.preprocess_and_align_sample(x_enc, x_mark_enc, x_tif)

        enc_out_list_final = []
        for i in range(self.layer):
            if i >= 1:
                step = len(enc_out_list) // 3
                for j in range(step):
                    enc_out_list[j] = self.recursive_fuse_time(enc_out_list_final[j]) + enc_out_list[j]
                    enc_out_list[step + j] = self.recursive_fuse_image(enc_out_list_final[j]) + enc_out_list[step + j]
                    enc_out_list[step * 2 + j] = self.recursive_fuse_concat(enc_out_list_final[j]) + enc_out_list[
                        step * 2 + j]
            enc_out_list_final = self.seq_decomp_fusion(enc_out_list)
        enc_out_list = enc_out_list_final

        dec_out = self.long_term_cross_site_predict(B, enc_out_list, x_list)

        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_tif=None, adj=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, batch_tif, adj)
            return dec_out
        else:
            raise ValueError('Other tasks implemented yet')
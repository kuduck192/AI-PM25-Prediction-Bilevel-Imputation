import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional

class MAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        ft_embed_dim = args.ft_embed_dim
        ft_enc_nhead = args.ft_enc_nhead
        ft_enc_num_layers = args.ft_enc_num_layers
        mae_hidden_dim = args.mae_hidden_dim
        mae_nhead = args.mae_nhead
        mae_num_layers = args.mae_num_layers
        activation = args.activation
        mae_dropout = args.mae_dropout
        mlp_ratio = args.mlp_ratio
        dim_feedforward = args.dim_feedforward
        self.pos_embed_indices = args.pos_embed_indices

        
        self.fc_ft_embed = nn.Linear(1, ft_embed_dim)
        self.ft_enc_layer = nn.TransformerEncoderLayer(
            ft_embed_dim, nhead=ft_enc_nhead, dim_feedforward=dim_feedforward,
            batch_first=True, activation=activation)
        self.ft_enc = nn.TransformerEncoder(self.ft_enc_layer, num_layers=ft_enc_num_layers)
        self.fc2 = nn.Linear(ft_embed_dim * args.num_features, mae_hidden_dim)
        
        self.station_enc_layer = nn.TransformerEncoderLayer(
            mae_hidden_dim, nhead=mae_nhead, dim_feedforward=int(mae_hidden_dim*mlp_ratio),
            batch_first=True, activation=activation, dropout=mae_dropout)
        self.station_enc = nn.TransformerEncoder(self.station_enc_layer, num_layers=mae_num_layers)
        
        self.time_enc_layer = nn.TransformerEncoderLayer(
            mae_hidden_dim, nhead=mae_nhead, dim_feedforward=int(mae_hidden_dim*mlp_ratio),
            batch_first=True, activation=activation, dropout=mae_dropout)
        self.time_enc = nn.TransformerEncoder(self.time_enc_layer, num_layers=mae_num_layers)

        self.fc3 = nn.Linear(mae_hidden_dim, mae_hidden_dim)
        self.fc4 = nn.Linear(mae_hidden_dim, mae_hidden_dim)
        
        self.rec_head = nn.Sequential(
            nn.Linear(mae_hidden_dim, mae_hidden_dim), nn.GELU(),
            nn.Linear(mae_hidden_dim, args.num_features)
        )
        
    
    def random_masking(self, x, m, mask_ratio, training=None):
        T, S, L, D = x.shape
        N = T * S
        x = x.reshape(T * S, x.shape[-2], x.shape[-1])
        m = m.reshape(T * S, m.shape[-1])
        if training is None:
            training = self.training
        if training:
        
            len_keep = int(L * (1 - mask_ratio))
            noise = torch.rand(N, L, device=x.device)
            noise[m < 1e-6] = 1
            noise[:, self.pos_embed_indices] = 0
            
            ids_shuffle = torch.argsort(noise, dim=1).to(x.device)
            ids_restore = torch.argsort(ids_shuffle, dim=1).to(x.device)
            ids_keep = ids_shuffle[:, :len_keep]
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore).to(x.device)
            mask = torch.logical_or(mask, ~m.bool())
            
            # mask = torch.logical_or(mask, row_mask.bool())
            
            nask = ~mask
            mask = mask.reshape(T, S, L)
            nask = nask.reshape(T, S, L)
            return mask, nask
            
        else:
            mask = ~m.bool()
            nask = m.bool()
            mask = mask.reshape(T, S, L)
            nask = nask.reshape(T, S, L)
            return mask, nask

    def forward_encoder(self, batch, mask_org, mask_ratio=0.5, training=None):
        T, S, N = batch.shape
        batch = batch.unsqueeze(-1)
        out = self.fc_ft_embed(batch) # (T, S, N, D0)
        D0 = out.shape[-1]

        mask, nask = self.random_masking(out, mask_org, mask_ratio, training)
        out = out * (nask.unsqueeze(-1)).float()

        out = self.ft_enc(out.reshape(T * S, N, D0), src_key_padding_mask=mask.reshape(T * S, N))
        out = out.nan_to_num(0.0)
        out = out.reshape(T, S, N, D0).reshape(T, S, -1) # (T, S, N * D0)
        out = self.fc2(out) # (T, S, D)
        

        station_out = self.station_enc(out) # (T, S, D)
        time_out = self.time_enc(out.transpose(0, 1)).transpose(0, 1) # (T, S, D)

        enc_out = self.fc3(station_out) + self.fc4(time_out)
        return enc_out, mask, nask

    def forward_decoder(self, enc_out):
        dec_out = self.rec_head(enc_out)
        return dec_out

    def forward_impute(self, batch, mask_org, training=False):
        if training:
            self.train()
            enc_out, _, _ = self.forward_encoder(batch, mask_org, 0.0, False)
            dec_out = self.forward_decoder(enc_out)
            return batch * mask_org + dec_out * (1 - mask_org)

        self.eval()
        with torch.no_grad():
            enc_out, _, _ = self.forward_encoder(batch, mask_org, 0.0, False)
            dec_out = self.forward_decoder(enc_out)

        return batch * mask_org + dec_out * (1 - mask_org)


    def forward_ssl(self, batch, mask_org, mask_ratio=0.5):
        target = batch.clone()

        enc_out, mask, nask = self.forward_encoder(batch, mask_org, mask_ratio)
        dec_out = self.forward_decoder(enc_out)

        rec_mask = mask * mask_org

        loss = (dec_out - target) ** 2
        loss = (loss * rec_mask).sum() / (rec_mask.sum() + 1e-6)
        return loss


class LSTMStudent(nn.Module):
    def __init__(self, args):
        super(LSTMStudent, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=args.num_features,
            hidden_size=100,
            num_layers=2,
            batch_first=True
        )
        self.fc = nn.Linear(100, args.num_features)
        
    def forward(self, x, train=True, inp_dropout=0.3):
        if train: x = F.dropout(x, p=inp_dropout)
        out, _ = self.lstm(x)
        
        last_output = out[:, -1, :]
        output = self.fc(last_output)
        
        return output
import torch
import torch.nn as nn
import numpy as np


def get_2d_sincos_pos_embed(embed_dim=128, max_width=40, max_height=20):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed.shape:[h, w, embed_dim]
    """
    grid_h = np.arange(max_height, dtype=np.float32)
    grid_w = np.arange(max_width, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    # grid坐标形式:[x,y],shape:[2,h,w]
    grid = np.stack(grid, axis=0)
    # shape:[2,1,h,w]
    # grid = grid.reshape([2, 1, max_height, max_width])
    # 输入grid.shape:[2,w,h],pos_embed.shape:[h, w, dim]
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, h=max_height, w=max_width)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid, h, w):
    assert embed_dim % 4 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]).reshape(h, w, -1)  # (H, W, D/2)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]).reshape(h, w, -1)  # (H, W, D/2)

    emb = np.concatenate([emb_x, emb_y], axis=-1) # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded, shape:[1,h,w]
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / (10000**omega)  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_width=200, max_height=200):
        # d_model是位置编码向量的维度
        super(PositionalEncoding, self).__init__()
        # shape:[h,w,dim]
        self.pos_embed = nn.Parameter(torch.zeros(max_height, max_width, d_model), requires_grad=False).cuda()
        self.pos_embed.data.copy_(torch.from_numpy(get_2d_sincos_pos_embed(embed_dim=d_model, max_width=max_width, max_height=max_height)).float())

    def forward(self, x):
        """ x, 表示文本序列的词嵌入表示 """
        # x.shape:[b, h, w, dim]
        grid_x_h = torch.arange(x.shape[1]).cuda()
        grid_x_w = torch.arange(x.shape[2]).cuda()
        grid_x = torch.meshgrid(grid_x_h, grid_x_w)
        # [y,x],shape:[2,h,w]->[h*w,2]
        grid_x = torch.stack(grid_x, dim=0).reshape(2, -1).permute(1, 0)
        pos_fea = self.pos_embed[grid_x[:, 0], grid_x[:, 1]].reshape(x.shape[1], x.shape[2], -1)
        return x + pos_fea.unsqueeze(0)


class STAN_transformer(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=1, n_head=4, num_layers=2):
        super(STAN_transformer, self).__init__()
        # TODO here 1.在做线性映射时需不需要拆成多步，2.需不需要初始化
        self.hidden_dim=hidden_dim
        self.embedding_layer = nn.Sequential(
                                # nn.Linear(input_dim, hidden_dim),
                                nn.Linear(input_dim, hidden_dim//4),
                                nn.ReLU(),
                                nn.Linear(hidden_dim//4, hidden_dim)
        )
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_head, dim_feedforward=hidden_dim*4), num_layers=num_layers)
        # TODO 最后的fc层，待定
        self.hidden2tag = nn.Sequential(
            # nn.Linear(hidden_dim, output_dim),
            nn.Linear(hidden_dim, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        # self.hidden2tag_2 = nn.Linear(hidden_dim//4, output_dim)

    def forward(self, Dt):
        # Dt.shape: [b, h, w]
        # input_row.shape: [b, h, w, 1]
        b, h, w = Dt.size(0), Dt.size(1), Dt.size(2)
        input_row = Dt.view(b, h, w, 1)
        # input_row.shape: [b, h, w, hidden_dim]
        input_row = self.embedding_layer(input_row)
        # 添加位置编码，输出的shape:[b, h, w, hidden_dim] -> [b, h*w, hidden_dim]
        input_row = self.positional_encoding(input_row).view(b, -1, self.hidden_dim)
        # input_row.shape: [h*w, b, hidden_dim]
        input_row = input_row.permute(1, 0, 2)
        # row_out.shape: [h*w, b, hidden_dim] -> [b, h*w, hidden_dim]
        row_out = self.transformer_encoder(input_row).permute(1, 0, 2)
        # 最后的fc映射,[b, h*w, 1]
        tag_space = self.hidden2tag(row_out)
        tag_scores = torch.sigmoid(tag_space)
        # [b, h, w]
        return tag_scores.view(b, h, w).contiguous()


class STAN_FC(nn.Module):
    def __init__(self, hidden_dim=128):
        super(STAN_FC, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(1, hidden_dim//4)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim//4, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim//4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim//4, 1)

    def forward(self, x):
        b, h, w = x.size(0), x.size(1), x.size(2)
        x = x.view(b, h, w, 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x.view(b, h, w).contiguous()


if __name__ == '__main__':
    stan=STAN_transformer().cuda()
    print(stan)
    # x=torch.randn(1, 5, 5).cuda()
    x = torch.randn(1, 120, 120).cuda()
    stan(x)

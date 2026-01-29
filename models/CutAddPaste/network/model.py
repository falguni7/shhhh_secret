from torch import nn
import torch


class base_Model(nn.Module):
    def __init__(self, configs, device):
        super(base_Model, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.window_size = configs.window_size
        self.device = device
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.project = configs.project

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.final_out_channels * self.features_len, self.final_out_channels * self.features_len // self.project),
            nn.BatchNorm1d(self.final_out_channels * self.features_len // self.project),
            nn.ReLU(inplace=True),
            nn.Linear(self.final_out_channels * self.features_len // self.project, 2),
        )
        self.logits = nn.Linear(self.final_out_channels * self.features_len, 2)

    def forward(self, x_in):
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        # 1D CNN feature extraction
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        # print(x.size())
        # Encoder
        hidden = x.permute(0, 2, 1)
        hidden = hidden.reshape(hidden.size(0), -1)
        logits = self.projection_head(hidden)
        # logits = self.logits(hidden)

        return logits


class dal_Model(nn.Module):
    """DAL-specific model: similar conv blocks and projection head but
    exposes `pred_emb`, `fc`, `fc_aug`, and `forward` methods expected by
    `syndal_main.py` DAL algorithm.
    """
    def __init__(self, configs, device):
        super(dal_Model, self).__init__()
        self.input_channels = configs.input_channels
        self.final_out_channels = configs.final_out_channels
        self.features_len = configs.features_len
        self.window_size = configs.window_size
        self.device = device
        self.kernel_size = configs.kernel_size
        self.stride = configs.stride
        self.dropout = configs.dropout
        self.project = configs.project

        # replicate the conv blocks from base_Model
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False, padding=(self.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(self.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        # Split the projection head into two parts:
        #  - `projection_head_base`: maps embedding -> intermediate features
        #  - `projection_head_final`: final linear layer mapping to logits
        # This lets the inner-loop operate on embeddings and reuse the final
        # linear layer (DAL-style) without needing a separate `fc_aug`.
        emb_dim = int(self.final_out_channels * self.features_len)
        hidden_dim = emb_dim // self.project
        self.projection_head_base = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # expose final linear as `self.classifier` so callers can access the final layer
        self.fc = nn.Linear(hidden_dim, 2)

    def pred_emb(self, x_in):
        """Return (logits, features) for input tensor `x_in`.

        - `logits`: class logits produced by applying the projection head
            base followed by the final linear layer (shape: (B, 2)).
        - `features`: the post-projection-base activations (the output of
            `projection_head_base`, i.e. Linear -> BatchNorm -> ReLU). These
            are NOT the raw flattened conv outputs; they are lower-dimensional
            features (shape: (B, hidden_dim)) intended for inner-loop
            perturbation (the inner optimization perturbs these features and
            maps them to logits via `self.fc`).
        """
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        hidden = x.permute(0, 2, 1).reshape(x.size(0), -1)
        # compute logits by passing through base then final
        h = self.projection_head_base(hidden)
        return self.fc(h), h

    def forward(self, x_in):
        """Forward returns logits for input x_in by running conv encoder and
        the projection head (explicit layers, DAL style).
        """
        if torch.isnan(x_in).any():
            print('tensor contain nan')
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        hidden = x.permute(0, 2, 1)
        hidden = hidden.reshape(hidden.size(0), -1)
        h = self.projection_head_base(hidden)
        return self.fc(h)


from torch import nn
from torch.nn import Sequential


class FeedForwardModule(nn.Module):
    def __init__(self, n_feats, expansion=4, p=0.1):
        super().__init__()

        self.seq = Sequential(
            nn.LayerNorm(n_feats),
            nn.Linear(in_features=n_feats, out_features=expansion * n_feats),
            nn.SiLU(),  # swish
            nn.Dropout(p),
            nn.Linear(in_features=expansion * n_feats, out_features=n_feats),
            nn.Dropout(p),
        )

    def forward(self, input):
        return self.seq(input)


class ConvModule(nn.Module):
    def __init__(self, n_feats, kernel_size, p=0.1):
        super().__init__()

        self.ln = nn.LayerNorm(n_feats)
        self.seq = Sequential(
            nn.Conv1d(
                in_channels=n_feats, out_channels=n_feats * 2, kernel_size=1
            ),  # bc this is pointwise conv
            nn.GLU(dim=1),
            nn.Conv1d(
                in_channels=n_feats,
                out_channels=n_feats,
                kernel_size=kernel_size,
                padding="same",
                groups=n_feats,
            ),
            nn.BatchNorm1d(num_features=n_feats, affine=False),
            nn.SiLU(),
            nn.Conv1d(in_channels=n_feats, out_channels=n_feats, kernel_size=1),
            nn.Dropout(p),
        )

    def forward(self, input):
        output = self.ln(input)
        output = output.transpose(1, 2)
        return self.seq(output).transpose(1, 2)


class MHSA(nn.Module):
    def __init__(self, n_feats, n_heads=16, p=0.1):
        super().__init__()

        self.ln = nn.LayerNorm(n_feats)
        self.attention = nn.MultiheadAttention(
            embed_dim=n_feats, num_heads=n_heads, dropout=p, batch_first=True
        )

    def forward(self, input):
        output = self.ln(input)
        output, _ = self.attention(output, output, output, need_weights=False)

        return output


class ConfBlock(nn.Module):
    def __init__(
        self,
        n_feats,
        n_heads=16,
        kernel_size=32,
        p=0.1,
        expansion=4,
    ):
        super().__init__()

        self.ff = FeedForwardModule(n_feats=n_feats, expansion=expansion, p=p)
        self.conv = ConvModule(n_feats=n_feats, kernel_size=kernel_size, p=p)
        self.mhsa = MHSA(n_feats=n_feats, n_heads=n_heads, p=p)
        self.layernorm = nn.LayerNorm(n_feats)

    def forward(self, input):
        res1 = input + self.ff(input) * 0.5
        res2 = res1 + self.mhsa(res1)
        res3 = res2 + self.conv(res2)
        res4 = res3 + 0.5 * self.ff(res3)
        return self.layernorm(res4)


class ConformerModel(nn.Module):
    """
    Conformer
    """

    def __init__(
        self,
        n_feats,
        encoder_dim,
        n_tokens,
        n_blocks=16,
        kernel_size=32,
        n_heads=16,
        p=0.1,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.subsample = nn.Conv2d(
            in_channels=1, out_channels=encoder_dim, kernel_size=(2, 1), stride=(2, 1)
        )
        self.linear = nn.Linear(encoder_dim * (n_feats >> 2 - 1), encoder_dim)
        self.dropout = nn.Dropout(p)

        self.layers = nn.ModuleList(
            [
                ConfBlock(
                    n_feats=encoder_dim, n_heads=n_heads, kernel_size=kernel_size, p=p
                )
                for _ in range(n_blocks)
            ]
        )

        # self.dec = nn.Linear(encoder_dim, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        output = self.subsample(spectrogram.transpose(1, 2).unsqueeze(1)).squeeze(1)
        output = self.linear(output)
        output = self.dropout(output)

        for layer in self.layers:
            output = layer(output)

        # output = self.dec(output)

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths // 2  # actually subsampling reduces

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info

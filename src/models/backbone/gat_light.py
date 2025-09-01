import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B,HW,C) — преобразование тензора к последовательности
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)
        res = x_seq
        x_seq = self.norm1(x_seq)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq, need_weights=False)
        x_seq = res + attn_out
        res2 = x_seq
        x_seq = self.norm2(x_seq)
        x_seq = res2 + self.mlp(x_seq)
        x = x_seq.transpose(1, 2).reshape(B, C, H, W)
        return x


class GATLight(nn.Module):
    """
    Облегчённый Geometry-Aware Transformer: даунсэмплинг 2x и несколько блоков Transformer.
    Вход: (B, C_in=192, H/8, W/8)
    Выход: (B, C_out=384, H/16, W/16)
    """

    def __init__(
        self,
        in_channels=192,
        out_channels=384,
        img_size=224,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2],
        num_heads=[3, 6],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        **kwargs
    ):
        super().__init__()
        # Сохраняем входное разрешение
        self.input_resolution = (
            img_size // (patch_size * 8),
            img_size // (patch_size * 8),
        )
        self.down = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(out_channels, num_heads=num_heads[0])
                for _ in range(depths[0])
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.down(x)
        for blk in self.blocks:
            x = blk(x)
        B, C, H, W = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return [x]

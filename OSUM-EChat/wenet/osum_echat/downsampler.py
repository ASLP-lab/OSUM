import torch
from torch import nn


class osum_echatConv1dSubsampling2(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class osum_echatConv1dSubsampling4(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConstantPad1d((2, 0), 0.0),
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.ConstantPad1d((2, 0), 0.0),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.ConstantPad1d((2, 0), 0.0),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
        )

    def forward(self, x, mask_pad):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        mask_pad = mask_pad[:, :, 0::2]
        mask_pad = mask_pad[:, :, 0::2]
        return x, mask_pad


class osum_echatConv1dSubsampling6(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 3),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x


class osum_echatConv1dSubsampling8(nn.Module):
    """Conv1d subsampling module.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int):
        """Construct an Conv1dSubsampling object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, 3, 1),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
            torch.nn.Conv1d(odim, odim, 3, 2),
            torch.nn.GELU(),
        )

    def forward(self, x):
        """

        Args:
            x: (B, T, idim)

        Returns:
        """
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x
    
class osum_echat2Conv1dSubsampling(torch.nn.Module):
    def __init__(
        self,
        enc_out_dim: int = 512,
        llm_embed_dim: int = 4096,
        kernel_size: int = 5,
        activation_func: str = 'relu',
        norm: str = 'batch',
    ):
        super().__init__()
        
        if enc_out_dim * 4 < llm_embed_dim:
            self.left_padding1 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d1 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 1, 0)
            self.bn1 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
            self.relu1 = nn.ReLU()

            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(2 * enc_out_dim, 4 * enc_out_dim, kernel_size, 2, 0)
            self.bn2 = nn.BatchNorm1d(4 * enc_out_dim, eps=1e-3, momentum=0.99)
            self.relu2 = nn.ReLU()
            
            self.project = nn.Linear(4 * enc_out_dim, llm_embed_dim)
            self.cnn_num = 2
        else:
            self.left_padding2 = nn.ConstantPad1d((kernel_size - 1, 0), 0.0)
            self.conv1d2 = nn.Conv1d(enc_out_dim, 2 * enc_out_dim, kernel_size, 2, 0)
            if norm == 'batch':
                self.bn2 = nn.BatchNorm1d(2 * enc_out_dim, eps=1e-3, momentum=0.99)
            elif norm == 'layer':
                self.bn2 = nn.LayerNorm(2 * enc_out_dim, eps=1e-3)
            if activation_func == 'gelu':
                self.relu2 = nn.GELU()
            else:
                self.relu2 = nn.ReLU()
            self.project = nn.Linear(2 * enc_out_dim, llm_embed_dim)
            self.cnn_num = 1
    
    def forward(self, x, mask_pad):
        """
            x: B, T, enc_out_dim
            mask: (B, T) or (B, 1, T)
        """
        x = x.transpose(1, 2)  # B, channels, T

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.cnn_num == 2:
            x = self.left_padding1(x)
            x = self.conv1d1(x)
            x = self.bn1(x)
            x = self.relu1(x)

        x = self.left_padding2(x)
        x = self.conv1d2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.bn2(x)
        if isinstance(self.bn2, nn.LayerNorm):
            x = x.transpose(1, 2)
        x = self.relu2(x)

        x = x.transpose(1, 2)
        x = self.project(x)

        return x, mask_pad[:, :, 0::2]
    
def get_downsampler(downsample_rate, ndim=1280):
    down_sample_2 = nn.Identity()
    if downsample_rate == 2:
        down_sample_2 = osum_echatConv1dSubsampling2(ndim, ndim)
    elif downsample_rate == 4:
        down_sample_2 = osum_echatConv1dSubsampling4(ndim, ndim)
    elif downsample_rate == 8:
        down_sample_2 = osum_echatConv1dSubsampling8(ndim, ndim)
    elif downsample_rate == 6:
        down_sample_2 = osum_echatConv1dSubsampling6(ndim, ndim)
    return down_sample_2
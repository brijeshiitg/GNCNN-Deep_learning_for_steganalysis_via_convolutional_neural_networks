"""
This is unofficial implementation of GNCNN:  Deep Learning for Steganalysis
  via Convolutional Neural Networks. """
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F


def gaussian(inp: Tensor) -> Tensor:
    """Computes the gaussian function on the input tensor.
    Args:
            inp (Tensor): input tensor.
    Returns:
            Tensor: tensor after applying gaussian.
    """
    # pylint: disable=E1101
    return torch.exp(
        -((inp - torch.mean(inp)) ** 2) / (torch.std(inp)) ** 2
    )  # pylint: enable=E1101


class ImageProcessing(nn.Module):
    """Computes convolution with KV filter over the input tensor."""

    def __init__(self) -> None:
        """Constructor"""

        super().__init__()
        # pylint: disable=E1101
        self.kv_filter = torch.tensor(
            [
                [-1.0, 2.0, -2.0, 2.0, -1.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-2.0, 8.0, -12.0, 8.0, -2.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-1.0, 2.0, -2.0, 2.0, -1.0],
            ],
        ).view(
            1, 1, 5, 5
        )  # pylint: enable=E1101

    def forward(self, inp: Tensor) -> Tensor:
        """Returns tensor convolved with KV filter"""

        return F.conv2d(inp, self.kv_filter)


class ConvPool(nn.Module):
    """This class returns building block for GNCNN class."""

    def __init__(
        self,
        in_channels: int = 16,
        kernel_size: int = 3,
        pool_padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=True,
        )
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=pool_padding)

    def forward(self, inp: Tensor) -> Tensor:
        """Returns conv->gaussian->average pooling."""

        return self.pool(gaussian(self.conv(inp)))


class GNCNN(nn.Module):
    """This class returns GNCNN model."""

    def __init__(self) -> None:
        super().__init__()
        self.layer1 = ConvPool(in_channels=1, kernel_size=5, pool_padding=1)
        self.layer2 = ConvPool(pool_padding=1)
        self.layer3 = ConvPool()
        self.layer4 = ConvPool()
        self.layer5 = ConvPool(kernel_size=5)
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=2),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image: Tensor) -> Tensor:
        """Returns logit for the given tensor."""
        with torch.no_grad():
            out = ImageProcessing()(image)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)
        out = self.softmax(out)
        return out


# if __name__ == "__main__":
#     net = GNCNN()
#     print(net)
#     inp_image = torch.randn((1, 1, 256, 256))
#     print(net(inp_image))

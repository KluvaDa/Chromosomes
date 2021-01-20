import torch
import torch.nn.functional
import numpy as np

from typing import List, Optional, Callable, Type


class FullyConv(torch.nn.Module):
    def __init__(self,
                 n_channels_in: int,
                 ns_channels_layers: List[int],
                 activation: Callable[[torch.Tensor], torch.Tensor],
                 kernel_size: int = 3,
                 groups: int = 1,
                 norm_layer: Optional[Type[torch.nn.Module]] = None,
                 raw_output: bool = False,
                 is3d: bool = False):
        """
        A fully convolutional network in 2D or 3D.
        Args:
            n_channels_in: number of input channels
            ns_channels_layers: list of the number of channels for each subsequent convolutional layer
            activation: What activation function to use after each convolution
            kernel_size: The size of the kernel
            groups: the number of channels must be divisible by groups
            norm_layer: What normalisation layer to use. Defaults to BatchNorm2d or BatchNorm3d if norm_layer is None
            raw_output: Whether to skip the last activation function
            is3d: Whether to apply 3d convolutions (True), or 2d convolutions (False)
        """
        super(FullyConv, self).__init__()
        if norm_layer is None:
            if is3d:
                norm_layer = torch.nn.BatchNorm3d
            else:
                norm_layer = torch.nn.BatchNorm2d
        if is3d:
            conv_layer = torch.nn.Conv3d
        else:
            conv_layer = torch.nn.Conv2d
        self.raw_output = raw_output
        self.layers = torch.nn.ModuleList([
            conv_layer(ch_in, ch_out, kernel_size, groups=groups, stride=1, padding=(kernel_size - 1) // 2)
            for ch_in, ch_out in zip([n_channels_in] + ns_channels_layers, ns_channels_layers)
        ])
        self.norm_layers = torch.nn.ModuleList([
            norm_layer(ch)
            for ch in ns_channels_layers
        ])
        self.activation = activation
        self.depth = len(ns_channels_layers)

    def forward(self, x_in):
        x = x_in
        for i, (conv_layer, norm_layer) in enumerate(zip(self.layers, self.norm_layers)):
            x = conv_layer(x)
            if i < self.depth - 1 or not self.raw_output:
                x = self.activation(x)
                x = norm_layer(x)
        return x


class Unet(torch.nn.Module):
    def __init__(self,
                 n_channels_in: int,
                 n_channels_out: int,
                 n_channels_start: int,
                 depth_encoder: int,
                 depth_decoder: int,
                 n_resolutions: int,
                 backbone_net: torch.nn.Module,
                 input_net: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 output_net: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
                 mode_add: bool = True,
                 norm_layer: Optional[Type[torch.nn.Module] ] = None,
                 is3d: bool = False):
        """
        :param n_channels_in:
        :param n_channels_out:
        :param n_channels_start: How many channels to work with at the native resolution. Will double with each
            downsample
        :param depth_encoder: How many layers in each encoder network
        :param depth_decoder: How many layers in each decoder network
        :param n_resolutions: How many times to downsample. There will be n-1 encoder/decoder modules
        :param backbone_net: Module that will be applied at the lowest resolution
        :param input_net: Pre-processing applied to the inputs.
        :param output_net: Post-processing applied to the outputs. This network should apply any desired activation
            functions
        :param mode_add: Whether to add the skip connections instead of concatenating
        :param norm_layer: Class, which normalises in 2d and takes the number of channels as an input. None for
            BatchNorm2d
        """
        super(Unet, self).__init__()
        self.n_resolutions = n_resolutions
        self.mode_add = mode_add
        self.backbone_net = backbone_net
        self.input_net = input_net
        self.output_net = output_net

        if is3d:
            self.pooling = torch.nn.functional.max_pool3d
            self.unpooling = torch.nn.functional.max_unpool3d
        else:
            self.pooling = torch.nn.functional.max_pool2d
            self.unpooling = torch.nn.functional.max_unpool2d

        self.encoder = []
        self.decoder = []

        channels_at_resolution = np.power(2, range(n_resolutions)) * n_channels_start
        encoder_n_channels_in = n_channels_in
        for resolution in range(n_resolutions - 1):
            # encoder
            self.encoder.append(FullyConv(encoder_n_channels_in,
                                          [channels_at_resolution[resolution]] * depth_encoder,
                                          torch.nn.functional.relu,
                                          norm_layer=norm_layer,
                                          is3d=is3d))
            encoder_n_channels_in = channels_at_resolution[resolution]

            # decoder
            decoder_n_channels_in = channels_at_resolution[resolution] if self.mode_add else \
                channels_at_resolution[resolution] * 2
            decoder_channels = [channels_at_resolution[resolution]] * (depth_decoder - 1)
            if resolution > 0:
                decoder_channels += [channels_at_resolution[resolution] // 2]
            else:
                decoder_channels += [n_channels_out]
            self.decoder.append(FullyConv(decoder_n_channels_in,
                                          decoder_channels,
                                          torch.nn.functional.relu,
                                          norm_layer=norm_layer,
                                          raw_output=(resolution == 0),
                                          is3d=is3d))

        self.encoder = torch.nn.ModuleList(self.encoder)
        self.decoder = torch.nn.ModuleList(self.decoder)

    def forward(self, x_in):
        skip_connections = []
        pooling_indices = []

        # input
        x = self.input_net(x_in) if self.input_net is not None else x_in

        # encoder
        for resolution in range(self.n_resolutions - 1):
            x = self.encoder[resolution](x)
            skip_connections.append(x)
            x, indices = self.pooling(x, 2, 2, return_indices=True)
            pooling_indices.append(indices)

        # backbone
        x = self.backbone_net(x)

        # decoder
        for resolution in reversed(range(self.n_resolutions - 1)):
            skip = skip_connections.pop()
            x = self.unpooling(x, pooling_indices.pop(), 2, 2, output_size=skip.size())
            if self.mode_add:
                x = torch.add(x, skip)
            else:
                x = torch.cat([x, skip], dim=1)
            x = self.decoder[resolution](x)

        # output
        x = self.output_net(x) if self.output_net is not None else x

        return x

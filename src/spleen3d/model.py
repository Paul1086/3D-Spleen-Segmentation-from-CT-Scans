"""
Build the 3D U-Net used for spleen segmentation.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""

from monai.networks.layers import Norm
from monai.networks.nets import UNet

def build_model(config):
    model_config = config["model"]

    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=tuple(model_config["channels"]),
        strides=tuple(model_config["strides"]),
        num_res_units=int(model_config["num_res_units"]),
        norm=Norm.INSTANCE,
        dropout=float(model_config["dropout"]),
    )
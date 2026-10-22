# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from ._channel_model import ChannelModel
from .ica_bad_channels import ChannelGap, ChannelGapResult, find_channel_gaps
from .base import (
    BadChannelWindow,
    channel_listlist_to_dict,
    find_flat_epochs,
    find_flat_evoked,
    find_noisy_channels,
    new_rejection_ds,
)

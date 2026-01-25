#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import logging
from typing import Union, Optional

# Third-party libraries


# User define module
from layoutgcn.utils.base_config import BaseConfig

# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------Main-----------------------------------------------------------
class LayoutGCNConfig(BaseConfig):

    def __init__(
        self,
        model_name: Optional[str]="LayoutGCN",
        num_word_embeddings: Optional[int]=None,
        padding_token_idx: Optional[int]=0,
        max_seq_length: Optional[int]=256,
        max_num_nodes: Optional[int]=256,
        hidden_size: Optional[int]=256,
        filter_sizes: Optional[list]=[2, 3, 4, 5],
        num_filters: Optional[list]=[32, 64, 128, 256],
        embedding_dim: Optional[int]=128,
        dropout_prob: Optional[float]=0.6,
        num_angle_embeddings: Optional[int]=36,
        use_image: Optional[bool]=False,
        efficientnet_model_path: Optional[str]="./efficientnet-b0",
        roi_pooling_size: Optional[int]=7,
        num_labels: Optional[int]=2,
        use_crf: Optional[bool]=False,
        use_return_dict: Optional[bool]=True,
        **kwargs
    ):
        self.model_name = model_name
        self.num_word_embeddings = num_word_embeddings
        self.padding_token_idx = padding_token_idx
        self.max_seq_length = max_seq_length
        self.max_num_nodes = max_num_nodes
        self.num_labels = num_labels
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.hidden_size = hidden_size
        self.use_image = use_image
        self.efficientnet_model_path = efficientnet_model_path
        self.roi_pooling_size = roi_pooling_size
        self.dropout_prob = dropout_prob
        self.num_angle_embeddings = num_angle_embeddings
        self.use_crf = use_crf
        self.use_return_dict = use_return_dict
        super().__init__(**kwargs)

    @classmethod
    def from_model_path(
        cls,
        model_path: Union[str, os.PathLike]
    ) -> "BaseConfig":
        r"""
        Instantiate a [`BaseConfig`] (or a derived class) from a existing model configuration.

        Args:
            model_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file .

        Returns:
            [`BaseConfig`]: The configuration object instantiated from this pretrained model.
    
        """
        config_file = os.path.join(model_path, "model_config.json")
        if not os.path.isfile(config_file):
            raise ValueError(f"Can't find a configuration file for model at {config_file}")
        config_dict = cls._dict_from_json_file(config_file)
        return cls.from_dict(config_dict)
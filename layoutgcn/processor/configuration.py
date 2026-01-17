#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
from typing import Union, Optional

# Third-party libraries
from transformers.utils import logging

# User define module
from layoutgcn.utils.base_config import BaseConfig


# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.get_logger(__name__)

# -----------------------------------------------------------Main-----------------------------------------------------------
class DocProcessorConfig(BaseConfig):

    def __init__(
        self,
        vocab_file: Optional[str]=None,
        max_seq_length: Optional[int]=256,
        max_num_nodes: Optional[int]=256,
        do_pre_tokenize: Optional[bool]=False,
        do_lower_case: Optional[bool]=True,
        radical_alpha: Optional[int]=50,
        angle_delta: Optional[int]=10,
        sep_token: Optional[str]=" ",
        unk_token: Optional[str]="<unk>",
        pad_token: Optional[str]="<pad>",
        label2id: Optional[dict]=None,
        use_image: Optional[bool]=False,
        efficientnet_model_path: Optional[str]=None,
        shuffle_prob: Optional[float]=0.2,
        dropped_prob: Optional[float]=0.1,
        task_type: Optional[str]="document_classification",
        **kwargs
    ):
        self.max_seq_length = max_seq_length
        self.max_num_nodes = max_num_nodes
        self.vocab_file = vocab_file
        self.radical_alpha = radical_alpha
        self.angle_delta = angle_delta
        self.do_pre_tokenize = do_pre_tokenize
        self.do_lower_case = do_lower_case
        self.label2id = label2id
        self.sep_token = sep_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.use_image = use_image
        self.efficientnet_model_path = efficientnet_model_path
        self.shuffle_prob = shuffle_prob
        self.dropped_prob = dropped_prob
        self.task_type = task_type
        if self.label2id is not None:
            self.id2label = {v:k for k, v in self.label2id.items()}
            if self.task_type == "information_extraction":
                self.id2label.update({v+1:k for k, v in self.label2id.items()})
            self.num_labels = max(self.label2id.values()) + 1 if min(self.label2id.values()) == 0 else max(self.label2id.values()) + 2
        else:
            self.id2label = None

        # document classification / node classification / link prediction / information extraction
        allowed_type_types = ("document_classification", "node_classification", "link_prediction", "information_extraction")
        if self.task_type is not None and self.task_type not in allowed_type_types:
            raise ValueError(
                f"The config parameter `task_type` was not understood: received {self.task_type} "
                "but only 'document_classificaiton', 'node_classification', 'link_prediction' and 'information_extraction' are valid."
            )
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
        config_file = os.path.join(model_path, "processor_config.json")
        if not os.path.isfile(config_file):
            raise ValueError(f"Can't find a configuration file for processor at {config_file}")
        config_dict = cls._dict_from_json_file(config_file)
        return cls.from_dict(config_dict)

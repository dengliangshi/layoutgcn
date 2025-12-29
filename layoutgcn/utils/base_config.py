#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import json
import copy
import logging
from typing import Any, Union

# Third-party libraries


# User define module



# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger(__name__)

# -----------------------------------------------------------Main-----------------------------------------------------------
class BaseConfig(object):

    base_config_key: str = ""
    attribute_map: dict[str, str] = {}

    def __init__(self, **kwargs):
        """Constructs BaseConfig."""
        # Drop the version info
        self.version = kwargs.pop("version", None)

        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def __setattr__(self, key, value):
        if key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        super().__setattr__(key, value)

    def __getattribute__(self, key):
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]
        return super().__getattribute__(key)
    
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
            raise ValueError(f"Can't find a configuration file at {config_file}")
        config_dict = cls._dict_from_json_file(config_file)
        if cls.base_config_key and cls.base_config_key in config_dict:
            config_dict = config_dict[cls.base_config_key]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            # sometimes the config has no `base_config_key` if the config is used in several composite models
            # e.g. LlamaConfig. In that case we try to see if there is match in `model_type` before raising a warning
            for k, v in config_dict.items():
                if isinstance(v, dict) and v.get("model_type") == cls.model_type:
                    config_dict = v

            # raise warning only if we still can't see a match in `model_type`
            if config_dict["model_type"] != cls.model_type:
                logger.warning(
                    f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                    f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
                )
        print(config_dict)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "BaseConfig":
        """
        Instantiates a [`BaseConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`BaseConfig`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)

        # Update config with kwargs if needed
        if "num_labels" in kwargs and "id2label" in kwargs:
            num_labels = kwargs["num_labels"]
            id2label = kwargs["id2label"] if kwargs["id2label"] is not None else []
            if len(id2label) != num_labels:
                raise ValueError(
                    f"You passed along `num_labels={num_labels}` with an incompatible id to label map: "
                    f"{kwargs['id2label']}. Since those arguments are inconsistent with each other, you should remove "
                    "one of them."
                )
        return config

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> "BaseConfig":
        """
        Instantiates a [`BaseConfig`] from the path to a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            [`BaseConfig`]: The configuration object instantiated from that JSON file.

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: Union[str, os.PathLike]):
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    
    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def __iter__(self):
        yield from self.__dict__

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type

        for key, value in output.items():
            # Deal with nested configs like CLIP
            if isinstance(value, BaseConfig):
                value = value.to_dict()

            output[key] = value

        self._remove_keys_not_serialized(output)

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=4, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def update(self, config_dict: dict[str, Any]):
        """
        Updates attributes of this class with attributes from `config_dict`.

        Args:
            config_dict (`dict[str, Any]`): Dictionary of attributes that should be updated for this class.
        """
        for key, value in config_dict.items():
            setattr(self, key, value)

    def update_from_string(self, update_str: str):
        """
        Updates attributes of this class with attributes from `update_str`.

        The expected format is ints, floats and strings as is, and for booleans use `true` or `false`. For example:
        "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"

        The keys to change have to already exist in the config object.

        Args:
            update_str (`str`): String with attributes that should be updated for this class.

        """

        d = dict(x.split("=") for x in update_str.split(","))
        for k, v in d.items():
            if not hasattr(self, k):
                raise ValueError(f"key {k} isn't in the original config dict")

            old_v = getattr(self, k)
            if isinstance(old_v, bool):
                if v.lower() in ["true", "1", "y", "yes"]:
                    v = True
                elif v.lower() in ["false", "0", "n", "no"]:
                    v = False
                else:
                    raise ValueError(f"can't derive true or false from {v} (key {k})")
            elif isinstance(old_v, int):
                v = int(v)
            elif isinstance(old_v, float):
                v = float(v)
            elif not isinstance(old_v, str):
                raise TypeError(
                    f"You can only update int, float, bool or string values in the config, got {v} for key {k}"
                )

            setattr(self, k, v)

    def _remove_keys_not_serialized(self, d: dict[str, Any]) -> None:
        """
        Checks and removes if there are any keys in the dict that should not be serialized when saving the config.
        Runs recursive check on the dict, to remove from all sub configs.
        """
        for value in d.values():
            if isinstance(value, dict):
                self._remove_keys_not_serialized(value)

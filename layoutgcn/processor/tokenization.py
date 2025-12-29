#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import json
import logging
from typing import Optional, Union

# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json"
}

# -----------------------------------------------------------Main-----------------------------------------------------------
class LayoutGCNTokenizer(object):
    """Tokenizer for LayoutGCN."""

    def __init__(self,
        vocab_file,
        max_seq_length=256,
        do_lower_case=True,
        do_pre_tokenize=False,
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token="<sep>"
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'."
            )
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        if pad_token is not None and pad_token not in self.encoder:
            self.encoder[pad_token] = len(self.encoder)
        if unk_token is not None and unk_token not in self.encoder:
            self.encoder[unk_token] = len(self.encoder)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.max_seq_length = max_seq_length
        self.do_lower_case = do_lower_case
        self.do_pre_tokenize = do_pre_tokenize
        self.unk_token = unk_token
        self.unk_token_id = self.encoder.get(unk_token)
        self.pad_token = pad_token
        self.pad_token_id = self.encoder.get(pad_token)
        self.sep_token = sep_token

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        vocab = dict(self.encoder).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        if self.do_lower_case:
            text = text.lower()
        if self.do_pre_tokenize:
            tokens = text.strip().split(self.sep_token)
        else:
            tokens = list(text.strip())
        return tokens
    
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    # Copied from transformers.models.roberta.tokenization_roberta.RobertaTokenizer._convert_id_to_token
    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)
    
    def tokenize(self, text: str, **kwargs) -> list[str]:
        if not text:
            return []
        return self._tokenize(text)

    def encode(self, text: str, **kwargs) -> list[str]:
        if not text:
            return []
        tokens = self._tokenize(text)
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        return token_ids

    def decode(self, token_ids: list[int], **kwargs) -> str:
        if not token_ids:
            return ""
        tokens = [self._convert_id_to_token(token_id) for token_id in token_ids]
        if self.do_pre_tokenize:
            text = self.sep_token.join(tokens)
        else:
            text = "".join(tokens)
        return text
 
    def __call__(self,
            text,
            padding: Union[bool] = True,
            truncation: Union[bool] = True,
            return_length: bool = False
        ):
        tokens = self._tokenize(text)
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        if truncation and self.max_seq_length and len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        length = len(token_ids)
        if padding and self.max_seq_length and len(token_ids) < self.max_seq_length:
            token_ids = token_ids + [self._convert_token_to_id(self.pad_token)] * (self.max_seq_length - len(token_ids))
        if return_length:
            return token_ids, length
        return token_ids

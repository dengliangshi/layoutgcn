#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import os
import json
import uuid
import logging
from typing import Optional
from dataclasses import dataclass, field

# Third-party libraries
import torch
import evaluate
import numpy as np
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset

# User define module
from layoutgcn.model.configuration import LayoutGCNConfig
from layoutgcn.processor.configuration import DocProcessorConfig
from layoutgcn.processor.doc_processing import LayougGCNDocProcessor


# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger("LayoutGCN")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# -----------------------------------------------------------Main-----------------------------------------------------------
@dataclass
class DatasetArguments:
    data_dir: str = field(
        metadata={"help": "Path to the directory containing the dataset"}
    )

    task_type: str = field(
        default="document_classification",
        metadata={"help": "The type of task. Defaults to 'document_classification' if not provided."}
    )

    max_seq_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum sequence length. Defaults to 256 if not provided."
        },
    )

    max_num_nodes: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum number of nodes. Defaults to 256 if not provided."
        },
    )

    do_pre_tokenize: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to pre-tokenize the input text. Defaults to False if not provided."
        },
    ) 
    
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to lowercase the input text. Defaults to True if not provided."
        },
    )
    
    radical_alpha: Optional[int] = field(
        default=50,
        metadata={
            "help": "The alpha parameter for the radical-based tokenization. Defaults to 50 if not provided."
        },
    )
    
    angle_delta: Optional[int] = field(
        default=10,
        metadata={
            "help": "The angle delta parameter for the angle-based tokenization. Defaults to 10 if not provided."
        },
    )
    
    sep_token: Optional[str] = field(
        default=" ",
        metadata={
            "help": "The separator token for the tokenizer. Defaults to space if not provided."
        },
    )
    
    unk_token: Optional[str] = field(
        default="<unk>",
        metadata={
            "help": "The unknown token for the tokenizer. Defaults to <unk> if not provided."
        },
    )
    
    pad_token: Optional[str] = field(
        default="<pad>",
        metadata={
            "help": "The padding token for the tokenizer. Defaults to <pad> if not provided."
        },
    )
    
    use_image: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use image features. Defaults to False if not provided."
        },
    )
    
    efficientnet_model_path: Optional[str] = field(
        default="./efficientnet-b0",
        metadata={
            "help": "The path to the efficientnet model. Defaults to ./efficientnet-b0 if not provided."
        },
    )
    
    shuffle_prob: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "The probability of shuffling the image features. Defaults to 0.2 if not provided."
        },
    )



def main():
    
    parser = HfArgumentParser((DatasetArguments, TrainingArguments))
    dataset_args, training_args = parser.parse_args_into_dataclasses()

    def make_building_map_fn(dataset_args, vocabulary, mappings, num_nodes, sequence_lengths, label_counts):

        def build_vocab_and_mappings(example):
            blocks = json.loads(example["blocks"])
            for block in blocks:
                # Build vocab
                if dataset_args.do_pre_tokenize:
                    tokens = block["content"].strip().split(dataset_args.sep_token)
                else:
                    tokens = list(block["content"].strip())
                sequence_lengths.append(len(tokens))
                for token in tokens:
                    if dataset_args.do_lower_case:
                        token = token.lower()
                    if token in vocabulary:
                        continue
                    vocabulary[token] = len(vocabulary)

                # Node classification task
                if dataset_args.task_type == "node_classification":
                    if block.get("category") is None:
                        continue
                    if block["category"] not in mappings:
                        mappings[block["category"]] = len(mappings) + 1
                    if block.get("category") not in label_counts:
                        label_counts[block["category"]] = 0
                    label_counts[block["category"]] += 1
                # Information extraction task
                if dataset_args.task_type == "information_extraction":
                    for label in block.get("labels", []):
                        if label.get("id") is None:
                            label["id"] = str(uuid.uuid4()).replace("-", "")
                        if label.get("category") is None:
                            raise ValueError("Category is required for information extraction task.")
                        if label["category"] not in mappings:
                            mappings[label["category"]] = 2 * len(mappings) + 1
                        if label["category"] not in label_counts:
                            label_counts[label["category"]] = 0
                        label_counts[label["category"]] += 1
            # Document classification task
            if dataset_args.task_type == "document_classification":
                if example.get("category") is None:
                    raise ValueError("Category is required for document classification task.")
                if example["category"] not in mappings:
                    mappings[example["category"]] = len(mappings)
                if example.get("category") not in label_counts:
                    label_counts[example["category"]] = 0
                label_counts[example["category"]] += 1
            example["blocks"] = json.dumps(blocks, ensure_ascii=False)
            # number of nodes
            num_nodes.append(len(blocks))
            return example

        return build_vocab_and_mappings

    # pre-process data
    raw_datasets = load_dataset("json", data_dir=dataset_args.data_dir)
    datasets = raw_datasets.map(LayougGCNDocProcessor.pre_process, batched=False, load_from_cache_file=False)

    mappings = {}
    vocabulary = {}
    num_nodes = []
    sequence_lengths = []
    label_count = {}
    # add special tokens
    if dataset_args.pad_token is not None:
        vocabulary[dataset_args.pad_token] = len(vocabulary)
    if dataset_args.unk_token is not None:
        vocabulary[dataset_args.unk_token] = len(vocabulary)
    # build vocabulary and mappings
    datasets["train"].map(
        function=make_building_map_fn(dataset_args, vocabulary, mappings, num_nodes, sequence_lengths, label_count),
        batched=False,
        load_from_cache_file=False
    )
    logger.info(f"Vocabulary size: {len(vocabulary)}.")
    logger.info(f"Maximum number of nodes: {max(num_nodes)}, average number of nodes: {sum(num_nodes) / len(num_nodes)}.")
    logger.info(f"Maximum length of sequence: {max(sequence_lengths)}, average length of sequence: {sum(sequence_lengths) / len(sequence_lengths)}.")
    logger.info(f"Number of samples for each category: {", ".join([f"{k}-{v}" for k, v in label_count.items()])}.")

    # create output directory if not exists
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    model_path = os.path.join(training_args.output_dir, "final_model")
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # save vocabulary
    vocab_file = os.path.join(training_args.output_dir, "final_model/vocab.json")
    with open(vocab_file, "w", encoding="utf-8") as fp:
        json.dump(vocabulary, fp, ensure_ascii=False, indent=4)

    # configuration for processor
    processor_config = DocProcessorConfig(
        vocab_file=vocab_file,
        max_seq_length=dataset_args.max_seq_length,
        max_num_nodes=dataset_args.max_num_nodes,
        do_lower_case=dataset_args.do_lower_case,
        do_pre_tokenize=dataset_args.do_pre_tokenize,
        radical_alpha=dataset_args.radical_alpha,
        angle_delta=dataset_args.angle_delta,
        sep_token=dataset_args.sep_token,
        unk_token=dataset_args.unk_token,
        pad_token=dataset_args.pad_token,
        label2id=mappings,
        use_image=dataset_args.use_image,
        efficientnet_model_path=dataset_args.efficientnet_model_path,
        shuffle_prob=dataset_args.shuffle_prob,
        task_type=dataset_args.task_type
    )
    processor_config.to_json_file(os.path.join(training_args.output_dir, "final_model/processor_config.json"))


if __name__ == "__main__":
    main()

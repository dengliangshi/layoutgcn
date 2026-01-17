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
from layoutgcn.model.modeling import LayoutGCNForNodeClassification
from layoutgcn.model.modeling import LayoutGCNForDocClassification
from layoutgcn.model.modeling import LayoutGCNForInfoExtraction
from layoutgcn.model.modeling import LayoutGCNForLinkPrediction
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


@dataclass
class ModelArguments:
    
    model_name: Optional[str] = field(
        default="LayoutGCN",
        metadata={
            "help": "The name of model. Defaults to 'LayoutGCN' if not provided."
        },
    )

    num_word_embeddings: Optional[int] = field(
        default=None,
        metadata={
            "help": "The size of the word embedding. Defaults to None if not provided."
        },
    )

    padding_token_idx: Optional[int] = field(
        default=0,
        metadata={
            "help": "The index of the padding token. Defaults to 0 if not provided."
        },
    )

    hidden_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "The hidden size of the model. Defaults to 256 if not provided."
        },
    )

    filter_sizes: Optional[list[int]] = field(
        default_factory=lambda:[2, 3, 4, 5],
        metadata={
            "help": "The filter sizes of the model. Defaults to [2, 3, 4, 5] if not provided."
        },
    )

    num_filters: Optional[list[int]] = field(
        default_factory=lambda:[32, 64, 128, 256],
        metadata={
            "help": "The number of filters of the model. Defaults to [32, 64, 128, 256] if not provided."
        },
    )

    embedding_dim: Optional[int] = field(
        default=128,
        metadata={
            "help": "The embedding dimension of the model. Defaults to 128 if not provided."
        },
    )

    dropout_prob: Optional[float] = field(
        default=0.6,
        metadata={
            "help": "The dropout probability of the model. Defaults to 0.6 if not provided."
        },
    )

    num_angle_embeddings: Optional[int] = field(
        default=36,
        metadata={
            "help": "The number of angle embeddings. Defaults to 36 if not provided."
        },
    )

    roi_pooling_size: Optional[int] = field(
        default=7,
        metadata={
            "help": "The size of the roi pooling. Defaults to 7 if not provided."
        },
    )

    num_labels: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of labels. Defaults to None if not provided."
        },
    )


def main():
    
    parser = HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments))
    dataset_args, model_args, training_args = parser.parse_args_into_dataclasses()

    model_path = os.path.join(training_args.output_dir, "final_model")

    # Load processor config and build processor
    processor_config_file = os.path.join(training_args.output_dir, "final_model/processor_config.json")
    processor_config = DocProcessorConfig.from_json_file(processor_config_file)
    doc_processor = LayougGCNDocProcessor(processor_config)

    # Load dataset
    datasets = load_dataset("json", data_dir=dataset_args.data_dir)

    # Build model config and model
    model_config = LayoutGCNConfig(
        model_name=model_args.model_name,
        num_word_embeddings=doc_processor.vocab_size,
        padding_token_idx=doc_processor.padding_idx,
        max_seq_length=processor_config.max_seq_length,
        max_num_nodes=processor_config.max_num_nodes,
        hidden_size=model_args.hidden_size,
        filter_sizes=model_args.filter_sizes,
        num_filters=model_args.num_filters,
        embedding_dim=model_args.embedding_dim,
        dropout_prob=model_args.dropout_prob,
        num_angle_embeddings=model_args.num_angle_embeddings,
        use_image=dataset_args.use_image,
        efficientnet_model_path=dataset_args.efficientnet_model_path,
        roi_pooling_size=model_args.roi_pooling_size,
        num_labels=processor_config.num_labels
    )
    model_config.to_json_file(os.path.join(training_args.output_dir, "final_model/model_config.json"))
    if dataset_args.task_type == "doc_classification":
        model_cls = LayoutGCNForDocClassification
    elif dataset_args.task_type == "node_classification":
        model_cls = LayoutGCNForNodeClassification
    elif dataset_args.task_type == "information_extraction":
        model_cls = LayoutGCNForInfoExtraction
    elif dataset_args.task_type == "link_prediction":
        model_cls = LayoutGCNForLinkPrediction
    else:
        raise ValueError(f"Task type {dataset_args.task_type} is not supported.")
    model = model_cls(config=model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def train_transform(batched_examples):
        def process_fn(example):
            output = doc_processor(
                blocks=json.loads(example["blocks"]),
                image=example.get("image"),
                height=example.get("height"),
                width=example.get("width"),
                category=example.get("category"),
                padding=True,
                truncation=True,
                is_training=True
            )
            return output

        processed_batch = {}

        keys = list(batched_examples.keys())
        batch_size = len(batched_examples[keys[0]])
        for index in range(batch_size):
            example = {key: batched_examples[key][index] for key in keys}
            processed_example = process_fn(example)
            for key, value in processed_example.items():
                if key not in processed_batch:
                    processed_batch[key] = []
                processed_batch[key].append(value)
        return processed_batch

    def valid_transform(batched_examples):
        def process_fn(example):
            output = doc_processor(
                blocks=json.loads(example["blocks"]),
                image=example.get("image"),
                height=example.get("height"),
                width=example.get("width"),
                category=example.get("category"),
                padding=True,
                truncation=True,
                is_training=False
            )
            return output

        processed_batch = {}

        keys = list(batched_examples.keys())
        batch_size = len(batched_examples[keys[0]])
        for index in range(batch_size):
            example = {key: batched_examples[key][index] for key in keys}
            processed_example = process_fn(example)
            for key, value in processed_example.items():
                if key not in processed_batch:
                    processed_batch[key] = []
                processed_batch[key].append(value)
        return processed_batch

    # Get Dataset
    datasets["train"].set_transform(train_transform)
    #datasets = datasets.map(process_fn, batched=False, remove_columns=column_names, load_from_cache_file=False)
    datasets["validation"].set_transform(valid_transform)

    current_path, _ = os.path.split(os.path.abspath(__file__))
    metric = evaluate.load(os.path.join(current_path, "metrics/accuracy.py"))

    def compute_metrics(p):
        if dataset_args.task_type == "doc_classification":
            result = metric.compute(
                predictions=np.argmax(p.predictions[0], axis=-1),
                references=p.label_ids
            )
        elif dataset_args.task_type == "node_classification":
            mask = p.predictions[2].reshape([-1])
            predictions = np.argmax(p.predictions[1], axis=-1).reshape([-1])
            references = p.label_ids.reshape([-1])
            result = metric.compute(
                predictions=predictions[mask > 0],
                references=references[mask > 0]
            )
        elif dataset_args.task_type == "information_extraction":
            mask = p.predictions[2].reshape([-1])
            predictions = np.argmax(p.predictions[1], axis=-1).reshape([-1])
            references = p.label_ids.reshape([-1])
            result = metric.compute(
                predictions=predictions[mask > 0],
                references=references[mask > 0]
            )

        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result
    
    training_args.remove_unused_columns = False
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if training_args.do_train else None,
        eval_dataset=datasets["validation"] if training_args.do_eval else None,
        data_collator=None,
        compute_metrics=compute_metrics
    )
    training_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.log_metrics("train", training_result.metrics)
    trainer.save_model(model_path)

    eval_result = trainer.evaluate(
        eval_dataset=datasets["test"] if training_args.do_eval else None,
        metric_key_prefix="test"
    )
    trainer.log_metrics("test", eval_result)

if __name__ == "__main__":
    main()

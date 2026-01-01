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
import argparse
import inspect
from functools import partial
from tqdm.auto import tqdm
from typing import Dict, List, Any

# Third-party libraries
import torch
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

# User define module
from layoutgcn.evaluator import Evaluator
from layoutgcn.processor.configuration import DocProcessorConfig
from layoutgcn.processor.doc_processing import LayougGCNDocProcessor
from layoutgcn.model.modeling import LayoutGCNForDocClassification
from layoutgcn.model.modeling import LayoutGCNForInfoExtraction
from layoutgcn.model.modeling import LayoutGCNForNodeClassification
from layoutgcn.model.modeling import LayoutGCNForLinkPrediction


# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger("LayoutGCN")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# -----------------------------------------------------------Main-----------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="LayoutGCN Predictor")
    parser.add_argument("--model-path", type=str, default=None, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data-dir", type=str, default=None, required=True, help="Path to the dataset.")
    parser.add_argument("--output-dir", type=str, default=None, required=True, help="Path to the output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    return parser.parse_args()


def process_fn(processor, sample):
    features = processor(
        blocks=json.loads(sample["blocks"]),
        image=sample.get("image"),
        height=sample.get("height"),
        width=sample.get("width"),
        category=sample.get("category"),
        padding=True,
        truncation=True
    )
    features["id"] = sample["id"]
    return features

def post_process(processor, sample, column_names):
    output = processor.post_process(
        probabilities=sample["probability"],
        mask=sample["mask"],
        blocks=json.loads(sample["blocks"])
    )
    result = {key: value for key, value in sample.items() if key in column_names}
    result.update(output)
    return result

def main(args):
    
    processor_config = DocProcessorConfig.from_model_path(args.model_path)
    processor = LayougGCNDocProcessor(processor_config)

    evaluator = Evaluator(task_type=processor_config.task_type)

    logger.info(f"Model is loaded from {args.model_path}.")
    if processor_config.task_type == "classification":
        model = LayoutGCNForDocClassification.load_from_model_path(args.model_path)
    elif processor_config.task_type == "information_extraction":
        model = LayoutGCNForInfoExtraction.load_from_model_path(args.model_path)
    elif processor_config.task_type == "node_classification":
        model = LayoutGCNForNodeClassification.load_from_model_path(args.model_path)
    elif processor_config.task_type == "link_prediction":
        model = LayoutGCNForLinkPrediction.load_from_model_path(args.model_path)
    else:
        raise ValueError(f"Task type {processor_config.task_type} is not supported.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Model loaded on {device}.")
    model.to(device).eval()

    def pre_process(sample):
        sample = LayougGCNDocProcessor.pre_process(sample)
        return sample
    logger.info(f"Dataset loaded from {args.data_dir}.")
    datasets = load_dataset("json", data_dir=args.data_dir)
    if isinstance(datasets, Dataset):
        dataset = datasets
    else:
        dataset = concatenate_datasets([datasets[key] for key in datasets if key =="test"])
    logger.info(f"Pre-processing dataset...")
    preprocessed_dataset = dataset.map(pre_process, batched=False, load_from_cache_file=False)
    column_names = preprocessed_dataset.column_names

    results = []
    fn_args = inspect.signature(model.forward).parameters.keys()
    with torch.no_grad():
        for start_idx in tqdm(range(0, len(preprocessed_dataset), args.batch_size), desc="Inference"):
            end_index = min(start_idx + args.batch_size, len(preprocessed_dataset))
            batch_dataset = preprocessed_dataset.select(range(start_idx, end_index))
            batch_dataset = batch_dataset.map(partial(process_fn, processor), batched=False, desc="Batch").with_format("torch")
            inputs = {
                key: batch_dataset[key].to(device)
                for key in batch_dataset.features if key in fn_args
            }
            output = model(**inputs, return_dict=True)
            probabilities = output.probabilities.detach().cpu().numpy().tolist()
            masks = output.mask.detach().cpu().numpy().tolist()
            for index in range(len(batch_dataset)):
                row = {key: batch_dataset[key][index] for key in batch_dataset.features}
                row["probability"] = probabilities[index]
                row["mask"] = masks[index]
                result = post_process(processor, row, column_names)
                results.append(result)

    eval_result = evaluator.evaluate(results)
    print(json.dumps(eval_result, ensure_ascii=False, indent=4))
    dataset = Dataset.from_list(results)
    dataset.to_json(os.path.join(args.output_dir, "result.json"), force_ascii=False)


if __name__ == "__main__":
    args = get_args()
    main(args)

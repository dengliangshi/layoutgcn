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
from tqdm.auto import tqdm
from typing import Dict, List, Any

# Third-party libraries
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset, concatenate_datasets

# User define module
from layoutgcn.evaluator import Evaluator
from layoutgcn.processor.configuration import DocProcessorConfig
from layoutgcn.processor.doc_processing import LayoutGCNDocProcessor
from layoutgcn.model.modeling import (
    LayoutGCNForDocClassification,
    LayoutGCNForInfoExtraction,
    LayoutGCNForNodeClassification,
    LayoutGCNForLinkPrediction
)


# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger("LayoutGCN")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

MODEL_CLASSES = {
    "classification": LayoutGCNForDocClassification,
    "information_extraction": LayoutGCNForInfoExtraction,
    "node_classification": LayoutGCNForNodeClassification,
    "link_prediction": LayoutGCNForLinkPrediction
}

# -----------------------------------------------------------Main-----------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="LayoutGCN Predictor")
    parser.add_argument("--model-path", type=str, default=None, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--data-dir", type=str, default=None, required=True, help="Path to the dataset.")
    parser.add_argument("--output-dir", type=str, default=None, required=True, help="Path to the output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--do-evaluate", action="store_true", default=False, help="Whether to evaluate the model.")
    return parser.parse_args()


def process_fn(processor, sample):
    features = processor(
        blocks=json.loads(sample["blocks"]),
        image=sample.get("image"),
        height=sample.get("height"),
        width=sample.get("width"),
        category=sample.get("category"),
        padding=True,
        truncation=True,
        return_tensors=True
    )
    features["id"] = sample["id"]
    return features


def post_process(processor, sample, column_names):
    output = processor.post_process(
        predictions=sample.get("predictions"),
        probabilities=sample["probabilities"],
        mask=sample["mask"],
        blocks=json.loads(sample["blocks"])
    )
    result = {key: value for key, value in sample.items() if key in column_names}
    result.update(output)
    return result


def load_model(model_path: str, task_type: str, device: torch.device):
    if task_type not in MODEL_CLASSES:
        raise ValueError(f"Task type {task_type} is not supported. Supported task types are: {list(MODEL_CLASSES.keys())}.")
    logger.info(f"Loading model for task type: {task_type} from {model_path}")
    model_cls = MODEL_CLASSES[task_type]
    model = model_cls.load_from_model_path(model_path)
    model.to(device).eval()
    return model


def load_and_preprocess_dataset(data_dir: str):
    logger.info(f"Loading dataset from {data_dir}.")
    datasets = load_dataset("json", data_dir=data_dir)
    if isinstance(datasets, Dataset):
        dataset = datasets
    else:
        test_datasets = [datasets[key] for key in datasets if key =="test"]
        if test_datasets:
            dataset = concatenate_datasets(test_datasets)
        else:
            dataset = concatenate_datasets([datasets[key] for key in datasets])
    logger.info(f"Pre-processing dataset...")
    return dataset.map(LayoutGCNDocProcessor.pre_process, batched=False, load_from_cache_file=False)


def collate_fn(batch, processor, fn_args):
    processed = [process_fn(processor, sample) for sample in batch]

    collated = {}
    for key in processed[0].keys():
        if key not in fn_args:
            continue
        values = [sample[key] for sample in processed]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.cat(values, dim=0)
        else:
            collated[key] = values
    return batch, collated


def run_inference(model, processor, dataset, batch_size, device, task_type, column_names):
    results = []
    fn_args = model.forward.__code__.co_varnames

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor, fn_args)
    )
    with torch.no_grad():
        for original_batch, batch_inputs in tqdm(dataloader, desc="Inference"):
            inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch_inputs.items() if key in fn_args}
            outputs = model(**inputs, return_dict=True)
            predictions = outputs.predictions.detach().cpu().numpy() if hasattr(outputs, "predictions") else None
            probabilities = outputs.probabilities.detach().cpu().numpy()
            masks = outputs.mask.detach().cpu().numpy()

            for index in range(len(original_batch)):
                row = {
                    "predictions": predictions[index] if predictions is not None else None,
                    "probabilities": probabilities[index],
                    "mask": masks[index],
                    "blocks": original_batch[index]["blocks"]
                }
                result = post_process(processor, row, column_names)
                for key in column_names:
                    if key not in result and key not in original_batch[index]:
                        result[key] = original_batch[index][key]
                results.append(result)

            if len(results) % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
    return results


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}.")

    # Load processor and model
    processor_config = DocProcessorConfig.from_model_path(args.model_path)
    processor = LayoutGCNDocProcessor(processor_config)
    model = load_model(args.model_path, processor_config.task_type, device)

    # Load dataset
    dataset = load_and_preprocess_dataset(args.data_dir)
    column_names = dataset.column_names

    # Run inference
    results = run_inference(
        model=model,
        processor=processor,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        task_type=processor_config.task_type,
        column_names=column_names
    )

    # Evaluate if needed
    if args.do_evaluate:
        evaluator = Evaluator(task_type=processor_config.task_type)
        eval_result = evaluator.evaluate(results)
        json_str = json.dumps(eval_result, ensure_ascii=False, indent=4)
        summary_file = os.path.join(args.output_dir, "summary.json")
        with open(summary_file, "w") as fp:
            fp.write(json_str)
        logger.info("Evaluation Results:\n" + json_str)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "result.json")
    Dataset.from_list(results).to_json(output_path, force_ascii=False)
    logger.info(f"Results saved to {output_path}.")


if __name__ == "__main__":
    args = get_args()
    main(args)

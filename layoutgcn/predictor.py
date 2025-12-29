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
from tqdm.auto import tqdm

# Third-party libraries
import torch
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets

# User define module
from layoutgcn.evaluator import Evaluator
from layoutgcn.processor.configuration import DocProcessorConfig
from layoutgcn.processor.doc_processing import LayougGCNDocProcessor
from layoutgcn.model.modeling import LayoutGCNForInfoExtraction


# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger("LayoutGCN")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default=None, help="Path to the model checkpoint.")
parser.add_argument("--data-dir", type=str, default=None, help="Path to the dataset.")
parser.add_argument("--output-dir", type=str, default=None, help="Path to the output directory.")
parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")

# -----------------------------------------------------------Main-----------------------------------------------------------
def main(args):
    processor_config = DocProcessorConfig.from_model_path(args.model_path)
    processor = LayougGCNDocProcessor(processor_config)

    evaluator = Evaluator(task_type=processor_config.task_type)

    model = LayoutGCNForInfoExtraction.load_from_model_path(args.model_path)
    model.eval()

    datasets = load_dataset("json", data_dir=args.data_dir)

    def pre_process(sample):
        sample = LayougGCNDocProcessor.pre_process(sample)
        return sample

    preprocessed_datasets = datasets.map(pre_process, batched=False, load_from_cache_file=False)

    def process_fn(sample):
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

    processed_datasets = preprocessed_datasets.map(process_fn, batched=False)

    concatenated_dataset = concatenate_datasets([processed_datasets[key] for key in processed_datasets])

    torch_dataset = concatenated_dataset.with_format("torch")

    results = []
    with torch.no_grad():
        for features in torch_dataset.batch(batch_size=args.batch_size):
            fn_args = inspect.signature(model.forward).parameters.keys()
            inputs = {key: features[key] for key in features.keys() if key in fn_args}
            output = model(**inputs, return_dict=True)
            probabilities = output.probabilities.numpy().tolist()
            masks = output.mask.numpy().tolist()
            for id, probability, mask in zip(features["id"], probabilities, masks):
                results.append({
                    "id": id,
                    "probability": probability,
                    "mask": mask
                })

    dataframes = concatenate_datasets([preprocessed_datasets[key] for key in preprocessed_datasets]).to_pandas()
    results = pd.DataFrame(results)
    results["id"].astype(object)
    dataframes = dataframes.join(results.set_index("id"), on="id")

    def post_process(sample):
        output = processor.post_process(
            probabilities=sample["probability"],
            mask=sample["mask"],
            blocks=json.loads(sample["blocks"])
        )
        sample.update(output)
        del sample["probability"]
        del sample["mask"]
        return output

    dataset = Dataset.from_pandas(dataframes).map(post_process, batched=False)
    samples = [sample for sample in dataset]
    eval_result = evaluator.evaluate(samples)
    print(json.dumps(eval_result, ensure_ascii=False, indent=4))
    dataset = Dataset.from_list(samples)
    dataset.to_json(os.path.join(args.output_dir, "result.json"), force_ascii=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

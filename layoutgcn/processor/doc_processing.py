#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import re
import json
import math
import uuid
import copy
import logging
import random
from typing import Optional, Union, List

# Third-party libraries
import torch
import numpy as np
from PIL import Image
from transformers import EfficientNetImageProcessor

# User define module
from .tokenization import LayoutGCNTokenizer
from .configuration import DocProcessorConfig

# ------------------------------------------------------Global Variables----------------------------------------------------
logger = logging.getLogger("LayoutGCN")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# -----------------------------------------------------------Main-----------------------------------------------------------
class LayoutGCNDocProcessor(object):


    tokenizer_class = "LayoutGCNTokenizer"
    image_processor_class = "EfficientNetImageProcessor"
    
    model_input_names = ["token_ids", "sequence_length", "pixel_values", "boxes", "num_nodes", "layout_features", "adj_angle", "adj_radical_dist", "labels"]

    def __init__(self, config: DocProcessorConfig):
        """See base class."""
        self.config = config
        
        # Image processor
        self.image_processor = (EfficientNetImageProcessor.from_pretrained(config.efficientnet_model_path)
            if self.config.use_image else None
        )

        # Tokenizer
        self.tokenizer = LayoutGCNTokenizer(
            vocab_file=config.vocab_file,
            max_seq_length=config.max_seq_length,
            do_lower_case=config.do_lower_case,
            do_pre_tokenize=config.do_pre_tokenize,
            unk_token=config.unk_token,
            pad_token=config.pad_token,
            sep_token=config.sep_token
        )
        self.vocab_size = self.tokenizer.vocab_size
        self.padding_idx = self.tokenizer.pad_token_id

    @classmethod
    def pre_process(cls, example):

        # Generate id for each example
        if example.get("id") is None:
            example["id"] = str(uuid.uuid4()).replace("-", "")
        original_blocks = json.loads(example["blocks"])
        
        filtered_blocks = []
        
        for block in original_blocks:
            # Skip empty and low score block
            content = block.get("content", "").strip()
            if not content or block.get("score", 1.0) < 0.2:
                continue

            # Skip invalid bndbox
            x_list = [point[0] for point in block.get("bndbox", [])]
            y_list = [point[1] for point in block.get("bndbox", [])]
            width = max(x_list) - min(x_list)
            height = max(y_list) - min(y_list)
            
            if width <=0 or height <=0 or height / width > 5:
                continue
            
            # Generate id for each block
            if block.get("id") is None:
                block["id"] = str(uuid.uuid4()).replace("-", "")

            filtered_blocks.append(block)

        example["blocks"] = json.dumps(filtered_blocks, ensure_ascii=False)
 
        return example

    def _calculate_doc_size(self, blocks: List[dict]):

        x_max = max([block["bndbox"][2][0] for block in blocks])
        y_max = max([block["bndbox"][2][1] for block in blocks])

        return 1.1 * y_max, 1.1 * x_max

    def _augment_blocks(self, blocks: List[dict]):

        copied_blocks = copy.deepcopy(blocks)

        if self.config.shuffle_prob > 0 and random.random() < self.config.shuffle_prob:
            random.shuffle(copied_blocks)

        if self.config.dropped_prob > 0 and random.random() < self.config.dropped_prob:
            copied_blocks = [block for block in copied_blocks if random.random() > self.config.dropped_blocks_prob]

        if not copied_blocks:
            copied_blocks = blocks

        if self.config.fake_size_prob > 0 and random.random() < self.config.fake_size_prob:
            height, width = self._calculate_doc_size(copied_blocks)
        else:
            height, width = None, None

        return copied_blocks, height, width

    def _get_angle_adj(self, bndboxes: np.ndarray, adj_mask: np.ndarray, delta: int = 10):

        # Calculate the center point
        center_y = bndboxes[:, [1, 3]].mean(axis=1)
        center_x = bndboxes[:, [0, 2]].mean(axis=1)
        center_xy = np.stack([center_x, center_y], axis=-1)

        bndboxes_i = bndboxes[:, None, :]
        bndboxes_j = bndboxes[None, :, :]

        i_left_top = np.maximum(bndboxes_i[..., :2], bndboxes_j[..., :2])
        i_right_bottom = np.minimum(bndboxes_i[..., 2:], bndboxes_j[..., 2:])
        intersection = np.maximum(i_right_bottom - i_left_top, 0)

        size_i = bndboxes_i[..., 2:] - bndboxes_i[..., :2]
        size_j = bndboxes_j[..., 2:] - bndboxes_j[..., :2]
        min_size = np.minimum(size_i, size_j)

        overlap_ratio = np.divide(
            intersection,
            min_size,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=min_size > 0
        )
        overlap_flag = (intersection > 0) & (overlap_ratio > 0.5)

        delta_xy = center_xy[None, :, :] - center_xy[:, None, :]
        delta_xy[overlap_flag] = 0

        rad = np.arctan2(delta_xy[..., 1], delta_xy[..., 0])
        angle = np.rad2deg(rad)

        angle = np.where(angle < 0, angle + 360, angle)

        return (angle * adj_mask / delta).astype(np.int32)

    def __call__(self,
            blocks: Optional[List[dict]] = None,
            image: Optional[Image.Image] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            category: Optional[Union[str, List[str]]] = None,
            padding: Union[bool] = True,
            truncation: Union[bool] = True,
            is_training: Optional[bool] = False,
            return_tensors: Union[bool] = False
        ):
        outputs = {}

        # Check inputs
        if blocks is None:
            raise ValueError(f"You need to provide at least one input to call {self.__class__.__name__}")

        # Apply data augmentation if training
        if is_training:
            blocks, height, width = self._augment_blocks(blocks)
       
       # Truncate the number of nodes
        if truncation and self.config.max_num_nodes is not None:
            blocks = blocks[:self.config.max_num_nodes]

        # Calculate document size if not provided
        if height is None or width is None:
            height, width = self._calculate_doc_size(blocks)

        for block in blocks:
            y_list = [point[1] for point in block["bndbox"]]
            offset = (max(y_list) - min(y_list)) * 0.1
            block["bndbox"][0][1] += offset
            block["bndbox"][1][1] += offset
            block["bndbox"][2][1] += offset
            block["bndbox"][3][1] += offset
        
        # Get the number of nodes
        num_nodes = len(blocks)
        outputs["num_nodes"] = torch.tensor([num_nodes], dtype=torch.int32) if return_tensors else num_nodes
        
        # Tokenize the text in each block
        token_ids = []
        sequence_length = []
        for block in blocks:
            block_token_ids, length = self.tokenizer(
                text=block["content"],
                padding=padding,
                truncation=truncation,
                return_length=True
            )
            token_ids.append(block_token_ids)
            sequence_length.append(length)
        
        if padding:
            padding_count = self.config.max_num_nodes - num_nodes
            padding_token_id_row = [self.tokenizer.pad_token_id, ] * self.config.max_seq_length
            token_ids.extend([padding_token_id_row, ] * padding_count)
            sequence_length.extend([0, ] * padding_count)
        outputs["token_ids"] = np.array(token_ids, dtype=np.int32)
        outputs["sequence_lengths"] = np.array(sequence_length, dtype=np.int32)

        if return_tensors:
            outputs["token_ids"] = torch.tensor([outputs["token_ids"]], dtype=torch.int32)
            outputs["sequence_lengths"] = torch.tensor([outputs["sequence_lengths"]], dtype=torch.int32)

        # Get the layout features
        boxes = np.array([
            [
                block["bndbox"][0][0], block["bndbox"][0][1],
                block["bndbox"][2][0], block["bndbox"][2][1]
            ] for block in blocks
        ], dtype=np.float32)

        # Padding boxes
        if padding and num_nodes < self.config.max_num_nodes:
            padding_boxes = np.zeros((self.config.max_num_nodes - num_nodes, 4), dtype=np.float32)
            boxes = np.vstack([boxes, padding_boxes])

        # Calculate layout features
        position = boxes / np.array([width, height, width, height])
        size = (boxes[..., 2:] - boxes[..., :2]) / np.array([width, height])
        shape = (boxes[..., 2] - boxes[..., 0]) / np.clip(boxes[..., 3] - boxes[..., 1], 1, None) / width
        layout_features = np.concatenate([position, size, np.expand_dims(shape, axis=-1)], axis=-1).astype(np.float32)

        # Apply mask
        mask = (np.arange(self.config.max_num_nodes) < num_nodes).reshape(-1, 1).astype(np.float32)
        outputs["layout_features"] = layout_features * mask

        if return_tensors:
            outputs["layout_features"] = torch.tensor([outputs["layout_features"]], dtype=torch.float32)
        
        # Get the adjacency mask
        adj_mask = mask.reshape(-1, 1) @ mask.reshape(1, -1)

        # Angle adjacency
        adj_angle = self._get_angle_adj(boxes, adj_mask=adj_mask, delta=self.config.angle_delta)
        outputs["adj_angle"] = torch.tensor([adj_angle], dtype=torch.int32) if return_tensors else adj_angle

        # Get the adjacency distance matrix
        dist_norm = math.sqrt(width ** 2 + height ** 2)
        boxes_i = boxes[:, None, :]
        boxes_j = boxes[None, :, :]

        left_top = boxes_i[..., :2] - boxes_j[..., 2:]
        right_bottom = boxes_j[..., :2] - boxes_i[..., 2:]
        delta_x = np.maximum(left_top[..., 0], right_bottom[..., 0])
        delta_y = np.maximum(left_top[..., 1], right_bottom[..., 1])
        
        distance = np.sqrt(np.square(np.clip(delta_x, 0, None)) + np.square(np.clip(delta_y, 0, None)))
        outputs["adj_radical_dist"] = np.exp(-1 * self.config.radical_alpha * distance / dist_norm) * adj_mask
        
        if return_tensors:
            outputs["adj_radical_dist"] = torch.tensor([outputs["adj_radical_dist"]], dtype=torch.float32)

        # Process image
        if self.config.use_image and image is not None:
            width, height = image.size
            image_features = self.image_processor(image.convert("RGB"))
            w_resized_ratio = self.image_processor.size["width"] / width
            h_resized_ratio = self.image_processor.size["height"] / height
            outputs["pixel_values"] = (torch.tensor([image_features["pixel_values"]], dtype=torch.float32)
                if return_tensors else image_features["pixel_values"])
            resized_boxes = boxes * np.array([w_resized_ratio, h_resized_ratio, w_resized_ratio, h_resized_ratio])
            outputs["rois"] = torch.tensor([resized_boxes], dtype=torch.float32) if return_tensors else resized_boxes

        outputs["labels"] = self._process_labels(blocks, category, padding, truncation)

        if outputs["labels"] is not None and return_tensors:
            outputs["labels"] = torch.tensor([outputs["labels"]], dtype=torch.int64)

        return outputs

    def _process_labels(self, blocks: List[dict], category: Optional[str]=None, padding: bool=True, truncation: bool=True):

        if category is not None and self.config.task_type == "document_classification":
            return self.config.label2id.get(category)
        
        elif self.config.task_type == "node_classification":
            labels = [self.config.label2id.get(block.get("category"), 0) for block in blocks]
            if padding:
                labels = labels + [-100, ] * (self.config.max_num_nodes - len(blocks))
            return labels
        
        elif self.config.task_type == "information_extraction":
            return self._process_ie_labels(blocks, padding, truncation)
        
        return None

    def _process_ie_labels(self, blocks: List[dict], padding: bool=True, truncation: bool=True):

        labels = []

        for block in blocks:
            previous = 0
            sub_labels = []
            for label in block.get("labels", []):
                tokens = self.tokenizer.encode(block["content"][previous:label["start"]])
                sub_labels.extend([0, ] * len(tokens))
                label_id = self.config.label2id.get(label.get("category"), 0)
                sub_labels.append(label_id)
                tokens = self.tokenizer.encode(block["content"][label["start"]:label["end"]])
                sub_labels.extend([label_id + 1, ] * (len(tokens) - 1))
                previous = label["end"]
            if previous < len(block["content"]):
                tokens = self.tokenizer.encode(block["content"][previous:])
                sub_labels.extend([0, ] * len(tokens))
            if truncation:
                sub_labels = sub_labels[:self.config.max_seq_length]
            if padding:
                sub_labels = sub_labels + [-100, ] * (self.config.max_seq_length - len(sub_labels))
            labels.append(sub_labels)

        if padding:
            row = [-100, ] * self.config.max_seq_length
            labels = labels + [row, ] * (self.config.max_num_nodes - len(labels))

        return labels

    def _post_process_for_doc_cls(self,
        probabilities: Optional[Union[np.array, torch.Tensor]]
    ):
        label_ids =  np.argmax(probabilities, axis=-1)
        scores = np.max(probabilities, axis=-1)
        predict_categories = [self.config.id2label[label_id] for label_id in label_ids]
        return {
            "predictions": predict_categories,
            "score": scores.tolist()
        }

    def _post_process_for_node_cls(self,
        probabilities: Optional[Union[np.array, torch.Tensor]],
        blocks: Optional[List[dict]]=None
    ):
        # [batch_size, max_num_nodes]
        label_ids = np.argmax(probabilities, axis=-1).tolist()
        scores = np.max(probabilities, axis=-1).tolist()
        copied_blocks = copy.deepcopy(blocks)
        for index in range(len(copied_blocks)):
            copied_blocks[index]["category"] = self.config.id2label[str(label_ids[index])]
            copied_blocks[index]["score"] = scores[index]
        return {
            "predict_blocks": json.dumps(copied_blocks, ensure_ascii=False)
        }

    def _find_labels(self, label_ids: List[int], scores: List[float], tokens: List[str], pattern: re.Pattern):
        labels = []
        sequence = " ".join([str(x) for x in label_ids])
        for match in pattern.finditer(sequence):
            str_start, str_end = match.span()
            if sequence[:str_start].strip():
                token_start = len(sequence[:str_start].strip().split(" "))
            else:
                token_start = 0
            token_end = token_start + len(sequence[str_start:str_end].strip().split(" "))
            start = sum([len(token) for token in tokens[:token_start]])
            end = start + sum([len(token) for token in tokens[token_start:token_end]])
            score = sum(scores[token_start:token_end]) / (token_end - token_start)
            labels.append({
                "category": self.config.id2label[str(label_ids[token_start])],
                "start": start,
                "end": end,
                "score": score
            })
        return labels
    
    def _is_newline(self, source: List[dict], target: List[dict]):

        source_x1, source_y1 = source["bndbox"][0]
        source_x2, source_y2 = source["bndbox"][2]
        source_width = source_x2 - source_x1
        source_height = source_y2 - source_y1
        source_char_len = source_width / len(source["content"])

        target_x1, target_y1 = target["bndbox"][0]
        target_x2, target_y2 = target["bndbox"][2]
        target_width = target_x2 - target_x1
        target_height = target_y2 - target_y1
        target_char_len = target_width / len(target["content"])

        if target_y1 > source_y2 or target_y2 < source_y1:
            y_gap = max(target_y1 - source_y2, source_y1 - target_y2)
            if y_gap > 1.5 * (target_height + source_height):
                return True
        else:
            x_gap = max(target_x1 - source_x2, source_x1 - target_x2)
            if x_gap > 2.5 * (target_char_len + source_char_len):
                return True
        return False
    
    def _sorted_blocks(self, blocks: List[dict]):

        if not blocks or len(blocks) <= 1:
            return blocks
        
        row_number = 0

        sorted_blocks = sorted(blocks, key=lambda x: x["bndbox"][0][1])

        y_list = [point[1] for point in sorted_blocks[0]["bndbox"]]
        row_end = max(y_list)

        rows = []
        for block in sorted_blocks:
            y_list = [point[1] for point in block["bndbox"]]
            if (min(y_list) <= row_end):
                rows.append((row_number, block))
                row_end = max(row_end, max(y_list))
            else:
                row_number += 1
                rows.append((row_number, block))
                row_end = max(y_list)
        
        sorted_rows = sorted(rows, key=lambda x: (x[0], x[1]["bndbox"][0][0]))

        return [row[1] for row in sorted_rows]

    def _concat_values(self, category, blocks: List[dict]):

        values = []
        scores = []
        is_newline = False
        previous = None

        for block in blocks:
            if previous is not None:
                is_newline = self._is_newline(previous, block)
            text = ""
            for label in block.get("labels"):
                if label["category"] != category:
                    continue
                text += block["content"][label["start"]:label["end"]]
            if not values or is_newline:
                values.append(text)
                scores.append([label["score"]])
            else:
                values[-1] = values[-1] + text
                scores[-1].append(label["score"])
            previous = block

        scores = [sum(score) / len(score) for score in scores]
        sorted_values = sorted(zip(values, scores), key=lambda x: x[1], reverse=True)
        return sorted_values[0]

    def _extract_kv(self, blocks: List[dict]):

        category2blocks = {}
        # collect blocks for each category
        for block in blocks:
            if not block.get("labels"):
                continue
            for label in block.get("labels"):
                if label["category"] not in category2blocks:
                    category2blocks[label["category"]] = []
                category2blocks[label["category"]].append(block)
        result = {}
        for category, values in category2blocks.items():
            value, score = self._concat_values(category, self._sorted_blocks(values))
            result[category] = {"text": value, "score": score}
        return result

    def _post_process_for_ie(self,
        predictions: Optional[Union[np.array, torch.Tensor]],
        probabilities: Optional[Union[np.array, torch.Tensor]],
        mask: Optional[Union[np.array, torch.Tensor]]=None,
        blocks: Optional[List[dict]]=None
    ):
        values = sorted(self.config.label2id.values(), reverse=True)
        expressions = [f"({value} |{value + 1} )*({value + 1}|{value})" for value in values]
        pattern = re.compile(f"({'|'.join(expressions)})(?!\\d)")
        # [max_num_nodes, max_seq_length]
        label_ids = (predictions * mask).tolist()
        scores = np.max(probabilities, axis=-1).tolist()
        copied_blocks = copy.deepcopy(blocks)
        for index in range(len(copied_blocks)):
            labels = self._find_labels(
                label_ids=label_ids[index], 
                scores=scores[index],
                tokens=self.tokenizer.tokenize(copied_blocks[index]["content"]), 
                pattern=pattern
            )
            copied_blocks[index]["labels"] = labels
        result = self._extract_kv(copied_blocks)
        return {
            "predict_blocks": json.dumps(copied_blocks, ensure_ascii=False),
            "predict_result": json.dumps(result, ensure_ascii=False)
        }

    def post_process(self,
        probabilities: Optional[Union[np.array, torch.Tensor]]=None,
        predictions: Optional[Union[np.array, torch.Tensor]]=None,
        mask: Optional[Union[np.array, torch.Tensor]]=None,
        blocks: Optional[list[dict]]=None
    ):
        if isinstance(probabilities, torch.Tensor):
            probabilities = probabilities.numpy()
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        if self.config.task_type == "document_classification":
            outputs = self._post_process_for_doc_cls(probabilities)
        if self.config.task_type == "node_classification":
            outputs = self._post_process_for_node_cls(probabilities, blocks)
        if self.config.task_type == "information_extraction":
            outputs = self._post_process_for_ie(predictions, probabilities, mask, blocks)

        return outputs

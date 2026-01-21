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
logger = logging.getLogger(__name__)


# -----------------------------------------------------------Main-----------------------------------------------------------
class LayoutGCNDocProcessor(object):


    tokenizer_class = "LayoutGCNTokenizer"
    image_processor_class = "EfficientNetImageProcessor"
    
    model_input_names = ["token_ids", "sequence_length", "pixel_values", "boxes", "num_nodes", "layout_features", "adj_angle", "adj_radical_dist", "labels"]

    def __init__(self, config: DocProcessorConfig):
        """See base class."""
        self.config = config
        # Image processor
        if self.config.use_image:
            self.image_processor = EfficientNetImageProcessor.from_pretrained(config.efficientnet_model_path)
        else:
            self.image_processor = None
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
        
        blocks = []
        for block in json.loads(example["blocks"]):
            # Skip empty block
            if block.get("content") is None or not block["content"].strip():
                continue
            # Skip low score block
            if block.get("score", 1.0) < 0.4:
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
            blocks.append(block)
        example["blocks"] = json.dumps(blocks, ensure_ascii=False)
        return example

    def _augment_blocks(self, blocks: list[dict]):

        copied_blocks = copy.deepcopy(blocks)

        if self.config.shuffle_prob > 0 and random.random() < self.config.shuffle_prob:
            random.shuffle(copied_blocks)

        if self.config.dropped_prob > 0 and random.random() < self.config.dropped_prob:
            copied_blocks = [block for block in copied_blocks if random.random() < self.config.dropped_blocks_prob]
        
        if not copied_blocks:
            copied_blocks = blocks

        if self.config.fake_size_prob > 0 and random.random() < self.config.fake_size_prob:
            height, width = self._calculate_doc_size(copied_blocks)
        else:
            height, width = None, None

        return copied_blocks, height, width

    def _calculate_doc_size(self, blocks: list[dict]):

        x_max = max([block["bndbox"][2][0] for block in blocks])
        y_max = max([block["bndbox"][2][1] for block in blocks])

        return 1.1 * y_max, 1.1 * x_max

    def _get_angle_adj(self, bndboxes: np.ndarray, delta: int = 10):

        # Calculate the center point
        center_y = bndboxes[:, [1, 3]].mean(axis=1)
        center_x = bndboxes[:, [0, 2]].mean(axis=1)
        center_yx = np.stack([center_y, center_x], axis=-1)

        bndboxes_i = bndboxes[:, None, :]
        bndboxes_j = bndboxes[None, :, :]

        i_left_top = np.maximum(bndboxes_i[..., :2], bndboxes_j[..., :2])
        i_right_bottom = np.minimum(bndboxes_i[..., 2:], bndboxes_j[..., 2:])
        intersection = np.maximum(i_right_bottom - i_left_top, 0)

        size_i = bndboxes_i[..., 2:] - bndboxes_i[..., :2]
        size_j = bndboxes_j[..., 2:] - bndboxes_j[..., :2]
        min_size = np.minimum(size_i, size_j)

        valid_intersection = (intersection > 0).all(axis=-1)
        overlap_ratio = np.divide(
            intersection,
            min_size,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=min_size > 0
        )
        overlap_flag = valid_intersection & (overlap_ratio > 0.5).all(axis=-1)

        delta_yx = center_yx[:, None, :] - center_yx[None, :, :]

        delta_yx[overlap_flag] = 0

        rad = np.arctan2(delta_yx[..., 0], delta_yx[..., 1])
        angle = np.rad2deg(rad)

        angle = np.where(angle < 0, angle + 360, angle)

        return (angle / delta).astype(np.int32)


    def __call__(self,
            blocks: Optional[list[dict]] = None,
            image: Optional[Image.Image] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            category: Optional[Union[str, list[str]]] = None,
            padding: Union[bool] = True,
            truncation: Union[bool] = True,
            is_training: Optional[bool] = False,
            return_tensors: Union[bool] = False
        ):
        outputs = {}
        # Check inputs
        if blocks is None:
            raise ValueError(f"You need to provide at least one input to call {self.__class__.__name__}")
        # Shuffle the blocks
        if is_training:
            blocks, height, width = self._augment_blocks(blocks)
        # Truncate the number of nodes
        if truncation and self.config.max_num_nodes is not None:
            blocks = blocks[:self.config.max_num_nodes]
        # Calculate document size if not provided
        if height is None or width is None:
            height, width = self._calculate_doc_size(blocks)
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
            padding_token_id_row = [self.tokenizer.pad_token_id, ] * self.config.max_seq_length
            token_ids += [padding_token_id_row, ] * (self.config.max_num_nodes - num_nodes)
            sequence_length += [0, ] * (self.config.max_num_nodes - num_nodes)
        outputs["token_ids"] = np.array(token_ids, dtype=np.int32)
        outputs["sequence_lengths"] = np.array(sequence_length, dtype=np.int32)
        if return_tensors:
            outputs["token_ids"] = torch.tensor([outputs["token_ids"]], dtype=torch.int32)
            outputs["sequence_lengths"] = torch.tensor([outputs["sequence_lengths"]], dtype=torch.int32)
        # Get the height and width of the document
        if height is None or width is None:
            if image is not None:
                width, height = image.size
            else:
                raise ValueError("If no document image is provided, height and width of document have to be specified.")
        # Get the layout features
        boxes = np.asarray([
            [
                block["bndbox"][0][0], block["bndbox"][0][1],
                block["bndbox"][2][0], block["bndbox"][2][1]
            ] for block in blocks
        ])
        # Padding boxes
        if padding and num_nodes < self.config.max_num_nodes:
            boxes = np.vstack([
                boxes,
                np.array([[0, 0, 0, 0], ] * (self.config.max_num_nodes - num_nodes))
            ])
        position = boxes / np.array([width, height, width, height])
        size = (boxes[..., 2:] - boxes[..., :2]) / np.array([width, height])
        shape = (boxes[..., 2] - boxes[..., 0]) / np.clip(boxes[..., 3] - boxes[..., 1], 1, None) / width
        layout_features = np.concatenate([position, size, np.expand_dims(shape, axis=-1)], axis=-1)
        mask = (np.arange(self.config.max_num_nodes) < num_nodes).reshape(-1, 1)
        masked_layout_features = layout_features * mask.astype(np.float32)
        outputs["layout_features"] = np.array(masked_layout_features, dtype=np.float32)
        if return_tensors:
            outputs["layout_features"] = torch.tensor([outputs["layout_features"]], dtype=torch.float32)
        # Get the adjacency angle matrix
        # [num_blocks, 1, 4]
        expand_boxes = np.expand_dims(boxes, axis=1)
        # [num_blocks, num_blocks, 4]
        tiled_boxes = np.tile(expand_boxes, [1, boxes.shape[0], 1])
        # [num_blocks, num_blocks, 4]
        trans_boxes = tiled_boxes.transpose(1, 0, 2)
        adj_angle = self._get_angle_adj(boxes, delta=self.config.angle_delta)
        outputs["adj_angle"] = torch.tensor([adj_angle], dtype=torch.int32) if return_tensors else adj_angle

        # Get the adjacency distance matrix
        dist_norm = math.sqrt(width ** 2 + height ** 2)
        left_top = tiled_boxes[..., :2] - trans_boxes[..., 2:]
        right_bottom = trans_boxes[..., :2] - tiled_boxes[..., 2:]
        delta_x = np.maximum(left_top[..., 0], right_bottom[..., 0])
        delta_y = np.maximum(left_top[..., 1], right_bottom[..., 1])
        distance = np.sqrt(np.square(np.clip(delta_x, 0, None)) + np.square(np.clip(delta_y, 0, None)))
        adj_mask = (mask.reshape(-1, 1) & mask.reshape(1, -1)).astype(np.float32)
        adj_radical_dist = np.exp(-1 * self.config.radical_alpha * distance / dist_norm) * adj_mask
        adj_radical_dist = np.array(adj_radical_dist, dtype=np.float32)
        outputs["adj_radical_dist"] = torch.tensor([adj_radical_dist], dtype=torch.float32) if return_tensors else adj_radical_dist

        # 
        if self.config.use_image and image is not None:
            width, height = image.size
            image_features = self.image_processor(image.convert("RGB"))
            w_resized_ratio = self.image_processor.size["width"] / width
            h_resized_ratio = self.image_processor.size["height"] / height
            outputs["pixel_values"] = torch.tensor([image_features["pixel_values"]], dtype=torch.float32) if return_tensors else image_features["pixel_values"]
            boxes = boxes / np.array([w_resized_ratio, h_resized_ratio, w_resized_ratio, h_resized_ratio])
            outputs["boxes"] = torch.tensor([boxes], dtype=torch.float32) if return_tensors else boxes

        if category is not None and self.config.task_type == "document_classification":
            if self.config.problem_type == "single_label_classification":
                outputs["labels"] = self.config.label2id.get(category)

            elif self.config.problem_type == "multi_label_classification":
                outputs["labels"] = [
                    self.tokenizer.label2id(label) for label in category
                ]

        elif self.config.task_type == "node_classification":
            outputs["labels"] = [self.config.label2id.get(block.get("category"), 0) for block in blocks]
            if padding:
                outputs["labels"] = outputs["labels"] + [-100, ] * (self.config.max_num_nodes - num_nodes)

        elif self.config.task_type == "information_extraction":
 
            outputs["labels"] = []
            
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
                outputs["labels"].append(sub_labels)

            if padding:
                row = [-100, ] * self.config.max_seq_length
                outputs["labels"] = outputs["labels"] + [row, ] * (self.config.max_num_nodes - len(outputs["labels"]))

        if outputs["labels"] is not None and return_tensors:
            outputs["labels"] = torch.tensor([outputs["labels"]], dtype=torch.int64)

        return outputs

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
        blocks: Optional[list[dict]]=None
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

    def _find_labels(self, label_ids: list[int], scores: list[float], tokens: list[str], pattern: re.Pattern):
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
    
    def _is_newline(self, source: list[dict], target: list[dict]):

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
    
    def _sorted_blocks(self, blocks: list[dict]):

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

    def _concat_values(self, category, blocks: list[dict]):

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

    def _extract_kv(self, blocks: list[dict]):

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
        blocks: Optional[list[dict]]=None
    ):
        values = sorted(self.config.label2id.values(), reverse=True)
        expressions = [f"({value} )?({value + 1} )*{value + 1}|{value}" for value in values]
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

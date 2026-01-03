#encoding=utf-8

# --------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025
# Auther: Dengliang Shi
# --------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------Libraries----------------------------------------------------------
# Standard library
import json

# Third-party libraries


# User define module


# ------------------------------------------------------Global Variables----------------------------------------------------


# -----------------------------------------------------------Main-----------------------------------------------------------
class Evaluator(object):

    def __init__(self, task_type):
        self.task_type = task_type

    def calculate_micro_f1(self, result):

        tp, fp, fn = 0, 0, 0
        for value in result.values():
            tp += value.get("tp", 0)
            fp += value.get("fp", 0)
            fn += value.get("fn", 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1
        }
    
    def calculate_macro_f1(self, result):

        tp, fp, fn = 0, 0, 0
        precision, recall, f1 = 0, 0, 0

        for values in result.values():
            tp += values.get("tp", 0)
            fp += values.get("fp", 0)
            fn += values.get("fn", 0)

            precision += values.get("precision", 0)
            recall += values.get("recall", 0)
            f1 += values.get("f1", 0)

        num_classes = len(result)
        return {
            "precision": precision / num_classes,
            "recall": recall / num_classes,
            "f1": f1 / num_classes,
            "tp": tp / num_classes,
            "fp": fp / num_classes,
            "fn": fn / num_classes,
        }

    def calculate_f1(self, eval_result):
        for value in eval_result.values():
            tp = value.get("tp", 0)
            fp = value.get("fp", 0)
            fn = value.get("fn", 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            value["precision"] = precision
            value["recall"] = recall
            value["f1"] = f1
        return eval_result

    def evaluate_for_doc_cls(self, samples):
        eval_result = {}
        for sample in samples:
            if sample["category"] == sample["predict_category"]:
                if sample["category"] not in eval_result:
                    eval_result[sample["category"]] = {}
                if "tp" not in eval_result[sample["category"]]:
                    result[sample["category"]]["tp"] = 0
                result[sample["category"]]["tp"] += 1
                sample["category_eval_result"] = 1
            else:
                if sample["category"] not in result:
                    result[sample["category"]] = {}
                if "fp" not in result[sample["category"]]:
                    result[sample["category"]]["fp"] = 0
                result[sample["category"]]["fp"] += 1
                if sample["predict_category"] not in result:
                    result[sample["predict_category"]] = {}
                if "fn" not in result[sample["predict_category"]]:
                    result[sample["predict_category"]]["fn"] = 0
                result[sample["predict_category"]]["fn"] += 1
                sample["category_eval_result"] = 0
        result = self.calculate_f1(result)
        return {
            "micro_f1": self.calculate_micro_f1(result),
            "macro_f1": self.caluculate_macro_f1(result),
            "details": result
        }

    def evaluate_for_ie(self, samples):

        global_eval_result = {}

        for sample in samples:

            kv_result = json.loads(sample["result"])
            predict_kv_result = json.loads(sample["predict_result"])

            eval_result = {}
            for key, value in predict_kv_result.items():
                if key not in eval_result:
                    eval_result[key] = {"tp": 0, "fp": 0, "fn": 0}
                if key not in global_eval_result:
                    global_eval_result[key] = {"tp": 0, "fp": 0, "fn": 0}
                if key not in kv_result or kv_result[key] != value["text"]:
                    eval_result[key]["fp"] += 1
                    global_eval_result[key]["fp"] += 1
                else:
                    eval_result[key]["tp"] += 1
                    global_eval_result[key]["tp"] += 1
            for key, value in kv_result.items():
                if key not in eval_result:
                    eval_result[key] = {"tp": 0, "fp": 0, "fn": 0}
                if key not in global_eval_result:
                    global_eval_result[key] = {"tp": 0, "fp": 0, "fn": 0}
                if key in predict_kv_result and predict_kv_result[key]["text"] == value:
                    continue
                eval_result[key]["fn"] += 1
                global_eval_result[key]["fn"] += 1

            eval_result = self.calculate_f1(eval_result)
            kv_eval_result = {
                "micro_f1": self.calculate_micro_f1(eval_result),
                "macro_f1": self.calculate_macro_f1(eval_result),
                "details": eval_result
            }
            sample["kv_eval_result"] = json.dumps(kv_eval_result, ensure_ascii=False)

        global_eval_result = self.calculate_f1(global_eval_result)

        return {
            "micro_f1": self.calculate_micro_f1(global_eval_result),
            "macro_f1": self.calculate_macro_f1(global_eval_result),
            "details": global_eval_result
        }
    
    def evaluate_for_node_cls(self, samples):
        
        global_eval_result = {}

        for sample in samples:

            blocks = json.loads(sample["blocks"])
            predict_blocks = json.loads(sample["predict_result"])

            eval_result = {}
            for index in range(len(predict_blocks)):
                predict_category = predict_blocks[index].get("category")
                if not predict_category:
                    continue
                if predict_category not in eval_result:
                    eval_result[predict_category] = {"tp": 0, "fp": 0, "fn": 0}
                if predict_category not in global_eval_result:
                    global_eval_result[predict_category] = {"tp": 0, "fp": 0, "fn": 0}
                category = blocks[index].get("category")
                if predict_category != category:
                    eval_result[predict_category]["fp"] += 1
                    global_eval_result[predict_category]["fp"] += 1
                else:
                    eval_result[predict_category]["tp"] += 1
                    global_eval_result[predict_category]["tp"] += 1
                if not category:
                    continue
                if category not in eval_result:
                    eval_result[category] = {"tp": 0, "fp": 0, "fn": 0}
                if category not in global_eval_result:
                    global_eval_result[category] = {"tp": 0, "fp": 0, "fn": 0}
                if predict_category != category:
                    eval_result[category]["fn"] += 1
                    global_eval_result[category]["fn"] += 1
            eval_result = self.calculate_f1(eval_result)
            kv_eval_result = {
                "micro_f1": self.calculate_micro_f1(eval_result),
                "macro_f1": self.calculate_macro_f1(eval_result),
                "details": eval_result
            }
            sample["kv_eval_result"] = json.dumps(kv_eval_result, ensure_ascii=False)

        global_eval_result = self.calculate_f1(global_eval_result)

        return {
            "micro_f1": self.calculate_micro_f1(global_eval_result),
            "macro_f1": self.calculate_macro_f1(global_eval_result),
            "details": global_eval_result
        }

    def evaluate(self, samples):

        if self.task_type == "document_classification":
            return self.evaluate_for_doc_cls(samples)
        elif self.task_type == "information_extraction":
            return self.evaluate_for_ie(samples)
        elif self.task_type == "node_classification":
            return self.evaluate_for_node_cls(samples)

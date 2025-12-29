

def is_newline(source: list[dict], target: list[dict]):

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

    if source_height / target_height > 2 or target_height / source_height > 2:
        return True

    if target_y1 > source_y2 or target_y2 < source_y1:
        y_gap = max(target_y1 - source_y2, source_y1 - target_y2)
        if y_gap > (target_height + source_height):
            return True
    else:
        x_gap = max(target_x1 - source_x2, source_x1 - target_x2)
        if x_gap > 2 * (target_char_len + source_char_len):
            return True
    return False

def concat_values(category, blocks: list[dict]):

    values = []
    is_newline = False
    previous = None

    for block in blocks:
        if previous is not None:
            is_newline = is_newline(previous, block)
        for label in block.get("labels"):
            if label["category"] != category:
                continue
            text = block["content"][label["start"]:label["end"]]
            if not values or is_newline:
                values.append(text)
            else:
                values[-1] = values[-1] + text
        previous = block

    return values[0]

def extract_kv(self, blocks: list[dict]):

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
        value = concat_values(category, values)
        result[category] = value
    return result

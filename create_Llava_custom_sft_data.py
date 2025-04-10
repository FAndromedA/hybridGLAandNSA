from datasets import Dataset, load_dataset, concatenate_datasets, Features, Value, Sequence, Image
import pandas as pd

import json


ds_en = load_dataset("BUAADreamer/llava-en-zh-300k", "en")['train']
ds_zh = load_dataset("BUAADreamer/llava-en-zh-300k", "zh")['train']
ds_pure_text = load_dataset("JunxiongWang/sftdataset")['train']
ds_en = ds_en.shuffle(seed=42)
ds_zh = ds_zh.shuffle(seed=42)
ds_pure_text = ds_pure_text.shuffle(seed=42)

print(ds_zh[0])
print(len(ds_zh))
print(ds_en[0])
print(len(ds_en))
print(ds_pure_text[1])
print(len(ds_pure_text))

# 保留一半的 en 数据（前 1/2）
len_en = len(ds_en)
subset_en = ds_en # 算了，直接保留所有的数据
# subset_en = ds_en.select(range(len_en // 2))

# 保留六分之一的 zh 数据（前 1/6）
len_zh = len(ds_zh)
subset_zh = ds_zh # 算了，直接保留所有的数据
# subset_zh = ds_zh.select(range(len_zh // 6))

# 混合数据（可随机打乱，也可直接拼接）
mixed_en_zh_dataset = concatenate_datasets([subset_en, subset_zh]).shuffle(seed=42)

def add_empty_image_column(example):
    example["images"] = None  # 或使用 np.zeros() 占位张量
    return example

subset_pure_text = ds_pure_text.select(range(len(mixed_en_zh_dataset) * 3 // 7)) # 将纯文本数据和多模态数据按照 3:7 的比例混合

# pyarrow.lib.ArrowTypeError: struct fields don't match or are in the wrong order: 
# Input fields: struct<content: string, role: string> output fields: struct<role: string, content: string>

# target_message_features = Features({
#     "messages": Sequence({
#         "role": Value("string"),
#         "content": Value("string"),
#     }),
#     "images": Sequence(Value("string")),
# })

data_ls = []
for item in subset_pure_text:
    messages = [{"role": m["role"], "content": m["content"]} for m in item["messages"]]
    data_ls.append({
        "messages": messages, 
        "images": None
    })

print(data_ls[1])
print(mixed_en_zh_dataset.features)

target_message_features = Features({'messages': [{'role': Value(dtype='string', id=None), 'content': Value(dtype='string', id=None)}], 
                                    'images': Sequence(feature=Image(mode=None, decode=True, id=None), length=-1, id=None)})

subset_pure_text_aligned = Dataset.from_list(data_ls, features=target_message_features)

print(subset_pure_text_aligned[1])
print(subset_pure_text_aligned.features)

mixed_dataset = concatenate_datasets([mixed_en_zh_dataset, subset_pure_text_aligned]).shuffle(seed=42)
# 这里的 mixed_dataset 是一个包含多模态数据和纯文本数据的混合数据集
# 保存到本地（保存为一个新的 dataset 目录）
mixed_dataset.save_to_disk("llava-mixed-dataset2")
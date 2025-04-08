from datasets import load_dataset, concatenate_datasets

ds_en = load_dataset("BUAADreamer/llava-en-zh-300k", "en")['train']
ds_zh = load_dataset("BUAADreamer/llava-en-zh-300k", "zh")['train']
ds_en = ds_en.shuffle(seed=42)
ds_zh = ds_zh.shuffle(seed=42)

print(ds_zh[0])
# 保留一半的 en 数据（前 1/2）
len_en = len(ds_en)
subset_en = ds_en.select(range(len_en // 2))

# 保留六分之一的 zh 数据（前 1/6）
len_zh = len(ds_zh)
subset_zh = ds_zh.select(range(len_zh // 6))

# 混合数据（可随机打乱，也可直接拼接）
mixed_dataset = concatenate_datasets([subset_en, subset_zh]).shuffle(seed=42)

# 保存到本地（保存为一个新的 dataset 目录）
mixed_dataset.save_to_disk("llava-mixed-dataset")
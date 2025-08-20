import copy

import yaml
from torch.utils.data import DataLoader
from wenet.dataset.dataset import Dataset
from wenet.utils.init_tokenizer import init_tokenizer
import torch
from common_utils.utils4infer import do_format_shard_manifest4one, get_test_conf



config_path = './conf/ct_config.yaml'
test_data_path="./common_utils/fake_data/raw/data.list"
data_type="raw"  # 数据类型，raw或shards_full_data
if data_type == "shard":
    test_data_path = do_format_shard_manifest4one(test_data_path)
configs, test_conf = get_test_conf(config_path)
tokenizer = init_tokenizer(configs)
test_dataset = Dataset(data_type,
                       test_data_path,
                       tokenizer,
                       test_conf,
                       partition=False)

test_data_loader = DataLoader(test_dataset,
                              batch_size=None,
                              num_workers=1)

print('测试 raw 数据集')
with torch.no_grad():
    for batch_idx, batch in enumerate(test_data_loader):
        keys = batch["keys"]
        feats = batch["feats"].to(torch.bfloat16)
        feats_lengths = batch["feats_lengths"]
        txts = batch["txts"]
        batch_size = feats.size(0)
        print(f"batch_idx: {batch_idx}, batch_size: {batch_size} feats_size: {feats.size()} feats_lengths_size: {feats_lengths}, txt: {txts}")



test_data_path = "/home/A02_tmpdata3/code/OSUM_tmp/OSUM-EChat/common_utils/fake_data/shard/shards_list.txt"
data_type="shard"  # 数据类型，raw或shard
if data_type == "shard":
    test_data_path = do_format_shard_manifest4one(test_data_path)
configs, test_conf = get_test_conf(config_path)
tokenizer = init_tokenizer(configs)
test_dataset = Dataset(data_type,
                       test_data_path,
                       tokenizer,
                       test_conf,
                       partition=False)

test_data_loader = DataLoader(test_dataset,
                              batch_size=None,
                              num_workers=1)
print('测试 shard 数据集')
with torch.no_grad():
    for batch_idx, batch in enumerate(test_data_loader):
        keys = batch["keys"]
        feats = batch["feats"].to(torch.bfloat16)
        feats_lengths = batch["feats_lengths"]
        txts = batch["txts"]
        batch_size = feats.size(0)
        print(f"batch_idx: {batch_idx}, batch_size: {batch_size} feats_size: {feats.size()} feats_lengths_size: {feats_lengths}, txt: {txts}")


test_data_path = "common_utils/fake_data/combine/combines_list.txt"
data_type="shard"  # 数据类型，raw或shard
if data_type == "shard":
    test_data_path = do_format_shard_manifest4one(test_data_path)
configs, test_conf = get_test_conf(config_path)
tokenizer = init_tokenizer(configs)
test_dataset = Dataset(data_type,
                       test_data_path,
                       tokenizer,
                       test_conf,
                       partition=False)

test_data_loader = DataLoader(test_dataset,
                              batch_size=None,
                              num_workers=1)
print('测试 combine 数据集')
with torch.no_grad():
    for batch_idx, batch in enumerate(test_data_loader):
        keys = batch["keys"]
        feats = batch["feats"].to(torch.bfloat16)
        feats_lengths = batch["feats_lengths"]
        txts = batch["txts"]
        batch_size = feats.size(0)
        print(f"batch_idx: {batch_idx}, batch_size: {batch_size} feats_size: {feats.size()} feats_lengths_size: {feats_lengths}, txt: {txts}")

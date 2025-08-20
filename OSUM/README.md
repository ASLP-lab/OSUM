 <p align="left">
        <a href="README_CN.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>

# How to Use the OSUM Model Framework for Training and Inference

## Prepare for Environment

Before you start, please make sure your Python environment is ready. The following is a recommended operation process. We assume that you have already installed the Conda software on your computer. If not, please refer to: [One-click Installation of Miniconda on Linux](https://blog.csdn.net/qq_41636123/article/details/130266232). We highly recommend that you run our code on a computer with a Linux system.

```shell
# Create a new Conda environment
conda create -n OSUM python=3.10
# Activate the newly created environment
conda activate OSUM
# Download our code and install the required Python packages
git clone https://github.com/ASLP-lab/OSUM.git
cd OSUM
# If you are training on a GPU, please first remove the entry of torch_npu in requirements.txt. If you are using an NPU, no action is required.
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Inference

First, let's see how to perform inference.

The main file used is: OSUM/examples/osum/infer.sh

### First, download the checkpoint:

Checkpoint. Download the model checkpoint from our Hugging Face repository. You can download it using Python:

```python
# Download the .pt file from Hugging Face
from huggingface_hub import hf_hub_download
pt_file_path = hf_hub_download(repo_id="ASLP-lab/OSUM", filename="infer.pt")  # At this time, pt_file_path is directly the specific path of the downloaded checkpoint.
```

Or download it from the Hugging Face website: https://huggingface.co/ASLP-lab/OSUM

Then set the `ckpt` variable in `infer.sh`:

```shell
ckpt_path=***/infer.sh
```

### Next, prepare the data

We support inference on two types of data (following the specifications of the [wenet](https://github.com/wenet-e2e/wenet) open-source framework):

- Raw format: You need to pass in a JSONL format file, where each line is in the format of a JSON dictionary. The dictionary key values include "key": the unique identifier of the audio, "wav": the specific path of the audio (not limited to the WAV format), and "txt": the text corresponding to the audio. In actual inference scenarios, it can be any value, but please try to ensure the existence of this key value to avoid possible errors at the code level.

  Specific example:

  ```json
  {"key": "BAC009S0764W0122", "wav": "***/wav/BAC009S0764W0122.wav", "txt": "First- and second-tier cities are also undergoing adjustments"}
  ```

- Shard_full_data format: It is more commonly used for training to accelerate the speed of the machine reading files and can also be used for inference. It saves several audio files (for example, 1000 files) in a tar package, and the content of the "key" exists as the file name. Taking the above audio entry as an example, in its tar package, it should be converted into the following two files: BAC009S0764W0122.txt and BAC009S0764W0122.wav. The file suffix is its corresponding variable value, and the file name is the unique identifier of the audio.

When you have prepared the data list file, set the data file variable in `infer.sh`:

```shell
data_path=***/data.jsonl
```

### Select the appropriate task

The current open-source version of OSUM supports multiple tasks. You can select the task you want to infer according to the task tags in the `OSUM/examples/osum/conf/prompt_config.yaml` file and fill in the tag of the task you want to infer in `OSUM/examples/osum/infer.sh`. At the same time, please note that the tag in `prompt_config.yaml` contains spaces, and the tag written in `infer.sh` needs to manually remove the spaces (due to the limitations of shell syntax).

Specific example:

```shell
task="<TRANSCRIBE><GENDER>"  # This tag is used for the ASR + gender task
```

### Select the appropriate GPU/NPU

The OSUM model is trained on Huawei Ascend 910B, but the training and inference codes support both GPU and NPU. If training on a GPU, please set `train_backend` to `nccl` instead of `hccl` in the startup script.

The inference of this project roughly requires 20G of video memory. If you have such a graphics card on your machine, set the serial number of the graphics card on which you want to perform inference:

```shell
gpu_id=3
```

### Start inference!

The following is a complete example of `infer.sh`:

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
lang=zh
prompt_file=conf/prompt_stage4.yaml
ckpt_path=./infer.sh
data_path=./data/aishell/data.list
data_type="raw"
gpu_id=3
output_dir=./output/aishell
task="<TRANSCRIBE><GENDER>"
bash decode/do_docode.sh --output_dir $output_dir --task $task --data_type $data_type --data_path $data_path --gpu_id $gpu_id --ckpt_path $ckpt_path --lang $lang --prompt_file $prompt_file
```


### How to Perform Inference More Directly?
- For inference on a single audio file
    ```shell
    # Set the checkpoint_path variable in this file to the locally downloaded ckpt, and then set the corresponding audio path and prompt in the main function
    python OSUM/examples/osum/runtime/infer_runtime.py
    ```

- Deploy and perform web-based inference
    ```shell
    # Deploy and start the service, which will deploy a test webpage on localhost 127.0.0.1:7860. You also need to set the checkpoint_path variable first.
    python OSUM/examples/osum/runtime/infer_gradio.py
    ```

## Training

Next, let's see how to perform training.

The main files involved are: OSUM/examples/osum/run_huawei_2p_master.sh, OSUM/examples/osum/run_huawei_2p_rank1.sh, and OSUM/examples/osum/run_huawei_2p_rank2.sh

This project supports multi-machine training by default. If the current code is not modified, it will perform 3-machine 8-card training by default.

### Checkpoint download

Same as the introduction in the Inference section.

### Data preparation

We strongly recommend using the shard_full_data format to maximize the reduction of the time consumption caused by loading data. The number of audio files contained in a single tar file is recommended to be 1000.

After the data is prepared, you need to write the data list to the `OSUM/examples/osum/conf/data_config_huawei.yaml` file. The specific content of `data_config_huawei.yaml` is as follows:

```yaml
data_name1:
  path: "***/shards_list.txt"
  weight: 3  # Weight, triple the amount of data in this data list file
data_name2:
  path: "***/shards_list.txt"
  weight: 1
```

Specific example of `shards_list.txt`:

```txt
***/***/0001.tar
***/***/0002.tar
***/***/0003.tar
***/***/0004.tar
......
```

### Multi-machine multi-card training

Multiple machines need multiple startup files, and one machine needs to be set as the coordinator machine. For the current code, we execute the file `OSUM/examples/osum/run_huawei_2p_master.sh` on the coordinator machine and execute `OSUM/examples/osum/run_huawei_2p_rank1.sh` and `OSUM/examples/osum/run_huawei_2p_rank2.sh` on other machines respectively.

We should ensure that the contents of these three files are almost identical, except for the `node_rank` here:

```shell
torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --node_rank=0 \
         --master_addr=$HOST_NODE_ADDR --master_port=$HOST_PORT \
```

In `run_huawei_2p_master.sh`, you need to set `node_rank` to 0, and on other machines, it should be 1, 2, 3, 4, ... in sequence.

For other configurations, there are:

- Number of machines

  ```
  num_nodes=3
  ```

- IP address of the coordinator machine

  ```
  HOST_NODE_ADDR=192.168.0.15
  ```

- Other variables

  ```
  checkpoint=*** # If it is empty, it will be randomly initialized
  data=***  # Directory where the data is stored
  dir=***   # Location where the new checkpoint will be stored
  ```

### Single-machine training

You only need to set the number of machines in `OSUM/examples/osum/run_huawei_2p_master.sh` to 1 and set the coordinator IP to the local machine IP.

During single-machine training, if you want to modify the number of training machines, you only need to flexibly modify:

```shell
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
```

### Training hyperparameters

The specific parameters are set in the following file:

OSUM/examples/osum/conf/config_llm_huawei_base-version.yaml
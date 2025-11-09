#!/bin/bash

. ./path.sh || exit 1;

if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
export HCCL_CONNECT_TIMEOUT=1200
export CPU_AFFINITY_CONF=1 # 绑核
export TASK_QUEUE_ENABLE=2 # 优化下发队列
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(1)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi

echo "Parsed device_ids: ${device_ids[@]}"

stage=0
stop_stage=0

HOST_NODE_ADDR=***  # IP address of the machine where the master node is located, such as 192.168.0.1
HOST_PORT=29401

num_nodes=3
job_id=2023

train_config=conf/config_llm_huawei_base-version.yaml
gxl_data_json_info_path=conf/data_config_huawei.yaml
checkpoint=*** # 为空则为随机初始化
data=***
dir=***
mkdir -p $dir
mkdir -p $data



data_type=shard_full_data
train_data=$data/tmp/tmp_master.list
python common_utils/handle_data_for_weight.py $gxl_data_json_info_path $train_data
cv_data=$data/asr_cv.list
#head -n 1 $train_data > $cv_data

train_engine=deepspeed # torch_ddp


tensorboard_dir=$dir/tensorboard
num_workers=6
prefetch=10
average_checkpoint=false
decode_checkpoint=$dir/final.pt
average_num=5
average_mode=step
max_step=88888888
decode_modes="attention"
decoding_chunk_size=-1
ctc_weight=0.5
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
decode_batch=10
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model+optimizer"


. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  dist_backend="hccl"  #  attention!!!  hccl is used for npu and nccl is used for gpu
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "$0: PYTORCH_CUDA_ALLOC_CONF is $PYTORCH_CUDA_ALLOC_CONF"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --node_rank=0 \
          --master_addr=$HOST_NODE_ADDR --master_port=$HOST_PORT \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data $train_data \
      --cv_data $cv_data \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --timeout 1200 \
      --use_amp \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states} \

fi



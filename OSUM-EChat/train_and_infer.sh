#!/bin/bash


# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
export HCCL_CONNECT_TIMEOUT=1200
# export ASCEND_LAUNCH_BLOCKING=1
export CPU_AFFINITY_CONF=1 # 绑核
export TASK_QUEUE_ENABLE=2 # 优化下发队列
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
#export CUDA_VISIBLE_DEVICES="2"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"
export PYTHONPATH=./

cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-""}
if [ -z "$cuda_visible_devices" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Using default device_ids."
  device_ids=(1)
else
  IFS=',' read -r -a device_ids <<< "$cuda_visible_devices"
  echo "Using CUDA_VISIBLE_DEVICES: $cuda_visible_devices"
fi
# shellcheck disable=SC2145
echo "Parsed device_ids: ${device_ids[@]}"

stage=0
stop_stage=0


HOST_NODE_ADDR=127.0.0.1
HOST_PORT=29401
num_nodes=1
job_id=2023

train_config=conf/ct_config.yaml
gxl_data_json_info_path_s2t=conf/data_s2t.yaml
gxl_data_json_info_path_t2s=conf/data_t2s.yaml
gxl_data_json_info_path_s2s=conf/data_s2s.yaml
gxl_data_json_info_path_t2t=conf/data_t2t.yaml


dir=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch1_s2t_t2s_t2t_s2s_hq_language_think
checkpoint=/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/epoch0_s2t_t2s_t2t_s2s_language_think/step_22499.pt



mkdir -p $dir
data=$dir/data
mkdir -p $data


data_type=shard
train_data_s2t=$data/tmp/tmp_master_s2t.list
train_data_t2s=$data/tmp/tmp_master_t2s.list
train_data_s2s=$data/tmp/tmp_master_s2s.list
train_data_t2t=$data/tmp/tmp_master_t2t.list
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_s2t $train_data_s2t # 只能在master执行，因为随机数是time的，如果每个节点都执行，会导致不同节点的随机数不一致
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_t2s $train_data_t2s
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_s2s $train_data_s2s
python common_utils/load_combine_type_yaml.py $gxl_data_json_info_path_t2t $train_data_t2t


cv_data=$data/asr_cv.list
head -n 1 $train_data_s2t > $cv_data
wc -l  "$train_data_s2t"
wc -l "$train_data_t2s"
wc -l "$train_data_s2s"
wc -l "$train_data_t2t"




train_engine=deepspeed # torch_ddp


tensorboard_dir=$dir/tensorboard
num_workers=1
prefetch=50
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model+optimizer"


. tools/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk  -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  # NOTE(xcsong): deepspeed fails with gloo, see
  #   https://github.com/microsoft/DeepSpeed/issues/2818
  dist_backend="nccl" #"nccl"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  echo "$0: PYTORCH_CUDA_ALLOC_CONF is $PYTORCH_CUDA_ALLOC_CONF"
  # torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --node_rank=1 \
  #          --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus --node_rank=0 \
          --master_addr=$HOST_NODE_ADDR --master_port=$HOST_PORT \
    wenet/bin/train.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type  $data_type \
      --train_data_s2t $train_data_s2t \
      --train_data_t2s $train_data_t2s \
      --train_data_s2s $train_data_s2s \
      --train_data_t2t $train_data_t2t \
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


      # --load_dir $dir \
      # --ckpt_id 'epoch_1' \ # 直接加载deepspeed目录
fi


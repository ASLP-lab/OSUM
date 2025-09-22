# 通用推理脚本
# cd ..
. ./path.sh || exit 1;

data_type= # raw  shard_full_data
data_path=
ref_path=
gpu_id=
ckpt_path=
task=
lang=  # en zh
output_dir=

train_config=conf/config_llm_huawei_base-version.yaml
decode_modes="llmasr_decode"
decoding_chunk_size=-1
ctc_weight=0
reverse_weight=0.0
blank_penalty=0.0
length_penalty=0.0
batch_size=1
prompt_file=conf/prompt_config.yaml

. tools/parse_options.sh || exit 1;
# 将所有传进来的元素打印出来
echo "传入的参数为:"
echo 'data_type is '$data_type
echo 'gpu_id is '$gpu_id
echo "ckpt_path is $ckpt_path"
echo 'task is '$task
echo 'lang is '$lang
echo 'print over'
echo 'prompt_file'$prompt_file
echo 'output_dir is '$output_dir
mkdir -p $output_dir



decode_checkpoint=$ckpt_path
export CUDA_VISIBLE_DEVICES="$gpu_id"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"

test_data=$data_path
echo "test_data is $test_data"

echo "lang is $lang"
base=$(basename $decode_checkpoint)
result_dir=$output_dir
echo "result_dir is $result_dir"
mkdir -p $result_dir
python wenet/bin/recognize4llmasr.py --gpu ${gpu_id} \
  --modes $decode_modes \
  --config $train_config \
  --data_type $data_type \
  --test_data $test_data \
  --checkpoint $decode_checkpoint \
  --beam_size 10 \
  --batch_size $batch_size \
  --blank_penalty ${blank_penalty} \
  --length_penalty ${length_penalty} \
  --ctc_weight $ctc_weight \
  --reverse_weight $reverse_weight \
  --result_dir $result_dir \
  --task $task \
  --lang $lang \
  ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}

mkdir -p ${result_dir}
for mode in ${decode_modes}; do
  python tools/compute-wer.py --char=1 --v=1 \
    $ref_path $result_dir/$mode/text > $result_dir/$mode/wer
done
# done


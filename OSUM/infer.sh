
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
lang=zh  # en zh
prompt_file=conf/prompt_stage4.yaml

ckpt_path=***  

data_path=***
ref_path=***  # 参考音频路径，用于计算wer, 其格式为scp文件的格式，每一行形如“key 真实答案”,其可以用空格或者tab分隔
# raw: jsonl格式，每行一个json字符串，至少包含key wav  txt
# shard_full_data: 每行一个shard文件路径
data_type="raw" #  raw or shard_full_data
gpu_id=3  
output_dir=***  # 结果的输出路径 

task="<TRANSCRIBE><GENDER>" # 不含空格，任务切换，更多任务参考conf/prompt_config.yaml
bash decode/do_docode.sh --ref_path $ref_path --output_dir $output_dir --task $task --data_type $data_type --data_path $data_path --gpu_id $gpu_id --ckpt_path $ckpt_path --lang $lang --prompt_file $prompt_file
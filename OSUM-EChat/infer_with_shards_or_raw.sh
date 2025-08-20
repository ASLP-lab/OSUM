#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 设置可见GPU
# 配置参数
test_data_path="/home/A02_tmpdata2/data/context_asr_sentence_few/wav_shards/shards/test_raw_data/data.list"
infer_res_path="/home/A02_tmpdata2/ckpt/osum_chat_new_start_0810/context_asr/epoch0/infer_think/infer_res.scp"
gpu_ids=(0 1 2 3 4 5 6 7)  # GPU ID列表（可根据实际可用GPU修改）
per_gpu_processes=1  # 每个GPU上的进程数
data_type="raw"  # 数据类型，raw或shards_full_data


true_text_path="/home/A02_tmpdata2/data/context_asr_sentence_few/wav_shards/shards/test_text.scp"
output_dir=$(dirname "$infer_res_path")


# 计算总进程数
total_gpus=${#gpu_ids[@]}
total_processes=$(( total_gpus * per_gpu_processes ))

# 创建数据分割目录
split_dir=$(dirname "$infer_res_path")/split_data
mkdir -p "$split_dir"
echo "数据分割目录: $split_dir"

# 计算输入文件总行数和每个分割文件的行数
total_lines=$(wc -l < "$test_data_path")
each_line=$(( (total_lines + total_processes - 1) / total_processes ))  # 向上取整，确保分配均匀
echo "总数据行数: $total_lines, 总进程数: $total_processes, 每个进程处理行数: $each_line"

# 分割输入文件为total_processes个部分
for ((i=0; i<total_processes; i++)); do
    start=$((i * each_line + 1))
    end=$(( (i + 1) * each_line ))
    # 最后一个文件处理剩余行数
    if (( end > total_lines )); then
        end=$total_lines
    fi
    # 提取对应行到分割文件
    sed -n "${start},${end}p" "$test_data_path" > "$split_dir/tmpdata_${i}.list"
    echo "生成分割文件: $split_dir/tmpdata_${i}.list (行数: $start-$end)"
done

# 启动多进程并行推理
echo "开始启动 $total_processes 个进程..."
for ((i=0; i<total_processes; i++)); do
    # 分配GPU（每个GPU分配per_gpu_processes个进程）
    gpu_index=$((i / per_gpu_processes))
    current_gpu=${gpu_ids[$gpu_index]}
    
    # 当前进程的输入和输出路径
    split_input="$split_dir/tmpdata_${i}.list"
    split_output="$split_dir/infer_res_${i}.scp"
    
    # 后台启动进程
    echo "启动进程 $i (GPU: $current_gpu)，输入: $split_input，输出: $split_output"
    python infer_with_shards_or_raw.py \
        --test_data_path "$split_input" \
        --infer_res_path "$split_output" \
        --data_type "$data_type" \
        --gpu_id "$current_gpu" &
done

# 等待所有后台进程完成
wait
echo "所有进程执行完毕，开始合并结果..."

# 合并所有子结果到总输出文件（按进程ID顺序合并，保持与输入顺序一致）
# shellcheck disable=SC2188
> "$infer_res_path"  # 清空总输出文件（避免追加历史内容）
for ((i=0; i<total_processes; i++)); do
    split_output="$split_dir/infer_res_${i}.scp"
    if [ -f "$split_output" ]; then
        cat "$split_output" >> "$infer_res_path"
        echo "已合并: $split_output"
    else
        echo "警告: 子结果文件不存在，跳过合并: $split_output"
    fi
done

echo "所有操作完成，总结果文件: $infer_res_path"
python common_utils/do_compute_wer.py $true_text_path $infer_res_path $output_dir

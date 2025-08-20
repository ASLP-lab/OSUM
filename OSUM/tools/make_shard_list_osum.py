import argparse
import sys

sys.path.insert(0, '../../../')
import io
import logging
import os
import tarfile
import time
import multiprocessing
import json

import torch
import torchaudio
from tqdm import tqdm  # 需要安装 tqdm 库

AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])


def _write_tar_file(data_list, tar_file, resample=16000, index=0, total=1):
    logging.info('Processing {} {}/{}'.format(tar_file, index, total))
    read_time = 0.0
    save_time = 0.0
    write_time = 0.0
    if os.path.exists(tar_file+".finished"):
        return

    with tarfile.open(tar_file, "w") as tar:
        # 使用 tqdm 显示进度条
        for i, item in enumerate(tqdm(data_list, desc=f'Creating {os.path.basename(tar_file)}', unit='file')):
            task = item["task"]
            key = item["key"]
            txt = item["txt"]
            wav = item["wav"]
            lang = item["lang"]
            speaker = item["speaker"]
            emotion = item["emotion"]
            gender = item["gender"]
            extra = item["extra"]

            # logging.info(f"Processing audio file: {wav}")

            suffix = wav.split('.')[-1]
            if suffix not in AUDIO_FORMAT_SETS:
                logging.warning(f"File format {suffix} not supported. Skipping file {wav}")
                continue

            try:
                # Process audio
                ts = time.time()
                audio, sample_rate = torchaudio.load(wav)
                audio = torchaudio.transforms.Resample(sample_rate, resample)(audio)
                read_time += (time.time() - ts)

                audio = (audio * (1 << 15)).to(torch.int16)
                ts = time.time()
                with io.BytesIO() as f:
                    torchaudio.save(f, audio, resample, format="wav", bits_per_sample=16)
                    suffix = "wav"
                    f.seek(0)
                    data = f.read()
                save_time += (time.time() - ts)

                ts = time.time()
                # Save text file
                txt_file = key + '.txt'
                txt = txt.encode('utf8')
                txt_data = io.BytesIO(txt)
                txt_info = tarfile.TarInfo(txt_file)
                txt_info.size = len(txt)
                tar.addfile(txt_info, txt_data)

                # Save wav file
                wav_file = key + '.' + suffix
                wav_data = io.BytesIO(data)
                wav_info = tarfile.TarInfo(wav_file)
                wav_info.size = len(data)
                tar.addfile(wav_info, wav_data)

                # Save metadata fields (task, lang, speaker, emotion, gender) each in separate files
                for field, value in {"task": task, "lang": lang, "speaker": speaker, "emotion": emotion, "gender": gender}.items():
                    field_file = f"{key}.{field}"  # 文件名格式修改
                    field_data = io.BytesIO(str(value).encode('utf8'))
                    field_info = tarfile.TarInfo(field_file)
                    field_info.size = len(str(value))
                    tar.addfile(field_info, field_data)
                    # logging.info(f"Added file {field_file} with content: {value}")

                # Extract duration from extra and save it to a separate file
                duration = extra.get("duration", 0)
                duration_file = key + '.duration'
                duration_data = io.BytesIO(str(duration).encode('utf8'))
                duration_info = tarfile.TarInfo(duration_file)
                duration_info.size = len(str(duration))
                tar.addfile(duration_info, duration_data)

                # Save remaining extra data (excluding duration) to a separate key.extra file
                remaining_extra = {k: v for k, v in extra.items() if k != "duration"}
                jsonl_line = json.dumps(remaining_extra, ensure_ascii=False)
                jsonl_data = jsonl_line.encode('utf8')
                jsonl_file = key + '.extra'  # 文件名格式修改
                jsonl_data_io = io.BytesIO(jsonl_data)
                jsonl_info = tarfile.TarInfo(jsonl_file)
                jsonl_info.size = len(jsonl_data)
                tar.addfile(jsonl_info, jsonl_data_io)

                write_time += (time.time() - ts)
            except Exception as e:
                logging.error(f"Error processing file {wav}: {e}")

        logging.info(f'read {read_time:.2f}s save {save_time:.2f}s write {write_time:.2f}s')
        with open(tar_file+".finished", 'rb'):
            pass

def make_shards_common(jsonl_file, shards_dir, num_utts_per_shard=1000, prefix='shards', resample=16000, num_threads=32):
    """"""
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    torch.set_num_threads(1)

    data = []
    with open(jsonl_file, 'r', encoding='utf8') as fin:
        for line in fin:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON on line: {line.strip()}")
                raise e

    logging.info(f"Total records loaded: {len(data)}")

    num = num_utts_per_shard
    chunks = [data[i:i + num] for i in range(0, len(data), num)]
    os.makedirs(shards_dir, exist_ok=True)

    # 使用线程池加速处理
    pool = multiprocessing.Pool(processes=num_threads)
    shards_list = []
    num_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        tar_file = os.path.join(shards_dir, '{}_{:09d}.tar'.format(prefix, i))
        shards_list.append(tar_file)
        pool.apply_async(
            _write_tar_file,
            (chunk, tar_file, resample, i, num_chunks))

    pool.close()
    pool.join()

    with open(os.path.join(shards_dir,'shards_list.txt'), 'w', encoding='utf8') as fout:
        for name in shards_list:
            fout.write(name + '\n')

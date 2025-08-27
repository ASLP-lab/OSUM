import logging
import os
from typing import Dict, List, Optional, Union
import torchaudio
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, StoppingCriteriaList, AutoConfig

from patches.cumstom_stop_criteria import InterruptStopper, S2SStopCriteria, MaxTokenStopper
from patches.custom_speech_ngram_blocking import SpeechOnlyNGramBlockingLogitsProcessor, OSUM_chat_LogitsProcessor
from wenet.osum_echat.utils4llmasr import make_streaming_mode_from_s2s, do_embedding_for_two_embeds
from wenet.transformer.encoder import TransformerEncoder, TransformerEncoder2
from wenet.osum_echat.utils4llmasr import *
from gxl_ai_utils.utils import utils_file

from wenet.osum_echat.downsampler import get_downsampler, osum_echat2Conv1dSubsampling
from wenet.transformer.swish import New_gelu4npu
from wenet.utils.mask import make_pad_mask


class LLMASR_Model(nn.Module):
    def __init__(self,
                 encoder,
                 encoder_output_dim,
                 llm_path,
                 lora=True, lora_alpha=32, lora_rank=8, lora_dropout=0.1,
                 is_inference=False,
                 downsample_rate=1,
                 adapter_type='osum_echat2',
                 speech_token_num=0,
                 train_speech_out=False):
        """"""
        super().__init__()
        utils_file.logging_limit_print(f"instruct_version: LLMASR_Model init, is_inference={is_inference}, downsample_rate={downsample_rate}, adapter_type={adapter_type}, speech_token_num={speech_token_num}, train_speech_out={train_speech_out}")
        self.downsample_rate = downsample_rate

        self.encoder = encoder
        self.ln_speech = nn.LayerNorm(encoder_output_dim)

        # 连接层, 51.6M
        if adapter_type == 'osum_echat':
            self.speech_transformer = TransformerEncoder(
                input_size=encoder_output_dim,
                output_size=encoder_output_dim,
                attention_heads=4,
                linear_units=2560,
                num_blocks=4,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.0,
                input_layer="linear",
                pos_enc_layer_type="abs_pos",
                normalize_before=True
            )
        else:
            self.speech_transformer = None

        # self.llama_model = AutoModelForCausalLM.from_pretrained(
        #     llm_path,
        #     # torch_dtype=torch.float32 if is_inference else torch.float16,
        #     torch_dtype=torch.bfloat16,
        #     trust_remote_code=True,
        #     output_hidden_states=True,
        # )
        self.config = AutoConfig.from_pretrained(
            llm_path,
            trust_remote_code=True,  # 若模型有自定义代码需开启
            output_hidden_states=True,  # 根据需求配置
        )
        # 2. 基于配置创建空模型（权重随机初始化，不加载预训练参数）
        self.llama_model = AutoModelForCausalLM.from_config(
            self.config,
            torch_dtype=torch.bfloat16,  # 指定数据类型
            trust_remote_code=True,  # 与config保持一致
        )
        self.s2s_stop_criteria = None
        self.max_token_criteria_list = None

        self.max_length = 4000
        self.min_length = 1
        self.num_beams = 4
        self.do_sample = True
        self.top_p = 0.9
        self.top_k = 5
        self.repetition_penalty = 1.05
        self.length_penalty = 1.0
        self.temperature = 1.0
        self.IGNORE_ID = -100

        # lora
        self.lora = lora
        if lora:
            utils_file.logging_limit_print("OSUM-EChat： 使用lora了")
            # target_modules = ['w_pack', 'o_proj', 'gate_proj', 'down_proj']
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj']
            self.peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=is_inference,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, use_fast=False, trust_remote_code=True)
        """
        设置分词器的pad_token和padding的方向。
        """
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

        if hasattr(self.llama_model.config, 'hidden_size'):
            utils_file.logging_limit_print(
                f"self.llama_model.config.hidden_size: {self.llama_model.config.hidden_size}")
            if adapter_type == 'osum_echat2':
                self.down_sample_2 = osum_echat2Conv1dSubsampling(encoder_output_dim, self.llama_model.config.hidden_size)
            elif adapter_type == 'osum_echat':
                self.down_sample_2 = get_downsampler(downsample_rate, encoder_output_dim)
                self.speech_llama_proj = nn.Linear(
                    encoder_output_dim, self.llama_model.config.hidden_size)
        else:
            raise NotImplementedError("self.llama_model.config.hidden_size not exist")

        self.embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        self.lm_head = self.llama_model.model.lm_head if self.lora else self.llama_model.lm_head
        self.llm_vocab_size  = self.lm_head.weight.shape[0]
        self.speech_token_num = speech_token_num
        # init speech token module
        if speech_token_num > 0:
            utils_file.logging_info(f'OSUM-EChat： 进行语音token生成任务， speech_token_num: {speech_token_num}')
            self.speech_token_emded = torch.nn.Embedding(speech_token_num + 2, self.llama_model.config.hidden_size)
            self.speech_head = torch.nn.Linear(self.llama_model.config.hidden_size, speech_token_num)
        else:
            # 不做任何处理
            self.speech_head = nn.Identity()
            self.speech_token_emded = nn.Identity()
            self.speech_model = nn.Identity()
        self.train_speech_out = train_speech_out
        utils_file.logging_info(f'OSUM-EChat： 是否进行语音输出训练：{self.train_speech_out}')
        self.loss_fct = CrossEntropyLoss(reduction='mean')
        self.unk_token_id = 7672 # &&对应的id
        self.add_embed_head = True
        self.init_custom_speech_repetition_penalty()
        self.init_custom_stop_criteria()

    def set_task_type(self, task_type: str):
        """设置任务类型，用于设置生成的初始类型
        Args:
            task_type (str): 任务类型，从("ASR", "TTS", "S2S")选择
        """
        assert task_type in ("ASR", "TTS", "S2S")
        if task_type == "ASR":
            self.llama_model.text_phase = True
        elif task_type == "TTS":
            self.llama_model.text_phase = False
        elif task_type == "S2S":
            self.llama_model.text_phase = True

    def do_add_speech_embed_head(self):
        if self.add_embed_head:
            self.llama_model.speech_token_emded = self.speech_token_emded.to(torch.bfloat16)
            self.llama_model.speech_head = self.speech_head.to(torch.bfloat16)
            # self.llama_model.speech_token_emded = self.speech_token_emded.to(torch.bfloat16)
            # self.llama_model.speech_head = self.speech_head.to(torch.bfloat16) # 带lora的时候用
            self.add_embed_head = False


    def init_custom_speech_repetition_penalty(self):
        """

        """
        self.s2s_repetition_penalty = LogitsProcessorList()
        # self.speech_repetition_penalty = SpeechOnlyRepetitionPenaltyLogitsProcessor(speech_token_num=4097, penalty=1.5)
        self.speech_repetition_penalty = SpeechOnlyNGramBlockingLogitsProcessor(speech_token_num=4097, repeat_times=5,
                                                                                special_token_repeat_times_dict={
                                                                                    1446: 10})
        self.osum_chat_logit_processor1 = OSUM_chat_LogitsProcessor([99119, 1808, 7863], [102185, 17714, 31252])
        self.s2s_repetition_penalty.append(self.osum_chat_logit_processor1)
        self.s2s_repetition_penalty.append(self.speech_repetition_penalty)
        self.llama_model.speech_repetition_penalty = self.speech_repetition_penalty

    def init_custom_stop_criteria(self):
        """
        创建需要的stop criteria
        1. 对于t2t任务，遇到text_eos停止
        2. 对于t2s任务，遇到speech_eos停止
        3. 对于s2s任务，遇到speech_eos停止
        同时要取消原本的停止条件
        if generation_config._eos_token_tensor is not None:
        取消 generation_config._eos_token_tensor 的停止，尝试直接给一个大于vocb_size的eos_token
        """
        self.interrupt = InterruptStopper()
        self.s2s_stop_criteria = StoppingCriteriaList()
        self.s2s_stop_criteria.append(S2SStopCriteria(text_eos_id=151645, speech_eos_id=self.speech_token_num - 1))
        self.s2s_stop_criteria.append(MaxTokenStopper(2000))
        self.s2s_stop_criteria.append(self.interrupt)

    def get_label_embedding(self, labels, labels_lengths, unk_id=7672):
        """"""
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        unk_mask = (labels == unk_id)  # B, L
        labels_pad_mask = labels_pad_mask | unk_mask  #
        labels = labels.masked_fill(labels_pad_mask, 0)
        labels_embeds = self.embed_tokens(labels)
        labels_target = labels.masked_fill(labels_pad_mask, self.IGNORE_ID)  # B, L
        labels_mask = ~labels_pad_mask
        return labels_embeds, labels_target, labels_mask

    def get_speech_token_label_embedding(self, speech_token_labels, speech_tokens_length):
        """"""
        speech_tokens_pad_mask = make_pad_mask(speech_tokens_length)  # B, L
        speech_token_labels = speech_token_labels.masked_fill(speech_tokens_pad_mask, 0)
        speech_token_labels_embeds = self.speech_token_emded(speech_token_labels)
        # utils_file.logging_limit_print(f'进行speech_token_labels修改，修改前 speech_token_labels',
        #    speech_token_labels.shape, speech_token_labels[0][-1], speech_token_labels[0][0])
        speech_token_labels = speech_token_labels + self.llm_vocab_size
        # utils_file.logging_limit_print(f'进行speech_token_labels修改，修改后 speech_token_labels',
        #    speech_token_labels.shape, speech_token_labels[0][-1], speech_token_labels[0][0])
        speech_token_labels_target = speech_token_labels.masked_fill(speech_tokens_pad_mask, self.IGNORE_ID)  # B, L
        speech_token_labels_mask = ~speech_tokens_pad_mask
        return speech_token_labels_embeds, speech_token_labels_target, speech_token_labels_mask



    def generate(
            self,
            wavs,
            wavs_len,
            prompt,
            **kwargs
    ):
        self.llama_model.eval()
        self.set_task_type("ASR")
        self.do_add_speech_embed_head()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)
        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, an audio understanding. You can transcribe speech accurately and anaosum_echat2e paralinguistic cues to provide precise text responses.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        embeds = torch.cat([prompt_pattern1_embeds, prompt_embeds, speech_embeds, prompt_pattern2_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            embeds = embeds.to(torch.bfloat16)
            atts = atts.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            cache_implementation="static",
            # num_beams=self.num_beams,
            do_sample=self.do_sample,
            min_length=self.min_length,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            # attention_mask=atts,
            eos_token_id=151645,
            pad_token_id=-100,
            stopping_criteria=self.max_token_criteria_list,
            do_compile=True,
        )
        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        return output_text

    def generate4chat(
            self,
            wavs,
            wavs_len,
            prompt=" ",
            do_sample=True,
            top_k=2,
            top_p=1,
            temperature=0.4,
            **kwargs
    ):
        print(f'do_sample: {do_sample}, top_k: {top_k}, top_p: {top_p}, temperature: {temperature}')
        self.llama_model.eval()
        self.set_task_type("ASR")
        self.do_add_speech_embed_head()
        # self.do_merge_embed_head()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        # # sft
        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a speech-to-text dialogue assistant by ASLP Lab. You understand both the meaning and paralinguistic cues in speech then respond exclusively with appropriate text.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        embeds = torch.cat([prompt_pattern1_embeds, speech_embeds, prompt_pattern2_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            # embeds = embeds.to(torch.float16)
            embeds = embeds.to(torch.bfloat16)
            atts = atts.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            cache_implementation="static",
            # num_beams=1,
            do_sample=do_sample,
            min_length=self.min_length,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=1,
            temperature=temperature,
            # attention_mask=atts,
            eos_token_id=151645,
            pad_token_id=-100,
            do_compile=True,
            stopping_criteria=self.max_token_criteria_list,
        )

        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def generate4chat_think(
            self,
            wavs,
            wavs_len,
            prompt=" ",
            do_sample=True,
            top_k=2,
            top_p=1,
            temperature=0.4,
            **kwargs
    ):
        print(f'do_sample: {do_sample}, top_k: {top_k}, top_p: {top_p}, temperature: {temperature}')
        self.llama_model.eval()
        self.set_task_type("ASR")
        self.do_add_speech_embed_head()
        # self.do_merge_embed_head()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        # # sft
        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a thinking-enabled speech-to-text dialogue assistant by ASLP Lab. You not only comprehend the semantic meaning and paralinguistic cues in speech but also engage in deliberate reasoning to process such information. Based on this thinking process, you then respond exclusively with appropriate text.<|im_end|>\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        embeds = torch.cat([prompt_pattern1_embeds, speech_embeds, prompt_pattern2_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            # embeds = embeds.to(torch.float16)
            embeds = embeds.to(torch.bfloat16)
            atts = atts.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            cache_implementation="static",
            # num_beams=1,
            do_sample=do_sample,
            min_length=self.min_length,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=1,
            temperature=temperature,
            # attention_mask=atts,
            eos_token_id=151645,
            pad_token_id=-100,
            do_compile=True,
            stopping_criteria=self.max_token_criteria_list,
        )

        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def generate_tts(
            self,
            device,
            text,
    ):
        # =====================准备input embedding=====================
        self.llama_model.eval()

        self.set_task_type("TTS")
        self.do_add_speech_embed_head()
        # labels_lengths = torch.tensor([len(text[0])], dtype=torch.int64, device=device)
        # labels = text[:,:]
        labels = self.tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids  # (1, L)
        labels = labels.to(device)
        labels_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)
        print(f'label_lengths:{labels_lengths}')
        print(f'labels:{labels}')
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        speech_embeds = self.embed_tokens(labels)  # B, L, D
        speech_target = torch.full(labels_pad_mask.shape, self.IGNORE_ID).to(
            speech_embeds.device)
        speech_masks = ~labels_pad_mask
        # speech_embeds, speech_masks, speech_target = self._add_bos_eos(0 + self.speech_token_num,
        #                                                                1 + self.speech_token_num,
        #                                                                speech_embeds, speech_masks, speech_target)

        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a speech synthesis assistant by ASLP Lab. You generate natural and fluent speech from text input.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)
        hyps = [self.speech_token_num - 1]
        speech_begin_token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        embeds = torch.cat([prompt_pattern1_embeds,
                            speech_embeds,
                            prompt_pattern2_embeds,
                            speech_begin_token_emb], dim=1).to(torch.bfloat16)
        # 指定top_k top_p temperature stop
        # max_len = 250
        top_k = 15  # 5
        top_p = 0.8  # 0.9
        temperature = 1.2  # 1

        print(f"tts eos id = {self.speech_token_num - 1}")
        llm_out = self.llama_model.generate(inputs_embeds=embeds,
                                            max_new_tokens=self.max_length,
                                            eos_token_id=self.speech_token_num - 1,
                                            cache_implementation="static",
                                            do_sample=True,
                                            temperature=temperature,
                                            top_k=top_k,
                                            top_p=top_p,
                                            stopping_criteria=StoppingCriteriaList([MaxTokenStopper(2000)]),
                                            do_compile=True,
                                            repetition_penalty=1.0,
                                            )

        return llm_out



    def generate_text2text(
            self,
            device,
            text,
    ):
        self.llama_model.eval()
        # labels_lengths = torch.tensor([len(text[0])], dtype=torch.int64, device=device)
        # labels = text[:,:]
        labels = self.tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids  # (1, L)
        labels = labels.to(device)
        labels_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)
        # print(f'label_lengths:{labels_lengths}')
        # print(f'labels:{labels}')
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        speech_embeds = self.embed_tokens(labels)  # B, L, D
        speech_target = torch.full(labels_pad_mask.shape, self.IGNORE_ID).to(
            speech_embeds.device)
        speech_masks = ~labels_pad_mask
        # speech_embeds, speech_masks, speech_target = self._add_bos_eos(0 + self.speech_token_num,
        #                                                                1 + self.speech_token_num,
        #                                                                speech_embeds, speech_masks, speech_target)

        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        # # sft
        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a text-to-text dialogue assistant by ASLP Lab. You understand user input in text then respond exclusively with appropriate text.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)
        embeds = torch.cat([prompt_pattern1_embeds, speech_embeds, prompt_pattern2_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            # embeds = embeds.to(torch.float16)
            embeds = embeds.to(torch.bfloat16)
            atts = atts.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=200,
            num_beams=1,
            do_sample=False,
            min_length=self.min_length,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=self.temperature,
            attention_mask=atts,
            eos_token_id=151645,
            pad_token_id=-100,
            do_compile=True,
            cache_implementation="static",
        )
        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        # output_text = [item.replace('<|endoftext|>', '') for item in output_text]
        return output_text

    def generate_s2s_no_stream_with_repetition_penalty(
            self,
            wavs,
            wavs_len,
    ):
        self.llama_model.eval()
        self.set_task_type("S2S")
        self.do_add_speech_embed_head()
        # =====================准备input embedding=====================
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a dialogue assistant by ASLP Lab. You understand both the meaning and paralinguistic cues in speech, as well as input text, and respond appropriately.<|im_end|>\n<|im_start|>user\n"
        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a speech-to-speech dialogue assistant by ASLP Lab. You understand both the meaning and paralinguistic cues in speech then respond with appropriate text and emotionally matching synthetic speech.<|im_end|>\n<|im_start|>user\n"
        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)


        hyps = [4098]
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)

        embeds = torch.cat(
            [prompt_pattern1_embeds, speech_embeds, token_emb, prompt_pattern2_embeds],
            dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)

        # max_len = 350
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        invalid_eos = 10000000
        self.osum_chat_logit_processor1.init_match_found()
        llm_out = self.llama_model.generate(inputs_embeds=embeds,
                                            max_new_tokens=self.max_length,
                                            eos_token_id=invalid_eos,
                                            cache_implementation="static",
                                            do_sample=True,
                                            temperature=temperature,
                                            top_k=top_k,
                                            top_p=top_p,
                                            logits_processor=self.s2s_repetition_penalty,
                                            stopping_criteria=self.s2s_stop_criteria,
                                            do_compile=True,
                                            repetition_penalty=1.0,
                                            )

        text_eos_idx = (llm_out[0] == 151645).nonzero(as_tuple=True)[0][0].item()
        text_res = llm_out[:, :text_eos_idx - 1]
        speech_res = llm_out[:, text_eos_idx + 1:-1]

        # print("llm_out", llm_out)
        output_text = self.tokenizer.batch_decode(text_res, add_special_tokens=False, skip_special_tokens=True)
        # print(f'output_text:{output_text}')
        # print(f'speech_res:{speech_res}')
        return (output_text, text_res, speech_res)


    def generate_s2s_no_stream_think_with_repetition_penalty(
            self,
            wavs,
            wavs_len,
    ):
        self.llama_model.eval()
        self.set_task_type("S2S")
        self.do_add_speech_embed_head()
        # =====================准备input embedding=====================
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a dialogue assistant by ASLP Lab. You understand both the meaning and paralinguistic cues in speech, as well as input text, and respond appropriately.<|im_end|>\n<|im_start|>user\n"
        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a speech-to-speech dialogue assistant by ASLP Lab. You understand both the meaning and paralinguistic cues in speech then respond with appropriate text and emotionally matching synthetic speech.<|im_end|>\n<|im_start|>user\n"
        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are OSUM-chat, a speech-to-speech dialogue assistant by ASLP Lab. You understand both the meaning and paralinguistic cues in speech. Before responding, first output your reasoning inside <think>...</think end>, analyzing the user’s words and vocal cues. Then generate a reply with appropriate text and emotionally matched synthetic speech.<|im_end|>\n<|im_start|>user\n"
        # qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)


        hyps = [4098]
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)

        embeds = torch.cat(
            [prompt_pattern1_embeds, speech_embeds, token_emb, prompt_pattern2_embeds],
            dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)

        # max_len = 350
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        invalid_eos = 10000000
        self.osum_chat_logit_processor1.init_match_found() # 非think不用匹配
        llm_out = self.llama_model.generate(inputs_embeds=embeds,
                                            max_new_tokens=self.max_length,
                                            eos_token_id=invalid_eos,
                                            cache_implementation="static",
                                            do_sample=True,
                                            temperature=temperature,
                                            top_k=top_k,
                                            top_p=top_p,
                                            logits_processor=self.s2s_repetition_penalty,
                                            stopping_criteria=self.s2s_stop_criteria,
                                            do_compile=True,
                                            repetition_penalty=1.0,
                                            )

        text_eos_idx = (llm_out[0] == 151645).nonzero(as_tuple=True)[0][0].item()
        text_res = llm_out[:, :text_eos_idx - 1]
        speech_res = llm_out[:, text_eos_idx + 1:-1]

        # print("llm_out", llm_out)
        output_text = self.tokenizer.batch_decode(text_res, add_special_tokens=False, skip_special_tokens=True)
        # print(f'output_text:{output_text}')
        # print(f'speech_res:{speech_res}')
        return (output_text, text_res, speech_res)


    def _get_embedding_from_wav(self, wavs, wavs_len):
        """
        return:
        wav_embedding: (b, l, v)
        wav_mask:  (b, l), wav为有效值的位置为true
        """
        encoder_out, encoder_mask = self.encoder(wavs, wavs_len)

        speech_embeds, encoder_mask = self.down_sample_2(encoder_out, encoder_mask)
        if self.speech_transformer is not None:
            filled_wavs_len = encoder_mask.squeeze(1).sum(-1)
            speech_embeds, encoder_mask = self.speech_transformer(speech_embeds, filled_wavs_len)
            # if rank == 0:
            #     utils_file.logging_limit_print(
            #         f'out of link shape: {speech_embeds.shape},encoder的第一帧的前20个数字：\n {speech_embeds[0][0][:20]}')

            # utils_file.logging_limit_print(
            #     'get_embedding_from_wav(): speech_embeds.shape,by  self.speech_transformer(speech_embeds, speech_lens):',
            #     speech_embeds.shape)
            speech_embeds = self.speech_llama_proj(speech_embeds)
            # if rank == 0:
            #     utils_file.logging_limit_print(
            #         f'out of speech_llama_proj shape: {speech_embeds.shape},encoder的第一帧的前20个数字：\n {speech_embeds[0][0][:20]}')

        # utils_file.logging_limit_print(
        #     'get_embedding_from_wav(): speech_embeds.shape,by  self.speech_llama_proj(speech_embeds):',
        #     speech_embeds.shape)

        return speech_embeds, encoder_mask.squeeze(1)

    def _get_embedding_from_text(self, text):
        """
        将字符串先量化，再转成词向量

        Args:
            text: str

        Returns:
            text_embeds: (1, L, D)

        """
        text_id = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids
        text_embeds = self.embed_tokens(text_id)
        text_embeds_len = torch.tensor([text_embeds.size(1)], dtype=torch.long)
        return text_embeds, text_embeds_len

    def _add_bos_eos(self, bos, eos, inputs_embeds, attention_mask, target=None):
        B = len(inputs_embeds)
        bos_eos_target = torch.full([B, 1], self.IGNORE_ID).to(inputs_embeds.device)  # B,1
        bos_eos_mask = torch.full([B, 1], True).to(inputs_embeds.device)  # B, 1

        if bos is not None:
            bos_embed = self.speech_token_emded(torch.full([B, 1],
                                                           bos).to(inputs_embeds.device))  # B, 1, D
            inputs_embeds = torch.cat((bos_embed, inputs_embeds), 1)  # B, (1+T), D
            attention_mask = torch.cat((bos_eos_mask, attention_mask), 1)  # B, (1+T)
            if target is not None:
                target = torch.cat((bos_eos_target, target), 1)  # B, (1+T), D

        if eos is not None:
            eos_embed = self.speech_token_emded(torch.full([B, 1],
                                                           eos).to(inputs_embeds.device))  # B, 1, D
            inputs_embeds = torch.cat((inputs_embeds, eos_embed), 1)  # B, (1+T+1), D
            attention_mask = torch.cat((attention_mask, bos_eos_mask), 1)  # B, (1+T+1)
            if target is not None:
                target = torch.cat((target, bos_eos_target), 1)  # B, (1+T+1), D

        return inputs_embeds, attention_mask, target




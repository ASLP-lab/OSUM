import logging
import os

import torchaudio
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from wenet.transformer.encoder import TransformerEncoder
from wenet.llm_asr.utils4llmasr import *
from gxl_ai_utils.utils import utils_file

from wenet.llm_asr.downsampler import get_downsampler, LyzConv1dSubsampling
from wenet.utils.mask import make_pad_mask



class LLMASR_Model(nn.Module):
    def __init__(self,
                 encoder,
                 encoder_output_dim,
                 llm_path,
                 lora=True, lora_alpha=32, lora_rank=8, lora_dropout=0.1,
                 prompt_pattern="{}：<Speech><SpeechHere></Speech>",
                 # "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:"
                 is_inference=False,
                 downsample_rate=1,
                 llm_embed_dim=4096,
                 task_num=2,
                 adapter_type='lyz',
                 speech_token_num=0,
                 train_speech_out=False):
        """"""
        super().__init__()
        self.downsample_rate = downsample_rate

        self.encoder = encoder
        self.ln_speech = nn.LayerNorm(encoder_output_dim)

        # 连接层, 51.6M
        if adapter_type == 'gxl':
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

        # LLM,
        self.low_resource = False
        if not self.low_resource:
                self.llama_model = AutoConfig.from_pretrained(llm_path)
                self.llama_model = AutoModelForCausalLM.from_config(self.llama_model,
                                                                    torch_dtype=torch.bfloat16,
                                                                    device_map="auto",
                                                                    trust_remote_code=True,
                                                                    output_hidden_states=True,)
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
            )

        self.max_length = 400
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
            utils_file.logging_limit_print("OSUM: 使用lora了")
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj']
            if is_inference:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
            else:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
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

        if hasattr(self.llama_model.config, 'hidden_size'):
            utils_file.logging_limit_print(
                f"self.llama_model.config.hidden_size: {self.llama_model.config.hidden_size}")
            if adapter_type == 'lyz':
                self.down_sample_2 = LyzConv1dSubsampling(encoder_output_dim, self.llama_model.config.hidden_size)
            elif adapter_type == 'gxl':
                self.down_sample_2 = get_downsampler(downsample_rate, encoder_output_dim)
                self.speech_llama_proj = nn.Linear(
                    encoder_output_dim, self.llama_model.config.hidden_size)
            # self.task_embeddings = torch.nn.Embedding(task_num, self.llama_model.config.hidden_size)
        else:
            raise NotImplementedError("self.llama_model.config.hidden_size not exist")

        self.embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        self.lm_head = self.llama_model.model.lm_head if self.lora else self.llama_model.lm_head

        self.speech_token_num = speech_token_num
        # init speech token module
        if speech_token_num > 0:
            self.speech_token_emded = torch.nn.Embedding(speech_token_num + 2, self.llama_model.config.hidden_size)
            self.speaker_head = torch.nn.Linear(self.llama_model.config.hidden_size, speech_token_num)
        else:
            # 不做任何处理
            self.speaker_head = nn.Identity()
            self.speech_token_emded = nn.Identity()
        self.train_speech_out = train_speech_out
        self.loss_fct = CrossEntropyLoss(reduction='mean')

    def get_label_embedding(self, labels, labels_lengths):
        """"""
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        labels_embeds = self.embed_tokens(labels)
        labels_target = labels.masked_fill(labels_pad_mask, self.IGNORE_ID)  # B, L
        labels_mask = ~labels_pad_mask
        return labels_embeds, labels_target, labels_mask

    def forward(self,
                batch,
                device,
                ):
        """"""
        rank = int(os.environ.get('RANK', 0))
        output_type = batch['output_type']
        assert output_type in ['text',], f"output_type:{output_type} not support"
        wavs = batch['feats'].to(device)
        wavs_len = batch['feats_lengths'].to(device)
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_target = torch.full(speech_masks.shape, self.IGNORE_ID).to(
            speech_embeds.device)

        # add bos and eos
        speech_embeds, speech_masks, speech_target = self._add_bos_eos(0 + self.speech_token_num,
                                                                       1 + self.speech_token_num,
                                                                       speech_embeds, speech_masks, speech_target)

        # prompt
        if 'prompt' in batch:
            prompt = batch['prompt'].to(device)
            prompt_lengths = batch['prompt_lengths'].to(device)
            prompt_pad_mask = make_pad_mask(prompt_lengths)  # B, L
            prompt = prompt.masked_fill(prompt_pad_mask, self.tokenizer.eos_token_id)
            prompt_embeds = self.embed_tokens(prompt)  # B, L, D
            prompt_target = torch.full(prompt.shape, self.IGNORE_ID).to(
                speech_embeds.device)  # B, L
            prompt_mask = ~prompt_pad_mask
        else:
            prompt_embeds = None
            prompt_mask = None
            prompt_target = None

        inputs_embeds_list = []
        attention_mask_list = []
        target_list = []
        if output_type == "text":
            labels = batch['target'].to(device)
            labels_lengths = batch['target_lengths'].to(device)
            labels_embeds, labels_target, labels_mask = self.get_label_embedding(labels, labels_lengths)
            if prompt_embeds is not None:
                inputs_embeds_list.append(prompt_embeds)
                attention_mask_list.append(prompt_mask)
                target_list.append(prompt_target)
            else:
                utils_file.logging_limit_print(f'prompt is None,task: {batch["task"]}, prompt_embeds:{prompt_embeds}, prompt_mask:{prompt_mask}')
            inputs_embeds_list.extend([speech_embeds, labels_embeds])
            attention_mask_list.extend([speech_masks, labels_mask])
            target_list.extend([speech_target, labels_target])
        else:
            raise NotImplementedError(f'output_type {output_type} not support')

        inputs_embeds = torch.cat(inputs_embeds_list, dim=1)
        attention_mask = torch.cat(attention_mask_list, dim=1)
        target = torch.cat(target_list, dim=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            labels=target,
            attention_mask=attention_mask,
            position_ids=position_ids.to(inputs_embeds.device)
        )
        loss = outputs['loss']
        return {"loss": loss}

    def generate(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        if prompt != "<no_prompt>":
            prompt = self.tokenizer([prompt], return_tensors="pt"
                                    )['input_ids'].to(speech_embeds.device)
            prompt_embeds = self.embed_tokens(prompt)
        else:
            prompt_embeds = None

        if prompt_embeds is not None:
            embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        else:
            embeds = speech_embeds

        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.bfloat16)
            atts = atts.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            min_length=self.min_length,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            attention_mask=atts,
            eos_token_id=151643,
            pad_token_id=-100,
        )

        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        return output_text


    def get_embedding_from_wav(self, wavs, wavs_len):
        """
        return:
        wav_embedding: (b, l, v)
        wav_mask:  (b, l), wav为有效值的位置为true
        """
        rank = int(os.environ.get('RANK', 0))
        encoder_out, encoder_mask = self.encoder(wavs, wavs_len)
        speech_embeds, encoder_mask = self.down_sample_2(encoder_out, encoder_mask)
        if self.speech_transformer is not None:
            filled_wavs_len = encoder_mask.squeeze(1).sum(-1)
            speech_embeds, encoder_mask = self.speech_transformer(speech_embeds, filled_wavs_len)
            speech_embeds = self.speech_llama_proj(speech_embeds)

        return speech_embeds, encoder_mask.squeeze(1)

    def get_embedding_from_text(self, text):
        text_id = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids
        text_embeds = self.embed_tokens(text_id)
        return text_embeds

    def get_embeds_from_wav_path(self, wav_path):
        wav_i2_path = wav_path
        waveform_i2, _ = torchaudio.load(wav_i2_path)
        if len(waveform_i2.shape) != 1:
            waveform_i2 = waveform_i2[0]
        waveform_i2 = waveform_i2.to(self.embed_tokens.weight.device)
        wavs_len_i2 = torch.tensor([len(waveform_i2)], device=self.embed_tokens.weight.device, dtype=torch.int32)
        wavs_i2 = waveform_i2.unsqueeze(0)
        sample_i2_embeds = self.get_embedding_from_wav(wavs_i2, wavs_len_i2)
        return sample_i2_embeds

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


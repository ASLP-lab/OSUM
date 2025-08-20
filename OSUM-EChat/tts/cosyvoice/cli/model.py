# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Generator
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt
is_npu = True
try:
    import torch_npu
except ImportError:
    is_npu = False
    print(f'torch_npu not found, set is_npu to False')

class CosyVoiceModel:

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool,
                 gpu_id: int = 0):
        if is_npu:
            self.device = torch.device(f'npu:{gpu_id}')
        else:
            self.device = torch.device(f'cuda:{gpu_id}')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # here we fix set flow.decoder.estimator.static_chunk_size = 0 for compatibability
        self.flow.decoder.estimator.static_chunk_size = 0
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

    def load(self, llm_model, flow_model, hift_model):
        self.llm.load_state_dict(torch.load(llm_model, map_location=self.device), strict=True)
        self.llm.to(self.device).eval()
        self.flow.load_state_dict(torch.load(flow_model, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, flow_decoder_onnx_model, fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = self.flow.decoder.estimator_engine.create_execution_context()

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            if isinstance(text, Generator):
                assert isinstance(self, CosyVoice2Model), 'streaming input text is only implemented for CosyVoice2!'
                for i in self.llm.inference_bistream(text=text,
                                                     prompt_text=prompt_text.to(self.device),
                                                     prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                                     prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                                     prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                                     embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
            else:
                for i in self.llm.inference(text=text.to(self.device),
                                            text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_text=prompt_text.to(self.device),
                                            prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                                            prompt_speech_token=llm_prompt_speech_token.to(self.device),
                                            prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                                            embedding=llm_embedding.to(self.device)):
                    self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        tts_mel, flow_cache = self.flow.inference(token=token.to(self.device),
                                                  token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_token=prompt_token.to(self.device),
                                                  prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                                  prompt_feat=prompt_feat.to(self.device),
                                                  prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                                  embedding=embedding.to(self.device),
                                                  flow_cache=self.flow_cache_dict[uuid])
        self.flow_cache_dict[uuid] = flow_cache

        # mel overlap fade in out
        if self.mel_overlap_dict[uuid].shape[2] != 0:
            tts_mel = fade_in_out(tts_mel, self.mel_overlap_dict[uuid], self.mel_window)
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0,
            token_list=None, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        # import pdb;pdb.set_trace()
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    # import pdb;pdb.set_trace()
                    gen_token = [1650, 2163, 3062, 41, 347, 754, 1705, 73, 38, 2583, 59, 1660, 1716, 28, 324, 1260, 1018, 254, 1650, 3552, 1804, 2515, 2368, 38, 1660, 3106, 848, 3250, 1611, 511, 1037, 2964, 2255, 1509, 890, 1494, 2250, 1349, 2621, 3420, 46, 2646, 2646, 3025, 2579, 393, 824, 1609, 2089, 2162, 24, 2, 3768, 1155, 343, 325, 2764, 814, 426, 1243, 2579, 3916, 20, 1611, 349, 701, 1346, 3768, 927, 3305, 8, 2099, 511, 3582, 8, 421, 1494, 2323, 2253, 3607, 692, 3929, 511, 3710, 3662, 3179, 1204, 7, 2579, 2579, 3025, 3025, 571, 540, 1509, 2786, 2548, 1404, 699, 1260, 2250, 202, 202, 84, 3458, 73, 3458, 1716, 302, 2105, 193, 974, 3761, 2893, 2250, 193, 754, 69, 69, 599, 2554, 890, 1608, 148, 1243, 480, 1, 489, 271, 1038, 1736, 1865, 3337, 569, 28, 2246, 2426, 2250, 3768, 569, 1027, 3305, 3106, 8, 3635, 269, 1854, 70, 1385, 1584, 1385, 2187, 3064, 3064, 2579, 3025, 3337, 2579, 3768]
                    token_list = [66, 2307, 599, 1602, 714, 1100, 1243, 2657, 349, 535, 3662, 1403, 2610, 669, 569, 49, 48, 1027, 2684, 373, 728, 728, 186, 186, 7, 2250, 754, 1346, 1289, 2691, 3740, 3082, 629, 2841, 432, 1513, 1716, 302, 3607, 3607, 692, 1609, 2579, 3025, 2513, 2513, 1043, 1043, 2704, 53, 2893, 1043, 2704, 1043, 2513, 2513, 1043, 1083, 3600, 421, 8, 8, 1256, 1243, 3278, 2932, 510, 2515, 2582, 1906, 4056, 1346, 1241, 2253, 1346, 1698, 962, 409, 1507, 1377, 2162, 10, 21, 396, 3649, 373, 728, 2513, 2513, 2513, 2513, 1865, 1893, 1712, 375, 4064, 3062, 41, 569, 3887, 1716, 472, 3830, 186, 408, 203, 3478, 3340, 800, 1243, 480, 271, 2162, 3240, 3238, 3193, 599, 2391, 1317, 1346, 269, 2253, 2209, 8, 1974, 2764, 1579, 421, 1073, 3929, 590, 31, 3898, 53, 53, 1043, 1957]
                    this_tts_speech_token = np.array(token_list)
                    this_tts_speech_token = torch.tensor(this_tts_speech_token)
                    # this_tts_speech_token = np.load("/home/node57_data/hkxie/4O/streaming_fm/data/s3token1/05343304771_EIjYa_VAD27_3.hubert_code.npy")
                    # this_tts_speech_token = torch.tensor(this_tts_speech_token)
    
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech_token = np.array(token_list)
            this_tts_speech_token = torch.tensor(this_tts_speech_token)
            this_tts_speech_token = torch.tensor(this_tts_speech_token).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        torch.cuda.empty_cache()

    def tts_gxl(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0,
            token_list=None, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        # p = threading.Thread(target=self.llm_job,
        #                      args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        # p.start()
        # p.join()
        # this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
        this_tts_speech_token = np.array(token_list)
        this_tts_speech_token = torch.tensor(this_tts_speech_token)
        this_tts_speech_token = torch.tensor(this_tts_speech_token).unsqueeze(dim=0)
        this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                         prompt_token=flow_prompt_speech_token,
                                         prompt_feat=prompt_speech_feat,
                                         embedding=flow_embedding,
                                         uuid=this_uuid,
                                         finalize=True,
                                         speed=speed)
        torch.cuda.empty_cache()
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
            self.flow_cache_dict.pop(this_uuid)
        return {'tts_speech': this_tts_speech.cpu()}

    def vc(self, source_speech_token, flow_prompt_speech_token, prompt_speech_feat, flow_embedding, stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = source_speech_token.flatten().tolist(), True
            self.hift_cache_dict[this_uuid] = None
            self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
            self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        if stream is True:
            token_hop_len = self.token_min_hop_len
            while True:
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     finalize=False)
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
        torch.cuda.empty_cache()


class CosyVoice2Model(CosyVoiceModel):

    def __init__(self,
                 llm: torch.nn.Module,
                 flow: torch.nn.Module,
                 hift: torch.nn.Module,
                 fp16: bool):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.llm.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.llm.half()
            self.flow.half()
        self.token_hop_len = 2 * self.flow.input_frame_rate
        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, token_offset, finalize=False, speed=1.0):
        tts_mel, _ = self.flow.inference(token=token.to(self.device),
                                         token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_token=prompt_token.to(self.device),
                                         prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                         prompt_feat=prompt_feat.to(self.device),
                                         prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                         embedding=embedding.to(self.device),
                                         finalize=finalize)
        tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
        # append hift cache
        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)
        # keep overlap mel and hift cache
        if finalize is False:
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
            self.hift_cache_dict[uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                          'source': tts_source[:, :, -self.source_cache_len:],
                                          'speech': tts_speech[:, -self.source_cache_len:]}
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
            if self.hift_cache_dict[uuid] is not None:
                tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[uuid]['speech'], self.speech_window)
        return tts_speech

    def tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, **kwargs):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.hift_cache_dict[this_uuid] = None
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        if stream is True:
            token_offset = 0
            while True:
                time.sleep(0.1)
                if len(self.tts_speech_token_dict[this_uuid]) - token_offset >= self.token_hop_len + self.flow.pre_lookahead_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_offset + self.token_hop_len + self.flow.pre_lookahead_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                                     prompt_token=flow_prompt_speech_token,
                                                     prompt_feat=prompt_speech_feat,
                                                     embedding=flow_embedding,
                                                     uuid=this_uuid,
                                                     token_offset=token_offset,
                                                     finalize=False)
                    token_offset += self.token_hop_len
                    yield {'tts_speech': this_tts_speech.cpu()}
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) - token_offset < self.token_hop_len + self.flow.pre_lookahead_len:
                    break
            p.join()
            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=token_offset,
                                             finalize=True)
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            # import pdb;pdb.set_trace()
            # this_tts_speech_token = np.load("/home/node57_data/hkxie/4O/streaming_fm/data/s3token2/05343304771_EIjYa_VAD27_3.hubert_code.npy")
            # this_tts_speech_token = np.load("/home/node57_data/hkxie/4O/streaming_fm/data/s3token2/05343304771_EIjYa_VAD41_6.hubert_code.npy")
            # token2 = [2745, 860, 393, 393, 2579, 2926, 1842, 2136, 480, 205, 3910, 3251, 73, 42, 38, 1346, 2554, 368, 40, 1660, 1660, 1055, 2597, 1712, 28, 2246, 386, 122, 38, 3607, 3818, 1098, 980, 38, 1353, 1660, 426, 1694, 1406, 511, 511, 396, 671, 2571, 2809, 2385, 3947, 229, 2000, 773, 2786, 858, 2554, 701, 46, 2646, 1608, 2890, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 3, 31, 758, 3438, 3438, 3438, 54, 269, 2246, 343, 1600, 1608, 3554, 3649, 60, 511, 701, 44, 3554, 3775, 20, 2099, 535, 2099, 3545, 3267, 1223, 1650, 3607, 3611, 2646, 3545, 3545, 802, 802, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 393, 3, 26, 1734, 571, 1240, 1509, 2786, 1509, 740, 890, 2426, 1241, 1241, 2399, 2, 3458, 2285, 25, 2105, 4082, 3761, 3121, 3121, 269, 4082, 1353, 2285, 463, 758, 1193, 421, 3662, 148, 1516, 101, 32, 615, 1660, 1038, 2597, 3554, 28, 2246, 2426, 1241, 22, 1406, 70, 2230, 2230, 3635, 302, 2537, 1385, 1385, 1385, 69, 754, 3489, 1055, 393, 393, 393, 393, 393, 393, 393, 393]
            
            # token_list3 = [2745, 599, 3238, 2554, 84, 73, 42, 2582, 2583, 4082, 1660, 1584, 1469, 1712, 2243, 1260, 1688, 269, 409, 3552, 1584, 2646, 38, 2385, 1660, 1038, 1516, 85, 3250, 1611, 109, 3611, 2255, 3947, 229, 451, 2786, 1044, 2621, 4056, 2646, 2646, 2890, 31, 3898, 3898, 2893, 2893, 2893, 2893, 1043, 52, 52, 52, 52, 1504, 2307, 202, 229, 358, 358, 266, 2907, 1516, 2246, 343, 1030, 122, 2409, 1694, 1406, 511, 2209, 51, 927, 1185, 1256, 1879, 2890, 2858, 203, 2426, 2253, 69, 3011, 3611, 2515, 2646, 492, 3662, 1608, 7, 31, 1406, 1406, 2893, 1043, 728, 380, 380, 571, 2385, 229, 740, 3193, 358, 202, 3331, 2, 1796, 35, 2285, 1893, 1516, 329, 3761, 2859, 122, 1241, 329, 1906, 59, 460, 463, 2554, 740, 1608, 60, 1516, 101, 1, 489, 1038, 1038, 3337, 3768, 569, 32, 1494, 2250, 3768, 3649, 20, 351, 1404, 1193, 44, 59, 3607, 2174, 1584, 1584, 1584, 1655, 1736, 1043, 1043, 1469, 569, 28, 2000, 2426, 2250, 3768, 927, 3250, 8, 2099, 1716, 59, 792, 3106, 1385, 1385, 1385, 1385, 1385, 3947, 1507, 864, 52, 52, 52] 
            token_list3 = [997, 966, 3554, 1854, 714, 3761, 3741, 2426, 103, 103, 1260, 1260, 2306, 2306, 2307, 824, 792, 193, 1879, 3478, 48, 511, 3420, 1317, 1761, 599, 1002, 980, 2646, 2646, 2646, 2646, 2646, 3366, 1949, 575, 575, 26, 26, 29, 3929, 229, 3910, 568, 3265, 3768, 28, 2004, 3910, 568, 3265, 3062, 41, 927, 699, 304, 2859, 2537, 28, 3741, 2841, 1688, 3768, 28, 1155, 855, 1570, 1570, 1570, 1570, 1570, 2876, 2680, 3, 3, 3636, 1555, 2844, 409, 1040, 2515, 1640, 3121, 3153, 882, 2385, 1796, 1796, 1796, 2368, 1785, 49, 671, 3830, 3025, 2844, 2105, 1037, 1729, 2105, 3265, 103, 1346, 580, 3922, 2876, 42, 271, 59, 3106, 2680, 3830, 2704, 2105, 2815, 59, 1698, 1223, 1342, 3267, 2786, 2250, 2250, 2208, 3, 1446, 1446, 1446, 1446, 1446, 1446, 1446, 1446, 1446, 1446, 1688, 1688, 1446, 1446, 1688, 1688, 1688, 1688, 1688]
            this_tts_speech_token = np.array(token_list3)
            this_tts_speech_token = torch.tensor(this_tts_speech_token)
            this_tts_speech_token = torch.tensor(this_tts_speech_token).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(token=this_tts_speech_token,
                                             prompt_token=flow_prompt_speech_token,
                                             prompt_feat=prompt_speech_feat,
                                             embedding=flow_embedding,
                                             uuid=this_uuid,
                                             token_offset=0,
                                             finalize=True,
                                             speed=speed)
            yield {'tts_speech': this_tts_speech.cpu()}
        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
        torch.cuda.empty_cache()

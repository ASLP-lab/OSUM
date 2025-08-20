import torch
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteria

class ASRLogitsProcessor(LogitsProcessor):
    def __init__(self, text_token_num: int):
        self.text_token_num = text_token_num
        
    def __call__(self, input_ids, scores):
        scores[..., self.text_token_num:] = torch.finfo(scores.dtype).min
        return scores

class TTSLogitsProcessor(LogitsProcessor):
    """
    TTS 任务使用的LogitsProcessor，把所有text位置的logits设置为负无穷
    """
    def __init__(self, text_token_num: int):
        self.text_token_num = text_token_num
        
    def __call__(self, input_ids, scores):
        scores[..., :self.text_token_num] = torch.finfo(scores.dtype).min
        return scores

class S2SLogitsProcessor(LogitsProcessor):
    """Speech 2 Speech 任务使用的 LogitsProcessor，当前只适用于batch_size=1

    Args:
        LogitsProcessor (_type_): _description_
    """
    def __init__(self, text_token_num: int, text_eos_id: int):
        self.text_token_num = text_token_num
        self.text_eos_id = text_eos_id
        self.text_phase = True
    def __call__(self, input_ids, scores):
        print(input_ids.shape)
        assert input_ids.size(0) == 1, "ERROR: S2SSpeechLogitsProcessor only support bs=1 now"
        if self.text_phase:
            scores[..., self.text_token_num:] = torch.finfo(scores.dtype).min
        else:
            scores[..., :self.text_token_num] = torch.finfo(scores.dtype).min
        
        if self.text_phase and torch.isin(input_ids, self.text_eos_id):
            self.text_phase = False
            
        return scores

class S2SStopCriteria(StoppingCriteria):
    """Speech 2 Speech 任务使用的 停止条件，当前只适用于batch_size=1

    Args:
        LogitsProcessor (_type_): _description_
    """
    def __init__(self, text_eos_id: int, speech_eos_id: int):
        self.text_eos_id = text_eos_id
        self.speech_eos_id = speech_eos_id
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        _input_ids = input_ids.flatten().view(-1)
        if torch.isin(_input_ids, self.text_eos_id).any():
            text_eos_idx = (_input_ids == self.text_eos_id).nonzero(as_tuple=True)[0][0].item()
            if torch.sum(_input_ids[text_eos_idx:] == self.speech_eos_id) > 1:
                return True
        return False

class MaxTokenStopper(StoppingCriteria):
    def __init__(self, max_tokens):
        self.max_tokens = max_tokens

    # TODO@wsy:期望能够修改max_tokens，但好像没用，后续注意
    def change_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens
        
    def __call__(self, input_ids, scores, **kwargs):
        return input_ids.shape[1] >= self.max_tokens  # 检查当前序列长度

class InterruptStopper(StoppingCriteria):
    def __init__(self):
        self.stop = False
        
    def __call__(self, input_ids, scores, **kwargs):
        if self.stop == True:
            # self.stop == False # reset
            return True
        else:
            return False
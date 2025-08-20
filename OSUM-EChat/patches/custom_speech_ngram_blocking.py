from transformers.generation.logits_process import LogitsProcessor
import torch

class SpeechOnlyNGramBlockingLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        speech_token_num,
        repeat_times=5,
        special_token_repeat_times_dict=None,
        window_size=8,
        window_repeat=5,
        special_token_window_dict=None
    ):
        """
        speech_token_num: int, speech token 的数量（token_id in [0, speech_token_num) 视为 speech token）
        repeat_times: int, 普通 speech token 的最大允许连续重复次数
        special_token_repeat_times_dict: dict, {token_id: repeat_times}，为特殊 speech token 单独指定最大连续重复次数
        window_size: int, 默认滑动窗口大小
        window_repeat: int, 默认窗口内最大允许出现次数
        special_token_window_dict: dict, {token_id: (window_size, window_repeat)}，为特殊 token 单独指定窗口参数
        """
        self.speech_token_num = speech_token_num
        self.repeat_times = repeat_times
        self.special_token_repeat_times_dict = special_token_repeat_times_dict or {}
        self.speech_phase = False  # 你需要在外部控制这个变量
        self.window_size = window_size
        self.window_repeat = window_repeat
        self.special_token_window_dict = special_token_window_dict or {1446: (13, 10)}

    def set_phase(self, speech_phase: bool):
        self.speech_phase = speech_phase

    def __call__(self, input_ids, scores):
        if not self.speech_phase:
            # text 阶段，什么都不做
            return scores
        batch_size, seq_len = input_ids.size()
        for batch_idx in range(batch_size):
            generated = input_ids[batch_idx].tolist()
            if seq_len == 0:
                continue
            last_token = generated[-1]
            if last_token >= self.speech_token_num:
                continue  # 不是 speech token

            # 统计最近的 token 连续重复了多少次
            repeat_count = 1
            for i in range(seq_len-2, -1, -1):
                if generated[i] == last_token:
                    repeat_count += 1
                else:
                    break
            # 获取该 token 的最大允许重复次数
            max_repeat = self.special_token_repeat_times_dict.get(last_token, self.repeat_times)
            if repeat_count >= max_repeat:
                scores[batch_idx, last_token] = -float('inf')  # 阻止生成

            # ====== 滑动窗口内频率抑制 ======
            # 对窗口内所有 speech token 检查
            window_tokens = set(generated[-max(self.window_size, max([v[0] for v in self.special_token_window_dict.values()], default=0)):])
            for token in window_tokens:
                if token >= self.speech_token_num:
                    continue
                # 获取该 token 的窗口参数
                window_size, window_repeat = self.special_token_window_dict.get(
                    token, (self.window_size, self.window_repeat)
                )
                window = generated[-window_size:]
                if window.count(token) >= window_repeat:
                    scores[batch_idx, token] = -float('inf')
            # ====== 滑动窗口内频率抑制结束 ======
        return scores




class OSUM_chat_LogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_tokens, sequence_to_match):
        """
        初始化OSUM_chat_LogitsProcessor。

        参数：
        allowed_tokens (list): 允许出现在当前时间步的token的ID列表
        sequence_to_match (list): 用来判断当前时间步允许token的前置序列
        """
        self.allowed_tokens = allowed_tokens
        self.sequence_to_match = sequence_to_match
        self.match_found = False  # 添加一个标志，表示是否已经找到匹配的序列

    def init_match_found(self):
        """
        初始化match_found标志。
        """
        self.match_found = False

    def __call__(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        """
        在每个时间步处理logits，对不符合条件的token设置极小的概率。

        参数：
        input_ids (torch.Tensor): 当前输入的token ID序列
        logits (torch.Tensor): 当前时间步的logits (shape: [batch_size, vocab_size])

        返回：
        torch.Tensor: 被处理过的logits
        """
        # 如果已经匹配过一次，就跳过匹配检测，直接返回logits
        # print("recent_tokens:！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")  # 打印当前生成的序列
        if self.match_found:
            return logits

        # 获取当前生成的序列的最后几个token（假设生成的长度大于等于序列长度）
        sequence_length = len(self.sequence_to_match)
        if input_ids.shape[-1] >= sequence_length:
            recent_tokens = input_ids[:, -sequence_length:].tolist()
            # print("recent_tokens:", recent_tokens)   # 打印当前生成的序列

            # 检查前面生成的token是否匹配我们需要的序列
            if all(recent_tokens[0][i] == self.sequence_to_match[i] for i in range(sequence_length)):
                # Create a mask for allowed tokens while preserving original logits
                mask = torch.zeros_like(logits, dtype=torch.bool)  # Initialize mask as False
                mask[:, self.allowed_tokens] = True  # Mark allowed tokens as True
                # Apply mask: keep original logits for allowed tokens, set others to -inf
                logits = torch.where(mask, logits, -float('inf'))
                # 设置标志，表示匹配已成功
                self.match_found = True
                print("match found!!!!!!!!!!!!!!!!!!!!!!!")

        return logits

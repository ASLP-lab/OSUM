from transformers.generation.logits_process import LogitsProcessor

class SpeechOnlyRepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, speech_token_num, penalty=1.2):
        self.speech_token_num = speech_token_num
        self.penalty = penalty
        self.speech_phase = False  # 你需要在外部控制这个变量

    def set_phase(self, speech_phase: bool):
        self.speech_phase = speech_phase

    def __call__(self, input_ids, scores):
        if not self.speech_phase:
            # text阶段，什么都不做
            return scores
        # speech阶段，只对speech token做重复抑制
        for batch_idx in range(input_ids.size(0)):
            generated = input_ids[batch_idx].tolist()
            for token_id in set(generated):
                if 0 <= token_id < self.speech_token_num:
                    scores[batch_idx, token_id] /= self.penalty
        return scores
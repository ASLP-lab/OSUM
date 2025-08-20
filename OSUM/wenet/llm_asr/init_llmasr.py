import logging

import torch

from wenet.llm_asr.llmasr_model import LLMASR_Model
from wenet.transformer.cmvn import GlobalCMVN
from wenet.utils.checkpoint import load_checkpoint, load_trained_modules
from wenet.utils.cmvn import load_cmvn

from gxl_ai_utils.utils import utils_file

def init_llmasr(args, configs, is_inference=False):
    llm_path = configs["llm_path"]
    lora = configs["use_lora"]
    lora_alpha = configs["lora_alpha"]
    lora_rank = configs["lora_rank"]
    lora_dropout = configs["lora_dropout"]
    # prompt_pattern = configs['prompt_pattern']

    encoder_output_dim = -1
    if configs['encoder'] == 'transformer':
        if configs.get('cmvn', None) == 'global_cmvn':
            mean, istd = load_cmvn(configs['cmvn_conf']['cmvn_file'],
                                   configs['cmvn_conf']['is_json_cmvn'])
            global_cmvn = GlobalCMVN(
                torch.from_numpy(mean).float(),
                torch.from_numpy(istd).float())
        else:
            global_cmvn = None
        encoder_type = configs.get('encoder', 'conformer')
        input_dim = configs['input_dim']
        from wenet.utils.init_model import WENET_ENCODER_CLASSES
        encoder = WENET_ENCODER_CLASSES[encoder_type](
            input_dim,
            global_cmvn=global_cmvn,
            **configs['encoder_conf'],
            **configs['encoder_conf']['efficient_conf']
            if 'efficient_conf' in configs['encoder_conf'] else {})
        encoder_output_dim = configs['encoder_conf']['output_size']
    elif configs['encoder'] == 'whisper':
        raise NotImplementedError('whisper 还没实现')
    elif configs['encoder'] == 'hubert':
        raise NotImplementedError('hubert 还没实现')
    else:
        encoder = None
    logging.info(f'encoder output dim:{encoder_output_dim}')


    # encoder = encoder.to(torch.float16)
    speech_token_num = configs.get('speech_token_num', 0)
    train_speech_out = speech_token_num != 0

    model = LLMASR_Model(
        encoder=encoder,
        encoder_output_dim=encoder_output_dim,
        llm_path=llm_path,
        lora=lora,
        lora_alpha=lora_alpha,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        is_inference=is_inference,
        downsample_rate=configs.get('downsample_rate',1),
        adapter_type=configs.get('adapter_type', 'lyz'),
        speech_token_num=speech_token_num,
        train_speech_out=train_speech_out,
    )

    utils_file.print_model_size(model.encoder)
    utils_file.print_model_size(model.llama_model)
    # utils_file.print_model_size(model.speech_transformer)
    # utils_file.print_model_size(model.speech_llama_proj)

    logging.info(f'init_salmonn()：开始加载初始化模型')
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        logging.info(f' 设置了初始化模型位置，开始加载，参数文件位置：{args.checkpoint}')
        infos = load_checkpoint(model, args.checkpoint)
    elif hasattr(args, 'checkpoint') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}

    if configs.get('init_step', False):
        infos = {}
    configs["init_infos"] = infos
    print(configs)
    logging.info('加载初始化模型完毕')

    if not is_inference:
        logging.info('不更换LLM的参数')
    else:
        logging.info('不更换LLM的参数')

    logging.info('开始选择性冻结模块')
    fire_module = configs.get("fire_module", None)
    if fire_module is None:
        logging.info('没有选择解冻的模块,也就是没有训练参数，直接报错返回')
        raise ValueError('没有选择解冻的模块,也就是没有训练参数，直接报错返回')
    for k, p in model.named_parameters():
        # if k.startswith("llama_model") or k.startswith("speech_encoder"):
        # if k.startswith("llama_model") or k.startswith("speech_transformer"):
        if fire_module == 'link':
            # link 包括下采样块， transformer块， 前后linear块
            if k.startswith("llama_model") or k.startswith("encoder"):
                p.requires_grad = False
        elif fire_module == 'encoder':
            if not k.startswith("encoder"):
                p.requires_grad = False
        elif fire_module == 'llm':
            if not k.startswith("llama_model"):
                p.requires_grad = False
        elif fire_module == 'link_and_encoder':
            # 这里和speech token相关的层不会被冻结
            if k.startswith("llama_model"):
                p.requires_grad = False
        elif fire_module == "link_and_encoder_and_lora":
            break
        logging.info(f"{k} {p.requires_grad}")
    logging.info('冻结完毕')

    return model, configs

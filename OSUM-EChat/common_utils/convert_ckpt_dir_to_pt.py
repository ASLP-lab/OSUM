from gxl_ai_utils.utils import utils_file
import torch
try:
    import torch_npu
except:
    pass
import os



def convert_ckpt_to_pt(pt_dir_path):
    exp_dir = os.path.dirname(pt_dir_path)
    pt_name = os.path.basename(pt_dir_path)
    weight_dict = torch.load(f"{exp_dir}/{pt_name}/mp_rank_00_model_states.pt", map_location=torch.device('cpu'))[
        'module']
    print(weight_dict.keys())
    torch.save(weight_dict, f"{exp_dir}/{pt_name}.pt")

if __name__ == '__main__':
    pt_dir_path, = utils_file.do_get_commandline_param(1, ["pt_dir_path"])
    exp_dir = os.path.dirname(pt_dir_path)
    pt_name = os.path.basename(pt_dir_path)
    weight_dict = torch.load(f"{exp_dir}/{pt_name}/mp_rank_00_model_states.pt", map_location=torch.device('cpu'))[
        'module']
    print(weight_dict.keys())
    torch.save(weight_dict, f"{exp_dir}/{pt_name}.pt")
# weigth_dict = torch.load("/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch24_cosyvoice1_new-set_token_1w_plus-multi_task_new/step_24999.pt")

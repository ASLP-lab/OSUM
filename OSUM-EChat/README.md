 <p align="left">
        <a href="README_CN.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>



<p align="center">
   <h1>OSUM-EChat: Enhancing End-to-End Empathetic Spoken Chatbot via Understanding-Driven Spoken Dialogue</h1>
<p>

Xuelong Geng, Qijie Shao, Hongfei Xue, Shuiyuan Wang, Hanke Xie, Zhao Guo, Yi Zhao, Guojian Li, Wenjie Tian, Chengyou Wang, Zhixian Zhao, Kangxiang Xia, Ziyu Zhang, Zhennan Lin, Tianlun Zuo, Mingchen Shao, Yuang Cao, Guobin Ma, Longhao Li, Yuhang Dai, Dehui Gao, Dake Guo, Lei Xie


<p align="center">
    <img src="../images/osum-echat/SUM.png" width="500"/>
<p>

<p align="center">
 <a href="https://www.osum-echat.npu-aslp.org/"> Test Page</a> |   <a href="https://huggingface.co/ASLP-lab/OSUM-EChat"> Ckpt</a>
<br>
📑 <a href="https://www.arxiv.org/abs/2508.09600">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/osum-echat.github.io/">Demo</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="../images/wechat.png">WeChat (微信)</a>&nbsp&nbsp 
</p>

Empathy is crucial in enabling natural interactions within spoken dialogue systems, allowing machines to recognize and respond appropriately to paralinguistic cues such as age, gender, and emotion. 
Recent advancements in end-to-end speech language models, which unify speech understanding and generation, provide promising solutions. 
However, several challenges persist, including an over-reliance on large-scale dialogue datasets, insufficient extraction of paralinguistic cues vital for conveying empathy, and the lack of empathy-specific datasets and evaluation frameworks. 
To address these issues, we introduce OSUM-EChat, an open-source, end-to-end spoken dialogue system designed to enhance empathetic interactions, particularly in resource-limited settings. 
Based on [OSUM](https://github.com/ASLP-lab/OSUM/tree/main/OSUM), OSUM-EChat introduces two key innovations: (1) a three-stage understanding-driven spoken dialogue training strategy that extends the capabilities of a large speech understanding model to spoken dialogue tasks, and (2) a linguistic-paralinguistic dual thinking mechanism that integrates paralinguistic understanding through a chain of thought with dialogue generation, enabling the system to produce more empathetic responses. This approach reduces reliance on large-scale dialogue datasets while maintaining high-quality empathetic interactions. Additionally, we introduce the EChat-200K dataset, a rich corpus of empathetic speech-to-speech dialogues, and the EChat-eval benchmark, a comprehensive framework for evaluating the empathetic capabilities of dialogue systems. Experimental results demonstrate that OSUM-EChat outperforms end-to-end spoken dialogue models regarding empathetic responsiveness, validating its effectiveness.
<p align="center">
    <img src="../images/osum-echat/demo_en.png" width="80%"/>
<p>


## Architecture

This section presents an overview of the overall architecture and core tasks of OSUM-EChat. OSUM-EChat consists of three modules: a speech encoder (with an adapter), a text LLM (Large Language Model), and a token-to-speech module. It also possesses a wide range of speech functions, including various speech understanding tasks (speech-to-text), speech synthesis tasks, speech dialogue tasks, and text dialogue tasks. Meanwhile, by leveraging internally constructed empathetic dialogue data and a paralinguistic information reasoning mechanism, OSUM-EChat can generate more empathetic responses in speech dialogue tasks.
<p align="center">
    <img src="../images/osum-echat/system.png" width="80%"/>
<p>


## Training Strategy
To enable OSUM-EChat to achieve empathetic dialogue in resource-constrained environments, the study proposes a three-stage training strategy called **"Understanding-Driven Spoken Dialogue"**, which consists of the stages of understanding, generation, and empathy. In the empathy stage, a **dual-thinking mechanism of linguistic and paralinguistic information** is introduced to explicitly separate paralinguistic and semantic information, thereby helping the model generate more empathetic responses.

### Stage 1: Understanding
The goal of this stage is to enable the LLM to understand both linguistic and paralinguistic information in speech. OSUM’s **“ASR+P” strategy** is employed (where *P* represents paralinguistic labels such as emotion, gender, age, and sound events). Multiple “ASR+P” tasks are jointly trained, with only the encoder and adapters being trainable.

### Stage 2: Generation
This stage aims to equip the OSUM-based understanding model with speech generation capabilities. A two-step training process is adopted: text-to-speech (TTS) generation and speech-to-speech (S2S) dialogue. Additionally, text-to-text (T2T) data is incorporated to maintain the model’s reasoning intelligence.

### Stage 3: Empathy
In this stage, linguistic and paralinguistic information obtained from speech understanding is integrated into the dialogue generation process, significantly improving the model’s ability to produce contextually coherent and empathetic responses. By introducing a **dual-thinking mechanism** before generating text and speech responses, the model first recognizes the linguistic content of the user’s speech, then infers paralinguistic details, and finally integrates these insights to generate appropriate responses.

In the design of the **Chain of Thought (CoT)**, the study explores two different textual forms: **label-based CoT** and **natural language-based CoT**, to investigate how different approaches affect the model’s empathetic understanding and response generation.

#### Label-Based CoT
This form follows a fixed template structure. Specifically, the model first outputs the transcription text obtained via automatic speech recognition (ASR), extracting semantic information from the user’s input. Then, it sequentially outputs predefined paralinguistic labels such as age, gender, speech events, and emotion. The main advantage is that the CoT stage produces content with relatively fixed format and short length, making the process easy to control and efficient in extracting and integrating core paralinguistic cues. However, this approach has limitations: due to the restricted number and scope of predefined labels, it cannot fully express richer and more nuanced paralinguistic states, such as subtle shifts in tone intensity or fine-grained emotional transitions.

#### Natural Language-Based CoT
This form abandons the fixed label template in favor of natural, fluent language descriptions. The model generates coherent textual paragraphs: first interpreting the semantic meaning of the user’s speech (rather than merely transcribing it), then describing the paralinguistic details in depth — including specific manifestations of age characteristics, gender-related vocal traits, the contexts and features of various speech events, and fine-grained emotional layers and dynamics. The advantage of this method is its flexibility in overcoming label limitations, allowing the model to capture and express complex paralinguistic states more comprehensively, thereby providing richer grounding for empathetic response generation. However, its content is harder to control in length and structure, which may increase computational overhead and require stronger language organization abilities from the model.

<p align="center">
    <img src="../images/osum-echat/dual_think_en.png" width="80%"/>
<p>




## Evaluation


### Empathetic Voice Dialogue

In the evaluation using the EChat-eval benchmark, OSUM-EChat demonstrated excellent performance in empathetic dialogue tasks. It achieved high GPT-4o automatic scores across various empathetic dialogue scenarios, particularly excelling in multi-label scenarios. Additionally, it showed strong capability in processing diverse acoustic events in input speech. Detailed results are presented in Table 1.

<p align="center">
    <img src="../images/osum-echat/table1.png" width="65%"/>
</p>
<p align="center"><b>Table 1: Automatic evaluation results on EChat-eval benchmark. Here, 'U-Driven' refers to the understanding-driven spoken dialogue training strategy, and 'Dual Think' refers to the linguistic-paralinguistic dual think mechanism.</b></p>

Human evaluation results from EChat-eval further indicate that OSUM-EChat's overall performance surpasses Qwen2.5-Omni. It demonstrated excellent performance in empathetic dialogue test cases within the emotional dimension, though it still lags behind commercial systems. Notably, in empathetic dialogue tasks involving other paralinguistic dimensions (such as age, gender, sound events, etc.), commercial systems currently cannot effectively capture relevant cues. Detailed data is shown in Table 2.

Ablation experiment results verified that applying the speech understanding model (OSUM) to spoken dialogue tasks, combined with the "linguistic-paralinguistic dual thinking mechanism," can significantly enhance the model's empathetic dialogue capabilities. Specific verification data is presented in Table 1.

<p align="center">
    <img src="../images/osum-echat/table3.png" width="65%"/>
</p>
<p align="center"><b>Table 2: Human evaluation results of representative models on the EChat-eval benchmark. † ByteDance's commercial system with response from a single fixed speaker.</b></p>

### Basic Voice Capabilities

OSUM-EChat demonstrated excellent and stable performance in evaluations of three core voice capabilities: language intelligence, speech understanding, and speech synthesis. Detailed analyses are as follows:

#### (1) Language Intelligence

Leveraging large-scale text dialogue data and an internally constructed knowledge-based voice question-answering dataset, OSUM-EChat's language intelligence level is comparable to mainstream industry end-to-end voice dialogue models. Specific evaluation results for spoken question-answering tasks are shown in Table 3.

<p align="center">
    <img src="../images/osum-echat/table2.png" width="100%"/>
</p>
<p align="center"><b>Table 3: Performance on VoiceBench Benchmarks.</b></p>

#### (2) Speech Understanding

This research verified OSUM-EChat's speech understanding capabilities on open-source test sets for five tasks: Automatic Speech Recognition (ASR), sound event recognition, emotion recognition, age recognition, and gender recognition. Results indicate that its performance is roughly equivalent to the speech understanding large model OSUM (with Qwen-7B as its LLM base) and approaches the level of the industrial-grade speech understanding model Qwen2-Audio.

<p align="center">
    <img src="../images/osum-echat/table4.png" width="80%"/>
</p>
<p align="center"><b>Table 4: Performance of speech understanding tasks</b></p>

#### (3) Speech Synthesis

In this study, the Text-to-Speech (TTS) capability of OSUM-EChat was evaluated on the SEED test set. The results show that its TTS performance is superior to that of the CosyVoice model; however, there is still a gap when compared with industrial-grade voice dialogue models and professional TTS models. Detailed metrics (Word Error Rate, Character Error Rate) are presented in Table 5.

<p align="center">
    <img src="../images/osum-echat/table5.png" width="60%"/>
</p>
<p align="center"><b>Table 5: Performance comparison between OSUM-EChat and recent spoken dialogue models on the SEED test set (Unit: %, ↓ indicates better performance of the metric)</b></p>







## How to Use the OSUM-EChat Code Framework for Training and Inference

### Preparing the Environment

Before starting, make sure your Python environment is ready. Below is a suggested workflow. We assume you already have **conda** installed on your computer.  
If not, please refer to: [Linux One-Click Install Miniconda](https://blog.csdn.net/qq_41636123/article/details/130266232).  
We highly recommend running our code on a Linux system.

```shell
# Create a new conda environment
conda create -n OSUM-EChat python=3.10
# Activate the newly created environment
conda activate OSUM-EChat
# Download our code and install required python packages
git clone https://github.com/ASLP-lab/OSUM.git
cd OSUM/OSUM-EChat
# If training on GPU, remove the torch_npu entry from requirements.txt. 
# If training on NPU, no changes are needed.
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
````

### Understanding Data Types

This project supports three types of data: **raw**, **shard**, and **combine**.
The **combine** type is newly designed based on Wenet’s existing raw and shard formats.

#### **Raw Type**:

Data is stored in **jsonl** format, one JSON object per line, with the following fields:

```
{
"key": "1023390_bed51684_10", 
"txt": "Hello sister, you want to join the expert service system, which shows you care a lot about this issue. Having professionals to answer your questions will definitely give you peace of mind. Don’t worry, such systems usually have detailed processes, take it step by step and you’ll join smoothly～", 
"wav": "./common_utils/fake_data/raw/wav/random.wav", 
"extra": {"age": "<ADULT>", "gender": "<FEMALE>", "think_str": "...", "question": "...", "speech_token": [...], "a_wav_path": "..."},
"task": "<S2TCHAT> <TEXT2TOKEN> <THINK>"
}
```

Example:

```
./common_utils/fake_data/raw/data.list
```

#### **Shard Type**:

Data is packed into **tar files**, storing multiple entries together for efficient bulk loading.

Example:

```
./common_utils/fake_data/shard/shards_list.txt
```

Conversion script (from raw type):

```shell
python ./common_utils/fake_data/shard/do_make_shard_from_raw.py 
```

#### **Combine Type**:

The shard format is efficient for reading but hard to modify. To address this, we designed the **combine type**, which stores audio in **tar packages** while metadata is stored in JSON files.

Example:

```
./common_utils/fake_data/combine/combines_list.txt   # Stores mapping between tar and jsonl
./common_utils/fake_data/combine/combines_tar_root.txt # Stores tar directory
```

* Tar packages must be in the same directory (tar-dir-path + tar-file-name = full path).
* Shares the same type flag as “shard”.
* Output files must be named **combines\_list.txt** and **combines\_tar\_root.txt**.

Conversion script (from shard type):

```shell
python ./common_utils/fake_data/combine/do_make_combine_from_shard.py 
```

#### Data Testing

The project provides scripts to directly construct dataloaders for all three data types:

```shell
python do_test_dataloader.py
```

---

### Inference

This project provides three types of **offline inference** (already open-sourced) and one **real-time online inference** (coming soon).

#### Step 1. Download Model Checkpoints

```python
from huggingface_hub import hf_hub_download

# For natural language think model
pt_file_path = hf_hub_download(repo_id="ASLP-lab/OSUM-EChat", filename="language_think_final.pt")  

# For tag-based think model
pt_file_path2 = hf_hub_download(repo_id="ASLP-lab/OSUM-EChat", filename="tag_think_final.pt")  

# Token2wav model (compressed tar file)
pt_file_path3 = hf_hub_download(repo_id="ASLP-lab/OSUM-EChat", filename="CosyVoice-300M-25Hz.tar")  

# Extract token2wav model parameters
import os
os.system(f"tar -xvf {pt_file_path3}")  
```

#### Gradio-Based Offline Inference

Before running, set checkpoint paths in `./infer_gradio.py`:

```python
CHECKPOINT_PATH_A="**/language_think_final.pt"
CHECKPOINT_PATH_B="**/tag_think_final.pt"
cosyvoice_model_path = "**/CosyVoice-300M-25Hz"
```

* Loads **language\_think**, **tag\_think**, and **token2wav** (requires \~19G memory).
* If `CHECKPOINT_PATH_B=None`, only **language\_think** is loaded.

Run:

```shell
python infer_gradio.py
```

#### Single Utterance Inference

Supports almost all task types: speech understanding, TTS, S2S dialogue, S2T dialogue, T2T dialogue, etc.

```shell
python infer_runtime.py
```

#### Batch Inference

Provides batch inference using dataloaders, supporting all three data types.

```shell
bash infer_with_shards_or_raw.sh
```

#### Flask-Based Online Inference

Coming soon.

---

### Training

Coming soon.

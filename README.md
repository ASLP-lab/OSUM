 <p align="left">
        <a href="README_CN.md">中文</a> &nbsp｜ &nbsp English&nbsp&nbsp
</p>
<p align="center">
   <h1>OSUM: Advancing Open Speech Understanding Models with Limited Resources in Academia</h1>
</p>

Xuelong Geng, Kun Wei, Qijie Shao, Shuiyun Liu*, Zhennan Lin*, Zhixian Zhao*, Guojian Li*, Wenjie Tian*, Peikun Chen, Yangze Li, Pengcheng Guo, Mingchen Shao, Shuiyuan Wang, Yuang Cao, Chengyou Wang, Tianyi Xu, Yuhang Dai, Xinfa Zhu, Yue Li, Li Zhang, Lei Xie†




<p align="center">
    <img src="images/SUM.png" width="400"/>
<p>


<p align="center">
OSUM <a href=""> comming soon</a> </a>&nbsp
<br>
📑 <a href="https://www.arxiv.org/pdf/2501.13306">Paper</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/OSUM.github.io/">Demo</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="images/wechat.png">WeChat (微信)</a>&nbsp&nbsp 
</p>

Large Language Models (LLMs) have made significant progress in various downstream tasks, inspiring the development of Speech Understanding Language Models (SULMs) to enable comprehensive speech-based interactions. However, most advanced SULMs are developed by the industry, leveraging large-scale datasets and computational resources that are not readily available to the academic community. Moreover, the lack of transparency in training details creates additional barriers to further innovation. In this study, we present OSUM, an Open Speech Understanding Model designed to explore the potential of training SLUMs under constrained academic resources. The OSUM model combines a Whisper encoder with a Qwen2 LLM and supports a wide range of speech tasks, including speech recognition (ASR), speech recognition with timestamps (SRWT), vocal event detection (VED), speech emotion recognition (SER), speaking style recognition (SSR), speaker gender classification (SGC), speaker age prediction (SAP), and speech-to-text chat (STTC). By employing an ASR+X training strategy, OSUM achieves efficient and stable multi-task training by simultaneously optimizing ASR alongside target tasks.
Beyond delivering strong performance, OSUM emphasizes transparency by providing openly available data preparation and training methodologies, offering valuable insights and practical guidance for the academic community. By doing so, we aim to accelerate research and innovation in advanced SULM technologies.

## Architecture

The overview of the architecture and tasks of OSUM.

<p align="center">
    <img src="images/system.png" width="80%"/>
<p>

## News and Updates
* Checkpoint and test page coming soon.


<br>

## Evaluation
 Comparison of Qwen2-Audio and our OSUM model. In most tasks, OSUM achieves a better
performance than Qwen2-Audio despite using significantly fewer computational resources and training data.
<p align="center">
    <img src="images/radar.png" width="80%"/>
<p>

Evaluation results of ASR tasks on public and internal test sets. The bold font represents the best
result among the same test set. All internal results are inferred by ourselves.
<p align="center">
    <img src="images/res_asr.png" width="80%"/>
<p>


Evaluation results of multi-tasking on public and internal test sets. The best results for each test set
are highlighted in bold font. Results shown in blue font, as well as those on internal test sets, are inferred
using the original released model by ourselves.
<p align="center">
    <img src="images/res_multi.png" width="80%"/>
<p>

<!--  We have provided **all** evaluation scripts to reproduce our results. Please refer to [eval_audio/EVALUATION.md](eval_audio/EVALUATION.md) for details.
  --> 


## Requirements
```
pip install requirements_little.txt
```
<!-- 
## Quickstart
## Demo
### Web UI

## Citation
 -->
## Contact Us

If you are interested to leave a message to either our research team or product team, feel free to send an email to `xlgeng@mail.nwpu.edu.cn`.

<p align="center">
    <img src="images/ASLP.jpg" width="400"/>
<p>

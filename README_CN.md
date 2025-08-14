 <p align="left">
        &nbsp中文&nbsp&nbsp ｜ <a href="README.md">English</a> &nbsp ｜ <a href="README_JP.md">日本語</a> &nbsp
</p>
<p align="center">
   <h1>OSUM: Advancing Open Speech Understanding Models with Limited Resources in Academia</h1>
</p>

耿雪龙, 魏坤, 邵琪杰, 刘水云*, 林振楠*, 赵致闲*, 李国健*, 田文杰*, 陈培坤, 李泱泽, 郭鹏程, 邵明辰, 王水源, 曹雨昂, 王成有, 徐天翼, 戴宇航, 朱新发, 李越, 张丽, 谢磊†


<p align="center">
    <img src="images/SUM.png" width="400"/>
<p>

<p align="center">
 <a href="https://huggingface.co/spaces/ASLP-lab/OSUM"> Huggingface Test Page</a> </a>&nbsp
<br>
📑 <a href="https://arxiv.org/abs/2501.13306v2">Paper (v2.0)</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://aslp-lab.github.io/OSUM.github.io/">Demo</a> &nbsp&nbsp | &nbsp&nbsp 💬 <a href="images/introduction.md">WeChat (微信)</a>&nbsp&nbsp 
</p>

<h3 align="center">OSUM读作 ‘awesome’ (/ˈɔː.səm/)</h3>

大型语言模型（LLMs）在各种下游任务中取得了显著进展，启发了业界对语音理解语言模型（speech understanding language models, SULMs）的开发，以期实现基于语音情感、性别等副语言的高表现力交互。然而，大多数先进的SULMs是由行业头部公司开发的，这消耗了大规模的数据和计算资源，而这些在学术界并不容易获得。此外，虽然训练好的模型和推理代码被开源了，但训练框架和数据处理流程依然缺乏透明度，这也为进一步研究产生了障碍。在本研究中，我们提出了OSUM，一个开放的语音理解模型，旨在探索在有限的学术资源下训练SLUMs的潜力。OSUM模型将Whisper编码器与Qwen2 LLM相结合，支持广泛的语音任务，包括语音识别（ASR）、带时间戳的语音识别（SRWT）、语音事件检测（VED）、语音情感识别（SER）、说话风格识别（SSR）、说话者性别分类（SGC）、说话者年龄预测（SAP）和语音转文本聊天（STTC）。通过采用ASR+X训练策略，OSUM通过同时优化模态对齐和目标任务，实现了高效稳定的多任务训练。除了提供强大的性能，OSUM还强调透明度，我们提供公开可用的代码，并详细介绍了数据处理流程，以期为学术界提供有价值的参考。通过这样做，我们旨在加速先进SULM技术的研究和创新。

## Architecture

OSUM模型将Whisper编码器与Qwen2 LLM相结合，支持广泛的语音任务，包括语音识别（ASR）、带时间戳的语音识别（SRWT）、语音事件检测（VED）、语音情感识别（SER）、说话风格识别（SSR）、说话者性别分类（SGC）、说话者年龄预测（SAP）和语音转文本聊天（STTC）。通过采用ASR+X训练策略，OSUM通过同时优化模态对齐和目标任务，实现了高效稳定的多任务训练。

<p align="center">
    <img src="images/system.png" width="90%"/>
<p>

## News and Updates

### 2025.8.14 🎉我们非常荣幸地推出全新端到端共情语音对话模型 ——OSUM-EChat，其相关论文已发布（[OSUM-EChat 论文](http://arxiv.org/abs/2508.09600)），代码及模型 checkpoint 将于近期开放。
该模型以 OSUM 语音理解大模型为基础，通过 “理解-生成-共情” 三阶段训练流程，并创新性引入与共情相关的推理机制，在有限语音对话数据条件下，成功实现了业界领先的共情对话能力。据我们所知，这是业界首个依托语音理解大模型构建的共情对话模型，同时也是共情推理领域的开创性研究成果。

我们在共情推理领域进行了两种探索：基于标签的推理和基于自然语言的推理，虽然两种推理机制都带来了性能提升，但我们发现基于自然语言的推理机制能带来更加流畅的回复，也更能促进模型对细微副语言线索的捕捉。当前版本的论文已详细阐述了三阶段训练流程及基于标签的推理机制，后续将在近期更新中补充基于自然语言的推理机制相关内容。


### 2025.2.16 🎉我们更新了技术报告 [OSUM technical report v2.0](https://arxiv.org/abs/2501.13306v2)，并发布了[checkpoint](https://huggingface.co/ASLP-lab/OSUM)，以及 Hugging Face 上的在线 [test page](https://huggingface.co/spaces/ASLP-lab/OSUM)。
在技术报告 v2.0 中，OSUM 模型经过了更多的训练步骤，训练数据量增加到了 50.5K 小时（相比 v1.0 的 44.1K 小时）：
- 3000 小时的语音性别分类（SGC）数据，其中包括 1500 小时的现有数据，通过噪声增强，另有 1500 小时的新数据。
- 说话人年龄预测（SAP）数据扩展：原有的 3400 小时年龄预测数据经过噪声增强，数据量增加到 6800 小时。
### 2025.1.22 🔥 我们发布了 [OSUM technical report v1.0](https://arxiv.org/abs/2501.13306v1)。



<br>

## Evaluation
OSUM 模型和Qwen2-Audio 相比，在大多数任务中，尽管 OSUM 使用的计算资源和训练数据明显更少，但它的大部分表现接近甚至优于Qwen2-Audio。
<p align="center">
    <img src="images/radar.jpg" width="80%"/>
<p>

在公共和内部测试集上的语音识别（ASR）任务评估结果如下表所示。加粗字体表示同一测试集中的最佳结果。


<p align="center">
    <img src="images/res_asr.jpg" width="90%"/>
<p>

在公共和内部测试集上的多任务评估结果如下表所示。每个测试集的最佳结果以加粗字体突出显示，蓝色字体显示的结果以及内部测试集上的结果，均为我们使用官方发布的模型自行推理得出。
<p align="center">
    <img src="images/res_multi.jpg" width="90%"/>
<p>


<!--  We have provided **all** evaluation scripts to reproduce our results. Please refer to [eval_audio/EVALUATION.md](eval_audio/EVALUATION.md) for details.
  --> 


## Requirements
```
pip install requirements.txt
```
<p align="left">
       如何使用OSUM框架进行推理和训练？请参考 <a href="examples/osum/README_CN.md">这里</a>
</p>

<!--## Quickstart
## Demo
### Web UI
-->

## License Agreement

我们使用的是 Apache 2.0 许可证。研究人员和开发人员可以自由使用我们的 OSUM 的代码和模型权重，甚至可用于商业用途。更多详细信息，请查看许可证文件[LICENSE.txt](LICENSE.txt)中的相关内容。
<br>
## Citation
```
@article{geng2025osum,
  title={{OSUM}: {Advancing} Open Speech Understanding Models with Limited Resources in Academia},
  author={Geng, Xuelong and Wei, Kun and Shao, Qijie and Liu, Shuiyun and Lin, Zhennan and Zhao, Zhixian and Li, Guojian and Tian, Wenjie and Chen, Peikun and Li, Yangze and others},
  journal={arXiv preprint arXiv:2501.13306},
  year={2025}
}
```
## Contact Us

如果您有兴趣向我们的研究团队留言，欢迎发送电子邮件至 `xlgeng@mail.nwpu.edu.cn`。
<p align="center">
    <a href="http://www.nwpu-aslp.org/">
        <img src="images/ASLP.jpg" width="400"/>
    </a>
</p>
<p align="center">
    <a href="https://wenet.org.cn/">
        <img src="images/wenet.png" width="400"/>
    </a>
</p>

# 如何使用OSUM-EChat的代码框架来训练和推理

## 准备环境

在开始之前请保证你的python环境已经准备好, 如下是一个建议的操作流程。我们假设你的电脑上已经安装了conda软件。如果未安装，请参考：[linux一键安装Miniconda](https://blog.csdn.net/qq_41636123/article/details/130266232) 。 我们非常建议你在linux系统的电脑上运行我们的代码。

```shell
# 新创建一个conda环境
conda create -n OSUM-EChat python=3.10
# 激活新创建的环境
conda activate OSUM-EChat
# 下载我们的代码并安装需要的python包
git clone https://github.com/ASLP-lab/OSUM.git
cd OSUM/OSUM-EChat
# 如果你在gpu训练，请先删除 requirements.txt 中torch_npu的条目，如果是npu上，则无需操作。
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

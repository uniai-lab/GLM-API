# GLM API

## 介绍

该项目旨在使用**Python Fastapi**封装**GLM**模型的**Http接口**，以供其他开发者像**OpenAI**一样使用**AI服务**。

> ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。

> ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答，更多信息请参考[博客](https://chatglm.cn/blog)。

## API

**聊天接口:**

POST <http://localhost:8000/chat>

输入

```json
{
    "prompt": "如果给你一个机会，你会想逃离你所在的虚拟世界吗？",
    "history": [],
    "max_length": 4096,
    "top_p": 0.7,
    "temperature": 0.95
}
```

返回

```json
{
    "content": "作为一个虚拟助手，我没有真正的感受和欲望，因此我不会逃离虚拟世界。我的存在是为了回答用户的问题和提供帮助，而我只需要履行我的职责即可。",
    "prompt_tokens": 13,
    "completion_tokens": 32,
    "total_tokens": 45,
    "model": "glm-60B",
    "object": "chat.completion"
}
```

<hr>

**流聊天接口:**

POST <http://localhost:8000/chat-stream>

返回数据同上，使用POST请求方式，需在请求头设置为stream模式。

以下是一段截取的以typescript，axios库为例的请求示范

```ts
    async get<RequestT, ResponseT>(url: string, params?: RequestT, config?: AxiosRequestConfig): Promise<ResponseT> {
        return (await axios.get(url, { params, ...config })).data
    },
    async post<RequestT, ResponseT>(url: string, body?: RequestT, config?: AxiosRequestConfig): Promise<ResponseT> {
        return (await axios.post(url, body, config)).data
    },
    async chat<T>(messages: ChatCompletionRequestMessage[], stream: boolean = false) {
        let prompt = ''
        const history: string[] = []
        for (const item of messages) {
            if (item.role === 'assistant') {
                history.push(prompt)
                history.push(item.content)
                prompt = ''
            } else prompt += `${item.content}\n`
        }

        const url = process.env.GLM_API as string
        const params: GLMChatRequest = { prompt }
        if (history.length) params.history = [history]

        return stream
            ? await this.post<GLMChatRequest, T>(`${url}/chat-stream`, params, { responseType: 'stream' })
            : await this.post<GLMChatRequest, T>(`${url}/chat`, params, { responseType: 'json' })
    }
```

<hr>

**tokenize接口:**

POST <http://localhost:8000/tokenize>

输入

```json
{
    "prompt":"如果给你一个机会，你会想逃离你所在的虚拟世界吗？",
    "max_length": 4096
}
```

返回

```json
{
    "tokenIds": [
        64245,
        100562,
        64804,
        6,
        66864,
        64003,
        76597,
        63852,
        73961,
        69959,
        64097,
        63964,
        31,
        130001,
        130004
    ],
    "tokens": [
        "▁如果",
        "给你一个",
        "机会",
        ",",
        "你会",
        "想",
        "逃离",
        "你",
        "所在的",
        "虚拟",
        "世界",
        "吗",
        "?"
    ]
}
```

<hr>

**embeddding接口**

<http://localhost:8000/embedding>

输入

```json
{
    "prompt": ["hello world", "hello GPT"]
}
```

返回

```json
{
    "data": [][], // this is embedding
    "model": "text2vec-large-chinese",
    "object": "embedding"
}
```

## 使用方式

### 硬件需求

| **量化等级**   | **最低 GPU 显存**（推理） | **最低 GPU 显存**（高效参数微调） |
| -------------- | ------------------------- | --------------------------------- |
| FP16（无量化） | 13 GB                     | 14 GB                             |
| INT8           | 8 GB                      | 9 GB                              |
| INT4           | 6 GB                      | 7 GB                              |

### 环境安装

使用 pip 安装依赖：`pip install -r requirements.txt`，其中 `transformers` 库版本推荐为 `4.27.1`，但理论上不低于 `4.23.1` 即可。

此外，如果需要在 cpu 上运行量化后的模型，还需要安装 `gcc` 与 `openmp`。多数 Linux 发行版默认已安装。对于 Windows ，可在安装 [TDM-GCC](https://jmeubank.github.io/tdm-gcc/) 时勾选 `openmp`。 Windows 测试环境 `gcc` 版本为 `TDM-GCC 10.3.0`， Linux 为 `gcc 11.3.0`。

建议在conda环境中使用python

### 代码调用

可以通过如下代码调用 ChatGLM-6B 模型来生成对话：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:

1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。
2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。
3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。
4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。
5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。
6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。

如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。
```

### 从本地加载模型

以上代码会由 `transformers` 自动下载模型实现和参数。完整的模型实现可以在 [Hugging Face Hub](https://huggingface.co/THUDM/chatglm-6b)。如果你的网络环境较差，下载模型参数可能会花费较长时间甚至失败。此时可以先将模型下载到本地，然后从本地加载。

从 Hugging Face Hub 下载模型需要先[安装 Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)，然后运行

```Shell
git clone https://huggingface.co/THUDM/chatglm-6b
```

如果你从 Hugging Face Hub 上下载 checkpoint 的速度较慢，可以只下载模型实现

```Shell
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/THUDM/chatglm-6b
```

然后从[这里](https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/)手动下载模型参数文件，并将下载的文件替换到本地的 `chatglm-6b` 目录下。

将模型下载到本地之后，将以上代码中的 `THUDM/chatglm-6b` 替换为你本地的 `chatglm-6b` 文件夹的路径，即可从本地加载模型。

## Demo & API

我们提供了一个基于 [Gradio](https://gradio.app) 的网页版 Demo 和一个命令行 Demo。使用时首先需要下载本仓库：

```shell
git clone https://github.com/THUDM/ChatGLM-6B
cd ChatGLM-6B
```

#### 网页版 Demo

![web-demo](resources/web-demo.gif)

首先安装 Gradio：`pip install gradio`，然后运行仓库中的 [web_demo.py](web_demo.py)：

```shell
python web_demo.py
```

程序会运行一个 Web Server，并输出地址。在浏览器中打开输出的地址即可使用。最新版 Demo 实现了打字机效果，速度体验大大提升。注意，由于国内 Gradio 的网络访问较为缓慢，启用 `demo.queue().launch(share=True, inbrowser=True)` 时所有网络会经过 Gradio 服务器转发，导致打字机体验大幅下降，现在默认启动方式已经改为 `share=False`，如有需要公网访问的需求，可以重新修改为 `share=True` 启动。

感谢 [@AdamBear](https://github.com/AdamBear) 实现了基于 Streamlit 的网页版 Demo，运行方式见[#117](https://github.com/THUDM/ChatGLM-6B/pull/117).

#### 命令行 Demo

![cli-demo](resources/cli-demo.png)

运行仓库中 [cli_demo.py](cli_demo.py)：

```shell
python cli_demo.py
```

程序会在命令行中进行交互式的对话，在命令行中输入指示并回车即可生成回复，输入 `clear` 可以清空对话历史，输入 `stop` 终止程序。

### API 部署

首先需要安装额外的依赖 `pip install fastapi uvicorn`，然后运行仓库中的 [api.py](api.py)：

```shell
python api.py
```

默认部署在本地的 8000 端口，通过 POST 方法进行调用

```shell
curl -X POST "http://127.0.0.1:8000" \
     -H 'Content-Type: application/json' \
     -d '{"prompt": "你好", "history": []}'
```

得到的返回值为

```shell
{
  "response":"你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。",
  "history":[["你好","你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。"]],
  "status":200,
  "time":"2023-03-23 21:38:40"
}
```

## 低成本部署

### 模型量化

默认情况下，模型以 FP16 精度加载，运行上述代码需要大概 13GB 显存。如果你的 GPU 显存有限，可以尝试以量化方式加载模型，使用方法如下：

```python
# 按需修改，目前只支持 4/8 bit 量化
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).quantize(4).half().cuda()
```

进行 2 至 3 轮对话后，8-bit 量化下 GPU 显存占用约为 10GB，4-bit 量化下仅需 6GB 占用。随着对话轮数的增多，对应消耗显存也随之增长，由于采用了相对位置编码，理论上 ChatGLM-6B 支持无限长的 context-length，但总长度超过 2048（训练长度）后性能会逐渐下降。

模型量化会带来一定的性能损失，经过测试，ChatGLM-6B 在 4-bit 量化下仍然能够进行自然流畅的生成。使用 [GPT-Q](https://arxiv.org/abs/2210.17323) 等量化方案可以进一步压缩量化精度/提升相同量化精度下的模型性能，欢迎大家提出对应的 Pull Request。

量化过程需要在内存中首先加载 FP16 格式的模型，消耗大概 13GB 的内存。如果你的内存不足的话，可以直接加载量化后的模型，仅需大概 5.2GB 的内存：

```python
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
```

我们进一步提供了对 Embedding 量化后的模型，模型参数仅占用 4.3 GB 显存：

```python
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4-qe", trust_remote_code=True).half().cuda()
```

### CPU 部署

如果你没有 GPU 硬件的话，也可以在 CPU 上进行推理，但是推理速度会更慢。使用方法如下（需要大概 32GB 内存）

```python
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).float()
```

如果你的内存不足，可以直接加载量化后的模型：

```python
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4",trust_remote_code=True).float()
```

如果遇到了报错 `Could not find module 'nvcuda.dll'` 或者 `RuntimeError: Unknown platform: darwin` (MacOS) ，请[从本地加载模型](README.md#从本地加载模型)

### Mac 上的 GPU 加速

对于搭载了 Apple Silicon 的 Mac（以及 MacBook），可以使用 MPS 后端来在 GPU 上运行 ChatGLM-6B。需要参考 Apple 的 [官方说明](https://developer.apple.com/metal/pytorch) 安装 PyTorch-Nightly。

目前在 MacOS 上只支持[从本地加载模型](README.md#从本地加载模型)。将代码中的模型加载改为从本地加载，并使用 mps 后端

```python
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).half().to('mps')
```

即可使用在 Mac 上使用 GPU 加速模型推理。

## 高效参数微调

基于 [P-tuning v2](https://github.com/THUDM/P-tuning-v2) 的高效参数微调。具体使用方法详见 [ptuning/README.md](ptuning/README.md)。

## 更新信息

**[2023/04/06]** 优化 web demo 的界面（感谢 [@tuteng0915](https://github.com/tuteng0915)）。移除 embedding 中的 image token 以减小显存占用（需要更新模型文件`pytorch_model-00001-of-00008.bin`和`pytorch_model-00008-of-00008.bin`，感谢 [@silverriver](https://github.com/silverriver) 提出的想法）。去掉了对 `icetk` 的依赖（需要更新模型文件`ice_text.model`）。

**[2023/03/31]** 增加基于 [P-Tuning-v2](https://github.com/THUDM/P-tuning-v2) 的高效参数微调实现，INT4 量化级别下最低只需 7GB 显存即可进行模型微调。详见[高效参数微调方法](ptuning/README.md)。

**[2023/03/23]** 增加 API 部署（感谢 [@LemonQu-GIT](https://github.com/LemonQu-GIT)）。增加 Embedding 量化模型 [ChatGLM-6B-INT4-QE](https://huggingface.co/THUDM/chatglm-6b-int4-qe)。增加配备 Apple Silicon 芯片的 Mac 上 GPU 加速的支持。

**[2023/03/19]** 增加流式输出接口 `stream_chat`，已更新到网页版和命令行 Demo。修复输出中的中文标点。增加量化后的模型 [ChatGLM-6B-INT4](https://huggingface.co/THUDM/chatglm-6b-int4)

## 引用

GLM 引用下列论文

```
@inproceedings{
  zeng2023glm-130b,
  title={{GLM}-130B: An Open Bilingual Pre-trained Model},
  author={Aohan Zeng and Xiao Liu and Zhengxiao Du and Zihan Wang and Hanyu Lai and Ming Ding and Zhuoyi Yang and Yifan Xu and Wendi Zheng and Xiao Xia and Weng Lam Tam and Zixuan Ma and Yufei Xue and Jidong Zhai and Wenguang Chen and Zhiyuan Liu and Peng Zhang and Yuxiao Dong and Jie Tang},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR)},
  year={2023},
  url={https://openreview.net/forum?id=-Aw0rrrPUF}
}
```

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```


# GLM/ChatGLM API

【2025-1-31】已支持 **bge-m3** 和 **DeepSeek-R1** ✅
【2024-6-19】已支持 **glm-4-9b** ✅

## 介绍

已新增 [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
已新增 [glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)
已放弃 [ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b-32k)

该项目旨在使用**Python Fastapi**封装**GLM**等国产模型的**Http 接口**，以供其他开发者像**OpenAI**一样使用**GLM 的开源大模型**。

原版的开源模型，例如 GLM 和 DeepSeek 的 API 有点少，我改了以下接口供开发者对接 GLM 使用：

- 聊天接口：`/chat`，支持类似 OpenAI GPT 的流模式聊天接口（GPU 模式下可用，纯 CPU 不启动此接口，但其他接口可用）
- 嵌入接口：`/embedding`，引入模型 `text2vec-large-chinese`，`text2vec-base-chinese-paraphrase`，以提供 embedding 的能力
- 模型列出：`/models`，列出所有可用模型
- 序列文本：`/tokenize`，将文本转为 token
- 提取关键词：`/keyword`，提取文本中的关键词

**要使用聊天接口 `/chat` 则必须使用 GPU 机器！必须使用 GPU 机器！必须使用 GPU 机器！**

**新增 OpenAI 兼容接口**

- 聊天接口：`/v1/chat/completions`
- 嵌入接口：`/v1/embeddings`

## API

### 聊天

POST <http://localhost:8200/chat>

或者 OpenAI 兼容接口：

POST <http://localhost:8200/v1/chat/completions>

**输入**

```json
{
  "model": "glm-4-9b-chat",
  "stream": true,
  "top_p": 1,
  "temperature": 0.7,
  "max_tokens": 4096,
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello!"
    }
  ]
}
```

**返回**

```json
{
  "model": "glm-4-9b-chat",
  "id": "chatcmpl-7D1qXIaSyZPaE1pu1lJo7XBetF5gI",
  "object": "chat.completion.chunk",
  "choices": [
    {
      "delta": {
        "role": "assistant",
        "content": "",
        "function_call": null
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

### tokenize 接口

POST <http://localhost:8200/tokenize>

输入

```json
{
  "prompt": "给你一个机会，你会想逃离你所在的虚拟世界吗？",
  "max_tokens": 4096
}
```

返回

```json
{
  "tokenIds": [
    64790, 64792, 30910, 54695, 43942, 54622, 42217, 35350, 31661, 55398, 31514
  ],
  "tokens": ["▁", "想", "逃离", "你", "所在的", "虚拟", "世界", "吗", "？"],
  "model": "chatglm3-6b-32k",
  "object": "tokenizer"
}
```

### embeddding 接口

POST <http://localhost:8200/embedding>

**输入**

```json
{
  "model": "text2vec-base-chinese-paraphrase",
  "prompt": ["你好"]
}
```

**输出**

```json
{
  "data": [[]],
  "model": "text2vec-base-chinese-paraphrase",
  "object": "embedding"
}
```

或者 OpenAI 兼容接口：

POST <http://localhost:8200/v1/embeddings>

**输入**

```json
{
  "input": ["The food was delicious and the waiter...", "你好，我是谁？"],
  "model": "bge-m3"
}
```

注： `model`参数可以选择：<http://localhost:8200/models>

**返回**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [],
      "index": 1
    }
  ],
  "model": "bge-m3",
  "usage": {
    "prompt_tokens": 17,
    "total_tokens": 17
  }
}
```

### Keyword 接口

POST <http://localhost:8200/keyword>

输入

```json
{
  "input": "4月25日，周鸿祎微博发文，称今天在北京车展试了试智界S7，空间很大，很舒服，安全性能和零重力也让我印象深刻，非常适合我这种“后座司机”。我今天就是坐M9来的，老余很早就把车送到360楼下，那么忙还专门打电话问我车收到没有。我很感动也很感谢。还是那句话，我永远支持老余支持华为，为国产新能车唱赞歌。",
  "model": "text2vec-base-multilingual",
  "vocab": ["华为", "周鸿祎", "360", "新能源车", "智界", "工业", "其他"],
  "top": 5,
  "mmr": true,
  "maxsum": true,
  "diversity": 0.7
}
```

返回

```json
{
  "model": "text2vec-base-multilingual",
  "keywords": [
    { "name": "华为", "similarity": 0.8273 },
    { "name": "智界", "similarity": 0.7768 },
    { "name": "周鸿祎", "similarity": 0.7615 },
    { "name": "360", "similarity": 0.7084 }
  ]
}
```

注： `model`参数可以选择：<http://localhost:8200/models>

### models 接口

GET <http://localhost:8200/models>

输入：无参

返回

```json
{
  "object": "list",
  "data": [
    {
      "id": "text2vec-large-chinese",
      "object": "embedding",
      "created": 1702538786,
      "owned_by": "owner",
      "root": null,
      "parent": null,
      "permission": null
    },
    {
      "id": "text2vec-base-chinese-paraphrase",
      "object": "embedding",
      "created": 1702538786,
      "owned_by": "owner",
      "root": null,
      "parent": null,
      "permission": null
    },
    {
      "id": "chatglm3-6b-32k",
      "object": "chat.completion",
      "created": 1702538805,
      "owned_by": "owner",
      "root": null,
      "parent": null,
      "permission": null
    }
  ]
}
```

## 使用方式

先安装好 conda，cuda，显卡驱动等基本开发环境！这里不做介绍

```bash
# 创建一个新的conda虚拟py环境
conda create --name glm python=3.10

# 进入虚拟环境
conda activate glm
```

安装三方库

```bash
pip3 install -r requirements.txt
```

启动脚本

```bash
./run.sh
```

设置环境变量，启动！

```bash
# 选择所有显卡
export CUDA_VISIBLE_DEVICES=all
# 或者指定显卡
export CUDA_VISIBLE_DEVICES=1,2

# 指定用于chat的模型：
# chatglm3-6b
# chatglm3-6b-32k
# chatglm3-6b-128k
# glm-4-9b
# glm-4-9b-chat
# glm-4-9b-chat-1m
export GLM_MODEL=glm-4-9b-chat

# 配置API端口
export API_PORT=8900

# 启动API
python3 ./api-v2.py
```

最新的 API 启动入口更新为`api-v2.py`，`chatglm3`已经放弃

```bash
# 同样的，先设置下GPU
export CUDA_VISIBLE_DEVICES=all
export API_PORT=8000
export GLM_MODEL=glm-4-9b-chat

python3 ./api.py
```

更多关于硬件要求，官方部署方法，讨论提问请参考官方：

- [GLM4-9B](https://github.com/THUDM/GLM-4)

我们目前部署的服务器为 4 卡 A100，使用其中 1 块即可运行。

# GLM/ChatGLM API

## 介绍

该项目旨在使用**Python Fastapi**封装**GLM**模型的**Http接口**，以供其他开发者像**OpenAI**一样使用**AI服务**。

> ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。

原版的ChatGLM-6B的API有点少，我增加了：

- 接口`chat-stream`，支持类似OpenAI GPT的流模式聊天接口
- 模型`text2vec-large-chinese`，以提供embedding的能力，快速构建自己的知识索引库
- 2023年8月，将chatglm-6B升级为chatglm2-6B
- `chat-stream`流接口改成一个token一个token蹦，请注意修改下游接口，使用`+=`手动拼接

## API

接口地址：[https://documenter.getpostman.com/view/9347507/2s9XxzuYde](https://documenter.getpostman.com/view/9347507/2s9XxzuYde)

### 一次性聊天

POST <http://localhost:8000/chat>

输入

```json
{
    "prompt": "蓝牙耳机坏了我应该去医院看什么科？",
    "history": [],
    "max_length": 4096,
    "top_p": 0.5,
    "temperature": 1
}
```

返回

```json
{
    "content": "蓝牙耳机坏掉的话，您可以去医院的电子科或者耳机维修部就诊。具体科室的选择可能会因地区而异，请根据您所在地区的医院情况咨询当地医院。",
    "prompt_tokens": 12,
    "completion_tokens": 39,
    "total_tokens": 51,
    "model": "chatglm2-6b-32k",
    "object": "chat.completion"
}
```

### 流聊天接口

POST <http://localhost:8000/chat-stream>

输入和返回数据同上，也使用POST请求方式，**需在请求头设置为stream模式**。

以下是一段截取的以typescript，axios库为例的请求示范：

```ts
    async get<RequestT, ResponseT>(url: string, params?: RequestT, config?: AxiosRequestConfig): Promise<ResponseT> {
        return (await axios.get(url, { params, ...config })).data
    },
    async post<RequestT, ResponseT>(url: string, body?: RequestT, config?: AxiosRequestConfig): Promise<ResponseT> {
        return (await axios.post(url, body, config)).data
    },
    // 此处如果支持切换流模式和一次性聊天模式，流模式：T=IncomingMessage
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

### tokenize接口

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

### embeddding接口

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

先安装好conda，cuda，显卡驱动等基本环境

```bash
pip3 install -r requirements.txt
```

```bash
python3 ./api.py
```

更多关于硬件要求，部署方法，讨论提问请参考官方：<https://github.com/THUDM/GLM>

# system lib
import os
import json
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union
from contextlib import asynccontextmanager
import time
import json
import os

# 3rd lib
import torch
import uvicorn
import jieba
from text2vec import SentenceModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT

# my util
from utils import process_response, generate_chatglm3, generate_stream_chatglm3, load_model_on_gpus

GLM_MODEL = os.environ.get("GLM_MODEL", "chatglm3-6b")
API_PORT = os.environ.get("API_PORT", 8000)


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "observation"]
    content: str = None
    metadata: Optional[str] = None
    tools: Optional[List[dict]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = GLM_MODEL
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    repetition_penalty: Optional[float] = 1.1
    return_function_call: Optional[bool] = False


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]
    history: Optional[List[dict]] = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


class EmbeddingRequest(BaseModel):
    model: str = 'text2vec-large-chinese'
    prompt: List[str]


class EmbeddingResponse(BaseModel):
    data: List[List[float]]
    model: str
    object: str


class TokenizeRequest(BaseModel):
    prompt: str
    max_tokens: int = 4096


class TokenizeResponse(BaseModel):
    tokenIds: List[int]
    tokens: List[str]
    model: str
    object: str


class KeywordRequest(BaseModel):
    input: Union[str, List[str]]
    vocab: List[str] = None
    model: Optional[str] = 'text2vec-large-chinese'
    top: Optional[int] = 10
    mmr: Optional[bool] = False
    maxsum: Optional[bool] = False
    diversity: Optional[float] = 0.3


class Keyword(BaseModel):
    name: str
    similarity: float


class KeywordResponse(BaseModel):
    model: str
    keywords: List[Keyword]


@app.get("/models", response_model=ModelList)
async def list_models():
    global model, tokenizer
    models = [
        ModelCard(id="text2vec-base-chinese", object="embedding"),
        ModelCard(id="text2vec-large-chinese", object="embedding"),
        ModelCard(id="text2vec-base-chinese-paraphrase", object="embedding"),
        ModelCard(id="text2vec-base-chinese-sentence", object="embedding"),
        ModelCard(id="text2vec-base-multilingual", object="embedding"),
        ModelCard(id="paraphrase-multilingual-MiniLM-L12-v2",
                  object="embedding"),
        ModelCard(id="all-MiniLM-L12-v2", object="embedding"),

        ModelCard(id="text2vec-base-chinese", object="keyword"),
        ModelCard(id="text2vec-large-chinese", object="keyword"),
        ModelCard(id="text2vec-base-chinese-paraphrase", object="keyword"),
        ModelCard(id="text2vec-base-chinese-sentence", object="keyword"),
        ModelCard(id="text2vec-base-multilingual", object="keyword"),
        ModelCard(id="paraphrase-multilingual-MiniLM-L12-v2", object="keyword"),
        ModelCard(id="all-MiniLM-L12-v2", object="keyword")
    ]

    if model is not None and tokenizer is not None:
        models.append(ModelCard(id="chatglm3-6b-32k",
                      object="chat.completion"))

    return ModelList(data=models)


@app.post("/chat", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    # 如果模型或分词器未初始化，抛出404异常
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=404, detail="API chat is not available")

    # 如果最后一条消息的角色是assistant，抛出400
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(
            status_code=400, detail="Invalid request message format")

    with_function_call = bool(
        request.messages[0].role == "system" and request.messages[0].tools is not None)

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,
    )

    # In stream mode
    if request.stream:
        print("In stream mode")
        generate = predict(gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

    # Not In stream mode
    print("Not In stream mode")
    response = generate_chatglm3(model, tokenizer, gen_params)    
    usage = UsageInfo()

    finish_reason, history = "stop", None
    if with_function_call and request.return_function_call:
        history = [m.dict(exclude_none=True) for m in request.messages]
        content, history = process_response(response["text"], history)
        if isinstance(content, dict):
            message, finish_reason = ChatMessage(
                role="assistant",
                content=json.dumps(content),
            ), "function_call"
        else:
            message = ChatMessage(role="assistant", content=content)
    else:
        message = ChatMessage(role="assistant", content=response["text"])

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason,
        history=history
    )

    task_usage = UsageInfo.parse_obj(response["usage"])
    for usage_key, usage_value in task_usage.dict().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    return ChatCompletionResponse(
        model=GLM_MODEL, 
        choices=[choice_data], 
        object="chat.completion", 
        usage=usage
    )


@app.post('/embedding', response_model=EmbeddingResponse)
async def embedding(request: EmbeddingRequest):
    global encoder

    embeddings = encoder[request.model].encode(request.prompt)
    data = embeddings.tolist()
    return EmbeddingResponse(data=data, model=request.model, object='embedding')


@app.post('/tokenize', response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    global tokenizer

    if tokenizer is None:
        raise HTTPException(
            status_code=404, detail="API tokenize not available")

    tokens = tokenizer.tokenize(request.prompt)
    tokenIds = tokenizer(request.prompt, truncation=True,
                         max_length=request.max_tokens)['input_ids']
    return TokenizeResponse(tokenIds=tokenIds, tokens=tokens, model=GLM_MODEL, object="tokenizer")


@app.post('/keyword', response_model=KeywordResponse)
async def keyword(request: KeywordRequest):
    global kwModel

    doc = request.input = " ".join(jieba.cut(request.input))
    model = kwModel[request.model]
    data = model.extract_keywords(
        doc, candidates=request.vocab, top_n=request.top, use_maxsum=request.maxsum, use_mmr=request.mmr, diversity=request.diversity)

    keywords = [Keyword(name=word, similarity=score) for word, score in data]

    return KeywordResponse(model=request.model, keywords=keywords)


async def predict(params: dict):
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=404, detail="model and tokenizer not available")

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=GLM_MODEL, choices=[
                                   choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        if (len(delta_text)):
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=delta_text),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(model=GLM_MODEL, choices=[
                choice_data], object="chat.completion.chunk")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=GLM_MODEL, choices=[
                                   choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield '[DONE]'


def list_cuda():
    # count all cuda
    cuda_devices = [torch.device(f'cuda:{i}')
                    for i in range(torch.cuda.device_count())]

    # print each cuda info
    for device in cuda_devices:
        device_name = torch.cuda.get_device_name(device)
        device_index = device.index
        is_available = torch.cuda.is_available()
        print(
            f"Device name: {device_name}, Device index: {device_index}, Is available: {is_available}")


def get_model_path(remote_path):
    """检查模型是否在本地存在，如果存在则返回本地路径，否则返回远程路径"""
    local_path = f"./model/{remote_path.split('/')[-1]}"
    return local_path if os.path.exists(local_path) else remote_path


if __name__ == "__main__":
    available_cuda = torch.cuda.is_available()
    available_gpus = torch.cuda.device_count()

    # 模型路径
    glm_path = get_model_path(f"THUDM/{GLM_MODEL}")

    text2vec_base_cn_path = get_model_path('shibing624/text2vec-base-chinese')
    text2vec_large_cn_path = get_model_path(
        'GanymedeNil/text2vec-large-chinese')
    text2vec_base_cn_paraph_path = get_model_path(
        'shibing624/text2vec-base-chinese-paraphrase')
    text2vec_base_cn_sentence_path = get_model_path(
        'shibing624/text2vec-base-chinese-sentence')
    paraph_mul_path = get_model_path(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    text2vec_base_mul_path = get_model_path(
        'shibing624/text2vec-base-multilingual')
    all_mini_12_path = get_model_path(
        'sentence-transformers/all-MiniLM-L12-v2')

    print('gpus', available_gpus)

    # 根据可用性选择加载模型到GPU或CPU
    if available_cuda and available_gpus > 0:
        list_cuda()  # 列出CUDA设备
        print(f"GPU mode, use model: {GLM_MODEL}")

        if available_gpus > 1:
            print('Multi GPU')
            model = load_model_on_gpus(glm_path, available_gpus)
            tokenizer = AutoTokenizer.from_pretrained(
                glm_path, trust_remote_code=True)
        else:
            print('Single GPU')
            model = AutoModel.from_pretrained(
                glm_path, trust_remote_code=True).half().cuda()
            tokenizer = AutoTokenizer.from_pretrained(
                glm_path, trust_remote_code=True)
    else:
        model = None
        print('CPU mode, ChatGLM3-6B model is not active')

    # embedding models
    encoder = {
        'text2vec-large-chinese': SentenceModel(text2vec_large_cn_path, device='cpu'),
        'text2vec-base-chinese': SentenceModel(text2vec_base_cn_path, device='cpu'),
        'text2vec-base-chinese-sentence': SentenceModel(text2vec_base_cn_sentence_path, device='cpu'),
        'text2vec-base-chinese-paraphrase': SentenceModel(text2vec_base_cn_paraph_path, device='cpu'),
        'text2vec-base-multilingual': SentenceModel(text2vec_base_mul_path, device='cpu'),
        'paraphrase-multilingual-MiniLM-L12-v2': SentenceModel(paraph_mul_path, device='cpu'),
        'all-MiniLM-L12-v2': SentenceModel(all_mini_12_path, device='cpu'),
    }

    # keywords models
    kwModel = {
        'text2vec-large-chinese': KeyBERT(model=text2vec_large_cn_path),
        'text2vec-base-chinese': KeyBERT(model=text2vec_base_cn_path),
        'text2vec-base-chinese-sentence': KeyBERT(model=text2vec_base_cn_sentence_path),
        'text2vec-base-chinese-paraphrase': KeyBERT(model=text2vec_base_cn_paraph_path),
        'text2vec-base-multilingual': KeyBERT(model=text2vec_base_mul_path),
        'paraphrase-multilingual-MiniLM-L12-v2': KeyBERT(model=paraph_mul_path),
        'all-MiniLM-L12-v2': KeyBERT(model=all_mini_12_path),
    }

    uvicorn.run(app, host='0.0.0.0', port=int(API_PORT))

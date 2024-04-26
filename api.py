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

MODEL = 'chatglm3-6b-32k'


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
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 4096
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    chunk: Optional[bool] = True

    # Additional parameters support for stop generation
    stop_token_ids: Optional[List[int]] = None
    repetition_penalty: Optional[float] = 1.1

    # Additional parameters supported by tools
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
    top_n: Optional[int] = 5


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
        ModelCard(id="paraphrase-multilingual-MiniLM-L12-v2",
                  object="embedding"),
        ModelCard(id="text2vec-base-chinese", object="keyword"),
        ModelCard(id="text2vec-large-chinese", object="keyword"),
        ModelCard(id="text2vec-base-chinese-paraphrase", object="keyword"),
        ModelCard(id="text2vec-base-chinese-sentence", object="keyword"),
        ModelCard(id="paraphrase-multilingual-MiniLM-L12-v2", object="keyword")
    ]

    if model is not None and tokenizer is not None:
        models.append(ModelCard(id="chatglm3-6b-32k",
                      object="chat.completion"))

    return ModelList(data=models)


@app.post("/chat", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(status_code=404, detail="chat API not available")

    if request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    with_function_call = bool(
        request.messages[0].role == "system" and request.messages[0].tools is not None)

    # stop settings
    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]

    request.stop_token_ids = request.stop_token_ids or []

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
        chunk=request.chunk,
        stop_token_ids=request.stop_token_ids,
        stop=request.stop,
        repetition_penalty=request.repetition_penalty,
        with_function_call=with_function_call,
    )

    if request.stream:
        generate = predict(MODEL, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")

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

    return ChatCompletionResponse(model=MODEL, choices=[choice_data], object="chat.completion", usage=usage)


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
    return TokenizeResponse(tokenIds=tokenIds, tokens=tokens, model=MODEL, object="tokenizer")


@app.post('/keyword', response_model=KeywordResponse)
async def keyword(request: KeywordRequest):
    global kwModel

    doc = request.input = " ".join(jieba.cut(request.input))
    model = kwModel[request.model]
    data = model.extract_keywords(
        doc, candidates=request.vocab, top_n=request.top_n)

    keywords = [Keyword(name=word, similarity=score) for word, score in data]

    return KeywordResponse(model=request.model, keywords=keywords)


async def predict(model_id: str, params: dict):
    global model, tokenizer

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=404, detail="model and tokenizer not available")

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
                                   choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_chatglm3(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        if params["chunk"]:
            delta_text = decoded_unicode[len(previous_text):]
            previous_text = decoded_unicode
        else:
            delta_text = decoded_unicode

        if (len(delta_text)):
            choice_data = ChatCompletionResponseStreamChoice(
                index=0,
                delta=DeltaMessage(content=delta_text),
                finish_reason=None
            )
            chunk = ChatCompletionResponse(model=model_id, choices=[
                choice_data], object="chat.completion.chunk")
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
        finish_reason="stop"
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[
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


def list_cuda():
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    # Check if CUDA is available before proceeding
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(
            f"Device name: {device_name}, Device index: {i}, Is available: True")


def get_model_path(remote_path):
    """检查模型是否在本地存在，如果存在则返回本地路径，否则返回远程路径"""
    local_path = f"./model/{remote_path.split('/')[-1]}"
    return local_path if os.path.exists(local_path) else remote_path


if __name__ == "__main__":
    available_cuda = torch.cuda.is_available()
    available_gpus = torch.cuda.device_count()

    # 模型路径
    chatglm_large_path = get_model_path('THUDM/chatglm3-6b-32k')
    chatglm_path = get_model_path('THUDM/chatglm3-6b')
    text2vec_base_path = get_model_path('shibing624/text2vec-base-chinese')
    text2vec_large_path = get_model_path('GanymedeNil/text2vec-large-chinese')
    text2vec_paraph_path = get_model_path(
        'shibing624/text2vec-base-chinese-paraphrase')
    text2vec_sentence_path = get_model_path(
        'shibing624/text2vec-base-chinese-sentence')
    paraph_mul_path = get_model_path(
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    print('gpus', available_gpus)

    # 根据可用性选择加载模型到GPU或CPU
    if available_cuda and available_gpus > 0:
        print('GPU mode')
        list_cuda()  # 列出CUDA设备
        if available_gpus > 1:
            model = load_model_on_gpus(chatglm_large_path, available_gpus)
            tokenizer = AutoTokenizer.from_pretrained(
                chatglm_large_path, trust_remote_code=True)
        else:
            model = AutoModel.from_pretrained(
                chatglm_path, trust_remote_code=True).cuda()
            tokenizer = AutoTokenizer.from_pretrained(
                chatglm_path, trust_remote_code=True)
    else:
        print('CPU mode, ChatGLM-6B model is not active')

    # embedding models
    encoder = {
        'text2vec-large-chinese': SentenceModel(text2vec_large_path, device='cpu'),
        'text2vec-base-chinese': SentenceModel(text2vec_base_path, device='cpu'),
        'text2vec-base-chinese-sentence': SentenceModel(text2vec_sentence_path, device='cpu'),
        'text2vec-base-chinese-paraphrase': SentenceModel(text2vec_paraph_path, device='cpu'),
        'paraphrase-multilingual-MiniLM-L12-v2': SentenceModel(paraph_mul_path, device='cpu'),
    }

    # keywords models
    kwModel = {
        'text2vec-large-chinese': KeyBERT(model=text2vec_large_path),
        'text2vec-base-chinese': KeyBERT(model=text2vec_base_path),
        'text2vec-base-chinese-sentence': KeyBERT(model=text2vec_sentence_path),
        'text2vec-base-chinese-paraphrase': KeyBERT(model=text2vec_paraph_path),
        'paraphrase-multilingual-MiniLM-L12-v2': KeyBERT(model=paraph_mul_path),
    }

    uvicorn.run(app, host='0.0.0.0', port=8200)

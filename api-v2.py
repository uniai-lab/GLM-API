# system lib
from vllm import AsyncEngineArgs, AsyncLLMEngine
from transformers import AutoTokenizer
from keybert import KeyBERT
import os
from contextlib import asynccontextmanager

# 3rd lib
from classType import *
from glm4 import (
    generate_id,
    generate_stream_glm4,
    parse_output_text_glm4,
    predict_glm4,
    process_response_glm4,
)
import torch
import uvicorn
import jieba
from loguru import logger
from text2vec import SentenceModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

EventSourceResponse.DEFAULT_PING_INTERVAL = 1000


# set chat model in env as GLM_MODEL=chatglm3-6b
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-4-9b-chat")
API_PORT = os.environ.get("API_PORT", 8000)
# 动态设置 MAX_MODEL_LENGTH
if GLM_MODEL == "chatglm3-6b":
    MAX_MODEL_LENGTH = 8 * 1024     # 8192
else:
    MAX_MODEL_LENGTH = 128 * 1024   # 131072


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
        ModelCard(id="all-MiniLM-L12-v2", object="keyword"),
    ]

    # Add chat model
    if model is not None and tokenizer is not None:
        models.append(ModelCard(id=GLM_MODEL, object="chat.completion"))

    return ModelList(data=models)


@app.post("/embedding", response_model=EmbeddingResponse)
async def embedding(request: EmbeddingRequest):
    global encoder
    if request.model not in encoder:
        raise HTTPException(
            status_code=400, detail="Embedding model not found")

    embeddings = encoder[request.model].encode(request.prompt)
    data = embeddings.tolist()
    return EmbeddingResponse(data=data, model=request.model, object="embedding")


@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    global tokenizer

    if tokenizer is None:
        raise HTTPException(
            status_code=404, detail="API tokenize not available")

    tokens = tokenizer.tokenize(request.prompt)
    tokenIds = tokenizer(
        request.prompt, truncation=True, max_length=request.max_tokens
    )["input_ids"]
    return TokenizeResponse(
        tokenIds=tokenIds, tokens=tokens, model=GLM_MODEL, object="tokenizer"
    )


@app.post("/chat", response_model=ChatCompletionResponse)
async def chat(request: ChatCompletionRequest):
    global model, tokenizer

    # 如果模型或分词器未初始化，抛出404异常
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=404, detail="API chat is not available")

    # 如果最后一条消息的角色是assistant，抛出400
    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(
            status_code=400, detail="Invalid request message format")

    # 生成参数字典，用于传递给生成函数
    gen_params = dict(
        model=GLM_MODEL,
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        echo=False,
        stream=request.stream,
        repetition_penalty=request.repetition_penalty,

        # -----------------------------------------------------------------------------------------
        # tools=request.tools,
        # 修改：确保 tools 参数总是存在，即使 request.tools 为 None（避免迭代 NoneType）
        tools=request.tools if request.tools else [],
        # -----------------------------------------------------------------------------------------

        tool_choice=request.tool_choice,
    )

    # In stream mode
    if request.stream:
        print("In stream mode")
        predict_stream_generator = predict_glm4(
            GLM_MODEL, gen_params, model, tokenizer)
        output = await anext(predict_stream_generator)
        if output:
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")
        logger.debug(f"First result output：\n{output}")

        function_call = None
        if output and request.tools:
            try:
                function_call = process_response_glm4(
                    output, request.tools, use_tool=True)
            except:
                logger.warning("Failed to parse tool call")

        if isinstance(function_call, dict):
            function_call = FunctionCall(**function_call)
            generate = parse_output_text_glm4(
                GLM_MODEL, output, function_call=function_call)
            return EventSourceResponse(generate, media_type="text/event-stream")
        else:
            return EventSourceResponse(predict_stream_generator, media_type="text/event-stream")

    # Not In stream mode
    print("Not In stream mode")
    response = ""
    # 异步生成响应
    async for response in generate_stream_glm4(gen_params, model, tokenizer):
        pass

    # 去除响应文本开头的换行符，并去除首位空格
    if response["text"].startswith("\n"):
        response["text"] = response["text"][1:]
    response["text"] = response["text"].strip()

    print("######### ----------------------------------------------------------")
    print("Final Response:\n", response["text"])
    print("######### ----------------------------------------------------------\n\n\n")

    # 初始化使用情况信息
    usage = UsageInfo()

    # 处理工具调用和完成原因
    function_call, finish_reason = None, "stop"
    tool_calls = None
    if request.tools:
        try:
            function_call = process_response_glm4(
                response["text"], request.tools, use_tool=True)
        except Exception as e:
            logger.warning(f"Failed to parse tool call: {e}")
    if isinstance(function_call, dict):
        finish_reason = "tool_calls"
        function_call_response = FunctionCall(**function_call)
        function_call_instance = FunctionCall(
            name=function_call_response.name,
            arguments=function_call_response.arguments
        )
        tool_calls = [
            ChatCompletionMessageToolCall(
                id=generate_id('call_', 24),
                function=function_call_instance,
                type="function")]

    # 创建消息对象，包含助手角色的响应内容或工具调用信息
    message = ChatMessage(
        role="assistant",
        content=None if tool_calls else response["text"],
        function_call=None,
        tool_calls=tool_calls,
    )

    # 创建响应的选择对象，包含消息和完成原因
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
        finish_reason=finish_reason
    )

    # 更新使用情况信息
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)

    # 最后返回最终的聊天完成响应
    return ChatCompletionResponse(
        model=GLM_MODEL,
        choices=[choice_data],
        id=generate_id('chatcmpl-', 29),
        object="chat.completion",
        usage=usage
    )


@app.post("/keyword", response_model=KeywordResponse)
async def keyword(request: KeywordRequest):
    global kwModel
    if request.model not in kwModel:
        raise HTTPException(status_code=400, detail="Keyword model not found")

    doc = request.input = " ".join(jieba.cut(request.input))
    model = kwModel[request.model]
    data = model.extract_keywords(
        doc,
        candidates=request.vocab,
        top_n=request.top,
        use_maxsum=request.maxsum,
        use_mmr=request.mmr,
        diversity=request.diversity,
    )

    keywords = [Keyword(name=word, similarity=score) for word, score in data]

    return KeywordResponse(model=request.model, keywords=keywords)


def list_cuda():
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    # Check if CUDA is available before proceeding
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        print(
            f"Device name: {device_name}, Device index: {i}, Is available: True")


def get_model_path(remote_path):
    local_path = f"./model/{remote_path.split('/')[-1]}"
    return local_path if os.path.exists(local_path) else remote_path


if __name__ == "__main__":
    # 检查CUDA是否可用
    available_cuda = torch.cuda.is_available()
    available_gpus = torch.cuda.device_count()
    print("Number of gpu: ", available_gpus)

    # 模型路径
    glm_path = get_model_path(f"THUDM/{GLM_MODEL}")

    # 嵌入模型
    text2vec_base_cn_path = get_model_path("shibing624/text2vec-base-chinese")
    text2vec_large_cn_path = get_model_path(
        "GanymedeNil/text2vec-large-chinese")
    text2vec_base_cn_paraph_path = get_model_path(
        "shibing624/text2vec-base-chinese-paraphrase")
    text2vec_base_cn_sentence_path = get_model_path(
        "shibing624/text2vec-base-chinese-sentence")
    paraph_mul_path = get_model_path(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    text2vec_base_mul_path = get_model_path(
        "shibing624/text2vec-base-multilingual")
    all_mini_12_path = get_model_path(
        "sentence-transformers/all-MiniLM-L12-v2")

    # 根据可用性选择加载模型到GPU或CPU
    # 如果有可用的GPU
    if available_cuda and available_gpus > 0:
        list_cuda()  # 列出CUDA设备

        # 新模型：glm-4-9b-chat
        print(f"GPU mode, use model: {GLM_MODEL}")
        print(f"Max Model Length: {MAX_MODEL_LENGTH}")
        tokenizer = AutoTokenizer.from_pretrained(
            glm_path, trust_remote_code=True)

        engine_args = AsyncEngineArgs(
            model=glm_path,
            tokenizer=glm_path,
            # 如果你有多张显卡，可以在这里设置成你的显卡数量
            tensor_parallel_size=available_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            # 占用显存的比例，请根据你的显卡显存大小设置合适的值，例如，如果你的显卡有80G，您只想使用24G，请按照24/80=0.3设置
            # gpu_memory_utilization=0.3,
            enforce_eager=True,
            worker_use_ray=False,
            engine_use_ray=False,
            disable_log_requests=True,
            max_model_len=MAX_MODEL_LENGTH,
        )
        model = AsyncLLMEngine.from_engine_args(engine_args)

    # 如果没有GPU，则不加载ChatGLM模型，仅打印出CPU模式的提示
    else:
        model = None
        print("CPU mode, Chat model is not active")

    # embedding models
    # 嵌入模型（文本数据 -> 数值向量）（便于模型处理）
    encoder = {
        "text2vec-large-chinese": SentenceModel(text2vec_large_cn_path, device="cpu"),
        "text2vec-base-chinese": SentenceModel(text2vec_base_cn_path, device="cpu"),
        "text2vec-base-chinese-sentence": SentenceModel(text2vec_base_cn_sentence_path, device="cpu"),
        "text2vec-base-chinese-paraphrase": SentenceModel(text2vec_base_cn_paraph_path, device="cpu"),
        "text2vec-base-multilingual": SentenceModel(text2vec_base_mul_path, device="cpu"),
        "paraphrase-multilingual-MiniLM-L12-v2": SentenceModel(paraph_mul_path, device="cpu"),
        "all-MiniLM-L12-v2": SentenceModel(all_mini_12_path, device="cpu"),
    }

    # keywords models
    # 关键词提取模型（从文本中提取最重要和最具代表性的词或短语）（便于进行文本摘要和信息检索）
    kwModel = {
        "text2vec-large-chinese": KeyBERT(model=text2vec_large_cn_path),
        "text2vec-base-chinese": KeyBERT(model=text2vec_base_cn_path),
        "text2vec-base-chinese-sentence": KeyBERT(model=text2vec_base_cn_sentence_path),
        "text2vec-base-chinese-paraphrase": KeyBERT(model=text2vec_base_cn_paraph_path),
        "text2vec-base-multilingual": KeyBERT(model=text2vec_base_mul_path),
        "paraphrase-multilingual-MiniLM-L12-v2": KeyBERT(model=paraph_mul_path),
        "all-MiniLM-L12-v2": KeyBERT(model=all_mini_12_path),
    }

    # 启动Uvicorn服务器
    uvicorn.run(app, host="0.0.0.0", port=int(API_PORT))

import uvicorn
import json
import torch

from text2vec import SentenceModel
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse


MAX_LENGTH = 4096
TOP_P = 0.7
TEMPERATURE = 0.95


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        with torch.cuda.device('cuda:1'):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


def predict(tokenizer, prompt, history, max_length, top_p, temperature):
    for response, history in model.stream_chat(tokenizer, prompt, history, max_length=max_length, top_p=top_p,
                                               temperature=temperature):
        yield json.dumps({
            'content': response,
            'prompt_tokens': count(prompt),
            'completion_tokens': count(response),
            'total_tokens': count(prompt)+count(response),
            'model': "glm-60B",
            'object': 'chat.completion'
        })
    return torch_gc()


def count(prompt):
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)


@app.post('/chat')
async def chat(request: Request):
    global model, tokenizer

    json_post_raw = await request.json()
    data = json.loads(json.dumps(json_post_raw))

    prompt = data.get('prompt', '')
    history = data.get('history', [])
    max_length = data.get('max_length', MAX_LENGTH)
    top_p = data.get('top_p', TOP_P)
    temperature = data.get('temperature', TEMPERATURE)

    response, history = model.chat(
        tokenizer, prompt, history=history, max_length=max_length, top_p=top_p, temperature=temperature)
    torch_gc()
    data = {
        'content': response,
        'prompt_tokens': count(prompt),
        'completion_tokens': count(response),
        'total_tokens': count(response)+count(prompt),
        'model': "glm-60B",
        'object': 'chat.completion'
    }
    return data


@app.post('/chat-stream')
async def chat_stream(request: Request):
    global model, tokenizer

    json_post_raw = await request.json()
    data = json.loads(json.dumps(json_post_raw))

    prompt = data.get('prompt', '')
    history = data.get('history', [])
    max_length = data.get('max_length', MAX_LENGTH)
    top_p = data.get('top_p', TOP_P)
    temperature = data.get('temperature', TEMPERATURE)

    res = predict(tokenizer, prompt, history, max_length, top_p, temperature)
    return EventSourceResponse(res)


@app.post('/embedding')
async def embedding(request: Request):
    global encoder

    json_post_raw = await request.json()
    data = json.loads(json.dumps(json_post_raw))
    prompt = data.get('prompt', [])
    embeddings = encoder.encode(prompt)
    data = embeddings.tolist()
    return {'data': data, 'model': 'text2vec-large-chinese', 'object': 'embedding'}


@ app.post('/tokenize')
async def tokenize(request: Request):
    global model, tokenizer

    json_post_raw = await request.json()
    data = json.loads(json.dumps(json_post_raw))

    prompt = data.get('prompt', '')
    max_length = data.get('max_length', MAX_LENGTH)

    tokens = tokenizer.tokenize(prompt)
    tokenIds = tokenizer(prompt, truncation=True,
                         max_length=max_length)['input_ids']
    return {'tokenIds': tokenIds, 'tokens': tokens}


if __name__ == '__main__':
    # load GLM 6B
    tokenizer = AutoTokenizer.from_pretrained(
        'THUDM/chatglm-6b', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'THUDM/chatglm-6b', trust_remote_code=True).half().cuda('cuda:0')
    model.eval()

    # load embedding model
    encoder = SentenceModel('GanymedeNil/text2vec-large-chinese')
    # start fastapi
    uvicorn.run(app, host='0.0.0.0', port=8000)

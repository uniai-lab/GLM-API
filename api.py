import uvicorn
import json
import torch

from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

DEVICE = 'cuda'
DEVICE_ID = '0'
CUDA_DEVICE = f'{DEVICE}:{DEVICE_ID}' if DEVICE_ID else DEVICE
MAX_LENGTH = 4096
TOP_P = 0.7
TEMPERATURE = 0.95


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
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
            'message': response,
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
        'message': response,
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


# @app.post('/embedding')
# async def embedding(text: str = Form(...), max_length: int = Form(...)):
#     global model, tokenizer, device
#     # genrate token ids
#     input = tokenizer(text, truncation=True,
#                       max_length=max_length if max_length else 2048)['input_ids']
#     # genrate tensor [[...]]
#     tensor = torch.tensor(input)[None, :]
#     embeddings = model(tensor.to(device))
#     output = embeddings[1][1][0]
#     print(output.shape)
#     output = output.cpu().detach().numpy().tolist()
#     return {'embedding': output}


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
    tokenizer = AutoTokenizer.from_pretrained(
        'THUDM/chatglm-6b', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'THUDM/chatglm-6b', trust_remote_code=True).half().cuda()
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000)

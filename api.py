import uvicorn
import json
import datetime
import torch

from fastapi import FastAPI, Form, Request
from transformers import AutoTokenizer, AutoModel
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + \
        prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


def predict(tokenizer, prompt, history, max_length, top_p, temperature):
    for response, history in model.stream_chat(tokenizer, prompt, history, max_length, top_p, temperature):
        yield response, history


@app.get("/chat-stream")
async def chat_stream(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json = json.loads(json.dumps(json_post_raw))
    prompt = json.get('prompt')
    history = json.get('history') if history else []
    max_length = json.get('max_length') if max_length else 2048
    top_p = json.get('top_p') if top_p else 0.7
    temperature = json.get('temperature') if temperature else 0.95
    return EventSourceResponse(
        predict(tokenizer, prompt, history, max_length, top_p, temperature)
    )


@app.post("/embedding")
async def embedding(text: str = Form(...), max_length: int = Form(...)):
    global model, tokenizer, device
    # genrate token ids
    input = tokenizer(text, truncation=True,
                      max_length=max_length if max_length else 2048)['input_ids']
    # genrate tensor [[...]]
    tensor = torch.tensor(input)[None, :]
    embeddings = model(tensor.to(device))
    output = embeddings[1][1][0]
    print(output.shape)
    output = output.cpu().detach().numpy().tolist()
    return {"embedding": output}


@app.post("/tokenize")
async def tokenize(request: Request):
    global model, tokenizer
    token = tokenizer.tokenize(text)
    ids = tokenizer(text, truncation=True,
                    max_length=max_length if max_length else 2048)['input_ids']
    return {"token": token, 'ids': ids}


if __name__ == '__main__':
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained(
        "THUDM/chatglm-6b", trust_remote_code=True).half().cuda(device)
    model.eval()
    print(dir(model))
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

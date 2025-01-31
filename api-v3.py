from flask import Flask, request, jsonify
from vllm import VLLMModel
import torch

app = Flask(__name__)

# Initialize the model; make sure to load the correct configuration
model_name = "./model/bge-m3"
model = VLLMModel.from_pretrained(model_name)
tokenizer = model.tokenizer


@app.route('/v1/embeddings', methods=['POST'])
def create_embeddings():
    data = request.json
    input_text = data.get('input')
    model_id = data.get('model', model_name)
    encoding_format = data.get('encoding_format', 'float')

    # Check input validity
    if not input_text or (isinstance(input_text, str) and not input_text.strip()) or (isinstance(input_text, list) and not all(input_text)):
        return jsonify({"error": "Input text cannot be empty."}), 400

    inputs = tokenizer(input_text, return_tensors='pt', truncation=True)

    # Convert text to embeddings
    with torch.no_grad():
        embeddings = model(
            **inputs).last_hidden_state.mean(dim=1).cpu().tolist()

    # Construct response
    response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": i
            }
            for i, embedding in enumerate(embeddings)
        ],
        "model": model_id,
        "usage": {
            "prompt_tokens": len(inputs['input_ids'][0]),
            "total_tokens": len(inputs['input_ids'][0])
        }
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

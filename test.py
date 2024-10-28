from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 设置模型名称和本地路径
local_model_path = "./model/chatglm3-6b-128k"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, trust_remote_code=True).cuda()

# 准备输入
input_text = "你好，世界！"
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# 执行推理
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

# 输出结果
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

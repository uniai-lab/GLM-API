

"""
# 单GPU运行环境（本地加载模型）
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置设备为CUDA（GPU）
device = "cuda"

# 使用本地路径加载模型
local_model_path = './model/glm-4-9b-chat'
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to(device).eval()

# 打印当前使用的GPU序号
current_device = torch.cuda.current_device()
print(f"GPU usage: {current_device}")

# 输入文本
query = "你好，介绍一下你自己"
# 应用聊天模版，生成输入张量
inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True).to(device)

# 生成文本的参数
gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}

# 在不同计算梯度的上下文中生成文本
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
"""





"""
# 多GPU运行环境（本地加载模型）
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置设备为CUDA（GPU）
device = "cuda"

# 读取本地模型
local_model_path = './model/glm-4-9b-chat'

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(local_model_path, 
                                          trust_remote_code=True, 
                                          use_safetensors=True)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(local_model_path, 
                                             torch_dtype=torch.bfloat16,
                                             low_cpu_mem_usage=True, 
                                             trust_remote_code=True)

# 使用DataParallel进行多GPU并行
model = torch.nn.DataParallel(model)
model = model.to(device).eval()

# 打印使用的GPU设备序号
print(f"GPU usage: {model.device_ids}")

# 输入文本
query = "你好，介绍一下你自己"
# 应用聊天模版生成输入张量
inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True).to(device)

# 生成文本的参数
gen_kwargs = {
    "max_length": 2500, 
    "do_sample": True, 
    "top_k": 1
}

# 在不同计算梯度的上下文中生成文本
with torch.no_grad():
    # 生成文本
    outputs = model.module.generate(**inputs, **gen_kwargs)
    # 去除输入部分，只保留生成的部分
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    # 解码并打印生成的文本
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""




# 官方文档
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4-9b-chat", trust_remote_code=True)

query = "你好"

inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )

inputs = inputs.to(device)
model = AutoModelForCausalLM.from_pretrained(
    "THUDM/glm-4-9b-chat",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))







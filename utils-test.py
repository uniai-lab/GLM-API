import gc
import os
from copy import deepcopy
from typing import Dict, Union, Optional

import torch
from torch.nn import Module
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    # 本文件来源于https://github.com/THUDM/ChatGLM-6B/blob/main/utils.py
    # 仅此处做少许修改以支持ChatGLM3
    device_map = {
        'transformer.embedding.word_embeddings': 0,
        'transformer.encoder.final_layernorm': 0,
        'transformer.output_layer': 0,
        'transformer.rotary_pos_emb': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.encoder.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = AutoModel.from_pretrained(
            checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response(output, history):
    content = ""
    history = deepcopy(history)
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            history.append(
                {

                    "role": "assistant",
                    "metadata": metadata,
                    "content": content
                }
            )
            content = content.replace("[[训练时间]]", "2023年")
        else:
            history.append(
                {
                    "role": "assistant",
                    "metadata": metadata,
                    "content": content
                }
            )
            if history[0]["role"] == "system" and "tools" in history[0]:
                content = "\n".join(content.split("\n")[1:-1])

                def tool_call(**kwargs):
                    return kwargs

                parameters = eval(content)
                content = {
                    "name": metadata.strip(),
                    "parameters": parameters
                }
            else:
                content = {
                    "name": metadata.strip(),
                    "content": content
                }
    return content, history


# 去除字符串首尾的空白字符
def trim(input):
    if isinstance(input, str):
        return input.strip()
    else:
        return input


# 流式生成ChatGLM3的响应（此函数用于逐步生成响应，并在每次生成后返回部分结果）
# 参数一：预训练的模型实例
# 参数二：预训练的分词器实例
# 参数三：包含生成参数的字典
@torch.inference_mode()
def generate_stream_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    # 提取生成参数
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    # 提取最新一条消息的内容和角色
    query, role = messages[-1].content, messages[-1].role

    # 构建消息历史，不包含最后一条消息
    history = [m.dict(exclude_none=True) for m in messages[:-1]]

    # 使用分词器构建输入
    inputs = tokenizer.build_chat_input(query, history=history, role=role)
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    # 检查输入长度是否超过模型的最大序列长度
    if input_echo_len >= model.config.seq_length:
        raise

    # 结束标志的token ID
    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
        tokenizer.get_command("<|observation|>")
    ]
    # 分布调试
    print(f"eos_token_id: {eos_token_id}")

    # 生成参数配置
    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    # 将当前消息添加到历史消息中
    history.append(
        {
            "role": role,
            "content": query
        }
    )

    total_len = 0
    # 使用模型进行流式生成
    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len:-1]

        response = tokenizer.decode(output_ids)

        # -----
        # print(trim(response))
        # -----

        if response and response[-1] != "�":
            yield {
                "text": trim(response),
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": None,
            }

    # 最后一个生成结果包含finish_reason，我们将其设置为stop
    ret = {
        "text": trim(response),
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    # 清理GPU缓存
    gc.collect()
    torch.cuda.empty_cache()


# 生成ChatGLM3的完整响应（此函数用于一次性生成完整响应）
# 参数一：预训练的模型实例
# 参数二：预训练的分词器实例
# 参数三：包含生成参数的字典
def generate_chatglm3(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    print("模型: chatglm3-6b")
    for response in generate_stream_chatglm3(model, tokenizer, params):
        pass
    # 完整的响应
    return response


"""
# 根据GLM3修改的GLM4接口


# 流式生成ChatGLM3的响应（此函数用于逐步生成响应，并在每次生成后返回部分结果）
# 参数一：预训练的模型实例
# 参数二：预训练的分词器实例
# 参数三：包含生成参数的字典
@torch.inference_mode()
def generate_stream_chatglm4(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    # 提取生成参数
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)

    # 提取最新一条消息的内容和角色
    query, role = messages[-1].content, messages[-1].role

    # 构建消息历史，不包含最后一条消息
    history = [m.dict(exclude_none=True) for m in messages[:-1]]

    # 使用分词器构建输入
    # 'ChatGLM4Tokenizer' object has no attribute 'build_chat_input'
    # inputs = tokenizer.build_chat_input(query, history=history, role=role)
    inputs = tokenizer.apply_chat_template(
        [{"role": role, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    # 检查输入长度是否超过模型的最大序列长度
    if input_echo_len >= model.config.seq_length:
        raise

    # 结束标志的token ID
    # 'ChatGLM4Tokenizer' object has no attribute 'get_command'
    # eos_token_id = [
        # tokenizer.eos_token_id,
        # tokenizer.get_command("<|user|>"),
        # tokenizer.get_command("<|observation|>")
    # ]
    # 一个不确定的修改
    # 在 GLM4 的官方文档中，没有提到需要明确设置 eos_token_id
    # 因此，我们可以依赖模型的默认行为来处理结束标志
    eos_token_id = tokenizer.eos_token_id
    # 分布调试
    print(f"eos_token_id: {eos_token_id}")

    # 生成参数配置
    gen_kwargs = {
        "max_length": max_new_tokens + input_echo_len,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    # 将当前消息添加到历史消息中
    history.append(
        {
            "role": role,
            "content": query
        }
    )

    total_len = 0
    # 使用模型进行流式生成
    # 对模型生成的每个输出进行处理
    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        # 将生成的张量转换为列表
        total_ids = total_ids.tolist()[0]
        # 计算生成的总长度
        total_len = len(total_ids)
        # 根据是否需要回显决定输出的ID
        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len:-1]

        # 使用分词器解码生成的ID，转换为字符串
        response = tokenizer.decode(output_ids)

        # -----
        print(trim(response))
        # -----

        # ----
        # 检查生成的内容是否包含特殊字样或结束标志符，如果是则停止生成
        if "<|user|>" in response or eos_token_id in total_ids:
            break
        # -----

        # 如果响应有效且最后一个字符不是替代字符
        # 则生成响应字典，并通过yield返回
        if response and response[-1] != "�":
            yield {
                "text": trim(response),
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": None,
            }

    # 最后一个生成结果包含finish_reason，我们将其设置为stop
    ret = {
        "text": trim(response)[:-9],        # 删除尾部的"<|user|>"
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }

    # 返回最终的响应
    yield ret

    # 清理GPU缓存
    gc.collect()
    torch.cuda.empty_cache()


# 生成ChatGLM4的完整响应（此函数用于一次性生成完整响应）
# 参数一：预训练的模型实例
# 参数二：预训练的分词器实例
# 参数三：包含生成参数的字典
def generate_chatglm4(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    print("模型: glm-4-9b-chat")
    for response in generate_stream_chatglm4(model, tokenizer, params):
        pass
    # 完整的响应
    return response

"""








# ------------------------------------------------------------------------------------------------------------------------------


def contains_custom_function(value: str) -> bool:
    """
    根据特定的函数前缀确定是否包含'function_call'。

    例如，在 "tools_using_demo/tool_register.py" 中定义的函数都是 "get_xxx"，并以 "get_" 开头。

    [注意] 这不是一种严格的判断方法，仅供参考。

    :param value: 需要检查的字符串值
    :return: 如果字符串包含 "get_" 前缀，返回 True；否则返回 False
    """
    # 检查 value 是否为非空字符串并且是否包含 "get_" 前缀
    return value and 'get_' in value


# ------------------------------------------------------------------------------------------------------------------------------


# 导入所需要的库
import re
import json
import time
import random
import string
from asyncio.log import logger
from typing import Union, List, Dict
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from sse_starlette.sse import EventSourceResponse
from typing import List, Literal, Optional, Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Response
from transformers import AutoTokenizer, LogitsProcessor
from vllm import SamplingParams, AsyncEngineArgs, AsyncLLMEngine


# 处理模型生成的输出字符串，并判断该输出是否包含工具调用的格式
# 如果包含，并设置了使用工具，则构造一个包含工具名称和参数的字典返回
# 如果不包含，则返回原始输出
def process_response(output: str, tools: dict | List[dict] = None, use_tool: bool = False) -> Union[str, dict]:
    # 将输出按行分割，并去除首位的空白字符
    lines = output.strip().split("\n")
    arguments_json = None                                       # 初始化变量，用于储存解析后的JSON参数
    special_tools = ["cogview", "simple_browser"]               # 特殊工具列表
    tools = {tool['function']['name'] for tool in tools}        # 提取工具名称集合

    # 这是一个简单的工具比较函数，不能保证拦截所有非工具输出的结果，比如参数未对齐等特殊情况。
    ##TODO 如果你希望做更多判断，可以在这里进行逻辑完善。

    # 判断输出是否包含工具调用的格式
    if len(lines) >= 2 and lines[1].startswith("{"):
        function_name = lines[0].strip()            # 提取函数名称
        arguments = "\n".join(lines[1:]).strip()    # 提取并拼接参数
        if function_name in tools or function_name in special_tools:
            # 尝试解析参数为JSON
            try:
                arguments_json = json.loads(arguments)
                is_tool_call = True
            # 如果解析失败了，判断是否为特殊工具
            except json.JSONDecodeError:
                is_tool_call = function_name in special_tools

            # 如果是工具调用且使用工具，则构造返回内容
            if is_tool_call and use_tool:
                content = {
                    "name": function_name,
                    "arguments": json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                            ensure_ascii=False)
                }
                if function_name == "simple_browser":
                    search_pattern = re.compile(r'search\("(.+?)"\s*,\s*recency_days\s*=\s*(\d+)\)')
                    match = search_pattern.match(arguments)
                    if match:
                        content["arguments"] = json.dumps({
                            "query": match.group(1),
                            "recency_days": int(match.group(2))
                        }, ensure_ascii=False)
                elif function_name == "cogview":
                    content["arguments"] = json.dumps({
                        "prompt": arguments
                    }, ensure_ascii=False)
                # 返回构造的内容
                return content
    # 入伏哦不是工具调用，返回原始输出
    return output.strip()


# 处理用户的输入消息，基于给定的采样参数生成响应
@torch.inference_mode()
async def generate_stream_glm4(params):
    messages = params["messages"]
    tools = params["tools"]
    tool_choice = params["tool_choice"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 8192))

    # 处理消息
    messages = process_messages(messages, tools=tools, tool_choice=tool_choice)
    # 应用聊天模版生成输入
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": -1,
        "repetition_penalty": repetition_penalty,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop_token_ids": [151329, 151336, 151338],     # 停止生成的token ID列表
        "ignore_eos": False,                            # 是否忽略结束符
        "max_tokens": max_new_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    # 创建采样参数对象
    sampling_params = SamplingParams(**params_dict)

    # 异步生成，并逐步返回输出结果
    async for output in engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}"):
        input_len = len(output.prompt_token_ids)         # 输入token长度
        output_len = len(output.outputs[0].token_ids)    # 输出token长度
        # 储存从模型中生成的响应信息
        ret = {
            "text": output.outputs[0].text,
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": output_len,
                "total_tokens": output_len + input_len
            },
            "finish_reason": output.outputs[0].finish_reason,
        }
        yield ret

    # 清理GPU缓存
    gc.collect()
    torch.cuda.empty_cache()


# 将用户的输入消息，处理成适合输入模型的格式
# （会根据工具选择和消息内容生成适当的系统消息、助手消息，和观察消息）
def process_messages(messages, tools=None, tool_choice="none"):
    _messages = messages
    processed_messages = []
    msg_has_sys = False

    # 过滤工具，选择需要使用的工具
    def filter_tools(tool_choice, tools):
        function_name = tool_choice.get('function', {}).get('name', None)
        if not function_name:
            return []
        filtered_tools = [
            tool for tool in tools
            if tool.get('function', {}).get('name') == function_name
        ]
        return filtered_tools

    # 根据工具选择，和消息内容，处理系统消息
    if tool_choice != "none":
        if isinstance(tool_choice, dict):
            tools = filter_tools(tool_choice, tools)
        if tools:
            processed_messages.append(
                {
                    "role": "system",
                    "content": None,
                    "tools": tools
                }
            )
            msg_has_sys = True

    # 处理assistant消息
    if isinstance(tool_choice, dict) and tools:
        processed_messages.append(
            {
                "role": "assistant",
                "metadata": tool_choice["function"]["name"],
                "content": ""
            }
        )
        
    # 遍历消息，处理不同角色的消息
    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        tool_calls = getattr(m, 'tool_calls', None)

        if role == "function":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content
                }
            )
        elif role == "tool":
            processed_messages.append(
                {
                    "role": "observation",
                    "content": content,
                    "function_call": True
                }
            )
        elif role == "assistant":
            if tool_calls:
                for tool_call in tool_calls:
                    processed_messages.append(
                        {
                            "role": "assistant",
                            "metadata": tool_call.function.name,
                            "content": tool_call.function.arguments
                        }
                    )
            else:
                for response in content.split("\n"):
                    if "\n" in response:
                        metadata, sub_content = response.split("\n", maxsplit=1)
                    else:
                        metadata, sub_content = "", response
                    processed_messages.append(
                        {
                            "role": role,
                            "metadata": metadata,
                            "content": sub_content.strip()
                        }
                    )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            processed_messages.append({"role": role, "content": content})

    # 如果没有工具或为选择工具，则添加系统消息
    if not tools or tool_choice == "none":
        for m in _messages:
            if m.role == 'system':
                processed_messages.insert(0, {"role": m.role, "content": m.content})
                break
    return processed_messages


# 定义异步生成聊天函数
async def generate_chatglm4(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, engine: AsyncLLMEngine, params: dict):
    print("模型: glm-4-9b-chat")  # 打印模型名称
    async for response in generate_stream_glm4(model, tokenizer, engine, params):  # 异步生成响应
        pass  # 跳过操作
    return response  # 返回响应


# ------------------------------------------------------------------------------------------------------------------------------

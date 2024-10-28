import gc
import json
from classType import (
    ChatCompletionResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FunctionCall,
)
import torch
import time
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.generation.logits_process import LogitsProcessor
from typing import Union, Tuple
from loguru import logger


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


def process_response_glm3(output: str, use_tool: bool = False) -> Union[str, dict]:
    content = ""
    for response in output.split("<|assistant|>"):
        metadata, content = response.split("\n", maxsplit=1)
        if not metadata.strip():
            content = content.strip()
            content = content.replace("[[训练时间]]", "2023年")
        else:
            if use_tool:
                content = "\n".join(content.split("\n")[1:-1])

                def tool_call(**kwargs):
                    return kwargs

                parameters = eval(content)
                content = {
                    "name": metadata.strip(),
                    "arguments": json.dumps(parameters, ensure_ascii=False),
                }
            else:
                content = {"name": metadata.strip(), "content": content}
    return content


@torch.inference_mode()
async def generate_stream_chatglm3(params: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    messages = params["messages"]
    tools = params["tools"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)
    messages = process_chatglm_messages(messages, tools=tools)
    query, role = messages[-1]["content"], messages[-1]["role"]

    inputs = tokenizer.build_chat_input(
        query, history=messages[:-1], role=role)
    inputs = inputs.to(model.device)
    input_echo_len = len(inputs["input_ids"][0])

    if input_echo_len >= model.config.seq_length:
        print(f"Input length larger than {model.config.seq_length}")

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
        tokenizer.get_command("<|observation|>")
    ]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "logits_processor": [InvalidScoreLogitsProcessor()],
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    for total_ids in model.stream_generate(**inputs, eos_token_id=eos_token_id, **gen_kwargs):
        total_ids = total_ids.tolist()[0]
        total_len = len(total_ids)
        if echo:
            output_ids = total_ids[:-1]
        else:
            output_ids = total_ids[input_echo_len:-1]

        response = tokenizer.decode(output_ids)
        if response and response[-1] != "�":
            response, stop_found = apply_stopping_strings(
                response, ["<|observation|>"])

            yield {
                "text": response,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
                "finish_reason": "function_call" if stop_found else None,
            }

            if stop_found:
                break

    # Only last stream result contains finish_reason, we set finish_reason as stop
    ret = {
        "text": response,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
        "finish_reason": "stop",
    }
    yield ret

    gc.collect()
    torch.cuda.empty_cache()


def process_chatglm_messages(messages, tools=None):
    _messages = messages
    messages = []
    msg_has_sys = False
    if tools:
        messages.append(
            {
                "role": "system",
                "content": "Answer the following questions as best as you can. You have access to the following tools:",
                "tools": tools,
            }
        )
        msg_has_sys = True

    for m in _messages:
        role, content, func_call = m.role, m.content, m.function_call
        if role == "function":
            messages.append({"role": "observation", "content": content})

        elif role == "assistant" and func_call is not None:
            for response in content.split("<|assistant|>"):
                metadata, sub_content = response.split("\n", maxsplit=1)
                messages.append(
                    {"role": role, "metadata": metadata,
                        "content": sub_content.strip()}
                )
        else:
            if role == "system" and msg_has_sys:
                msg_has_sys = False
                continue
            messages.append({"role": role, "content": content})
    return messages


async def predict_glm3(
    model_id: str, params: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, id="", choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    async for new_response in generate_stream_chatglm3(params, model, tokenizer):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode

        finish_reason = new_response["finish_reason"]
        if len(delta_text) == 0 and finish_reason != "function_call":
            continue

        function_call = None
        if finish_reason == "function_call":
            try:
                function_call = process_response_glm3(
                    decoded_unicode, use_tool=True)
            except:
                logger.warning(
                    "Failed to parse tool call, maybe the response is not a tool call or have been answered."
                )

        if isinstance(function_call, dict):
            function_call = FunctionCall(**function_call)

        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
            function_call=(
                function_call if isinstance(
                    function_call, FunctionCall) else None
            ),
        )

        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=delta, finish_reason=finish_reason
        )
        chunk = ChatCompletionResponse(
            model=model_id, id="", choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, id="", choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield "[DONE]"


async def predict_glm3_stream(
    model_id: str, params: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer
):
    output = ""
    is_function_call = False
    has_send_first_chunk = False
    async for new_response in generate_stream_chatglm3(params, model, tokenizer):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(output):]
        output = decoded_unicode

        # When it is not a function call and the character length is> 7,
        # try to judge whether it is a function call according to the special function prefix
        if not is_function_call and len(output) > 7:

            # Determine whether a function is called
            is_function_call = output and "get_" in output
            if is_function_call:
                continue

            # Non-function call, direct stream output
            finish_reason = new_response["finish_reason"]

            # Send an empty string first to avoid truncation by subsequent next() operations.
            if not has_send_first_chunk:
                message = DeltaMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                )
                choice_data = ChatCompletionResponseStreamChoice(
                    index=0, delta=message, finish_reason=finish_reason
                )
                chunk = ChatCompletionResponse(
                    model=model_id,
                    id="",
                    choices=[choice_data],
                    created=int(time.time()),
                    object="chat.completion.chunk",
                )
                yield "{}".format(chunk.model_dump_json(exclude_unset=True))

            send_msg = delta_text if has_send_first_chunk else output
            has_send_first_chunk = True
            message = DeltaMessage(
                content=send_msg,
                role="assistant",
                function_call=None,
            )
            choice_data = ChatCompletionResponseStreamChoice(
                index=0, delta=message, finish_reason=finish_reason
            )
            chunk = ChatCompletionResponse(
                model=model_id,
                id="",
                choices=[choice_data],
                created=int(time.time()),
                object="chat.completion.chunk",
            )
            yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    if is_function_call:
        yield output
    else:
        yield "[DONE]"


async def parse_output_text_glm3(model_id: str, value: str):

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant", content=value), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id, id="", choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, id="", choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    yield "[DONE]"


def apply_stopping_strings(reply, stop_strings) -> Tuple[str, bool]:
    stop_found = False
    for string in stop_strings:
        idx = reply.find(string)
        if idx != -1:
            reply = reply[:idx]
            stop_found = True
            break

    if not stop_found:
        # If something like "\nYo" is generated just before "\nYou: is completed, trim it
        for string in stop_strings:
            for j in range(len(string) - 1, 0, -1):
                if reply[-j:] == string[:j]:
                    reply = reply[:-j]
                    break
            else:
                continue

            break

    return reply, stop_found

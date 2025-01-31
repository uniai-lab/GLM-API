from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

import time


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


class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None


class ChatCompletionMessageToolCall(BaseModel):
    index: Optional[int] = 0
    id: Optional[str] = None
    function: FunctionCall
    type: Optional[Literal["function"]] = "function"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "tool", "function"]
    content: str = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    tool_choice: Optional[Union[str, dict]] = None
    repetition_penalty: Optional[float] = 1.1


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call", "tool_calls"]


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop",
                                    "length", "function_call", "tool_calls"]]
    index: int


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    id: Optional[str]
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice,
                        ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


class EmbeddingRequest(BaseModel):
    model: str = "text2vec-large-chinese"
    prompt: List[str]


class EmbeddingResponse(BaseModel):
    data: List[List[float]]
    model: str
    object: str


class EmbeddingRequestV1(BaseModel):
    model: str  # Model ID, e.g., "BAAI/bge-m3"
    input: Union[str, List[str]]  # Input can be a string or a list of strings
    encoding_format: Optional[str] = "float"  # Optional, default is "float"
    # Optional, to allow specifying dimensions if necessary
    dimensions: Optional[int] = None
    user: Optional[str] = None  # Optional user identifier


class EmbeddingObject(BaseModel):
    object: str = "embedding"  # Should be "embedding"
    embedding: List[float]  # Embedding vector
    index: int  # Index of the embedding in the response list


class EmbeddingUsageInfo(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponseV1(BaseModel):
    object: str = "list"  # Response object type
    data: List[EmbeddingObject]  # List of embedding objects
    model: str  # Model ID used
    usage: Optional[EmbeddingUsageInfo] = None  # Optional usage information


class TokenizeRequest(BaseModel):
    prompt: str
    max_tokens: int = 4096
    model: str


class TokenizeResponse(BaseModel):
    tokenIds: List[int]
    tokens: List[str]
    model: Optional[str] = "bge-m3"
    object: str


class KeywordRequest(BaseModel):
    input: Union[str, List[str]]
    vocab: List[str] = None
    model: Optional[str] = "text2vec-large-chinese"
    top: Optional[int] = 10
    mmr: Optional[bool] = False
    maxsum: Optional[bool] = False
    diversity: Optional[float] = 0.3


class Keyword(BaseModel):
    name: str
    similarity: float


class KeywordResponse(BaseModel):
    model: str
    keywords: List[Keyword]

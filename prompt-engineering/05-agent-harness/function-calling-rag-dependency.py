"""04-function-calling-rag 的共用依賴：以 openai>=2.0 SDK 實作 embeddings、
chat completion 與「遞迴執行 tool_calls」的工具執行器（含 max_depth 上限）。
"""
from dotenv import load_dotenv
import os
import json

load_dotenv()

from openai import OpenAI

client = OpenAI()  # 讀取 OPENAI_API_KEY


def get_embeddings(input, dimensions=1536, model="text-embedding-3-small"):
    return client.embeddings.create(input=input, model=model, dimensions=dimensions).data[0].embedding


def get_completion(messages, model="gpt-4o", temperature=0, max_tokens=300, tools=None, tool_choice=None):
    """回傳 assistant message 的 dict（含 tool_calls），可直接 append 回 messages。"""
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs).choices[0].message.model_dump(exclude_none=True)


def get_completion_with_function_execution(
    messages, available_tools, model="gpt-4o", temperature=0,
    max_tokens=4096, tools=None, tool_choice=None, max_depth=5,
):
    """遞迴執行 tool_calls，直到模型不再要求工具呼叫；max_depth 防止無限遞迴。"""
    if max_depth <= 0:
        return {"role": "assistant", "content": "[已達工具呼叫上限 max_depth，停止]"}

    response = get_completion(
        messages, model=model, temperature=temperature,
        max_tokens=max_tokens, tools=tools, tool_choice=tool_choice,
    )

    if response.get("tool_calls"):
        messages.append(response)
        for tool_call in response["tool_calls"]:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            function_to_call = available_tools[function_name]
            function_response = function_to_call(**function_args)
            messages.append({
                "tool_call_id": tool_call["id"],
                "role": "tool",
                "name": function_name,
                "content": str(function_response),
            })
        return get_completion_with_function_execution(
            messages, available_tools, model=model, temperature=temperature,
            max_tokens=max_tokens, tools=tools, tool_choice=tool_choice, max_depth=max_depth - 1,
        )
    return response


# --- PDF 解析（pypdf）---
from pypdf import PdfReader


def read_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


# --- 向量資料庫（chromadb>=1.0；免 pysqlite3 覆寫）---
import chromadb

chroma_client = chromadb.EphemeralClient()

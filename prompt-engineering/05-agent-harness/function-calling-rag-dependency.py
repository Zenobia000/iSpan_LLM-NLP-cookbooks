# Import necessary libraries
## 設定 OpenAI API Key 變數
from dotenv import load_dotenv
import os

# Load the environment variables from .env file
load_dotenv()

# Access the API key
openai_api_key = os.getenv('OPENAI_API_KEY')

import requests
import json
from pprint import pp

def get_embeddings(input, dimensions = 1536, model="text-embedding-3-small"):
  payload = { "input": input, "model": model, "dimensions": dimensions }
  headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
  response = requests.post('https://api.openai.com/v1/embeddings', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  if response.status_code == 200 :
    return obj["data"][0]["embedding"]
  else :
    return obj["error"]
  
def get_completion(messages, model="gpt-4-turbo-preview", temperature=0, max_tokens=300, tools=None, tool_choice=None):
  payload = { "model": model, "temperature": temperature, "messages": messages, "max_tokens": max_tokens }
  if tools:
    payload["tools"] = tools
  if tool_choice:
    payload["tool_choice"] = tool_choice

  headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
  response = requests.post('https://api.openai.com/v1/chat/completions', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  print(f"   response: {obj}")
  if response.status_code == 200 :
    return obj["choices"][0]["message"] # 改成回傳上一層 message 物件
  else :
    return obj["error"]
  
def get_completion_with_function_execution(messages, available_tools, model="gpt-4-turbo-preview", temperature=0, max_tokens=4096, tools=None, tool_choice=None):
  print(f"called prompt: {messages}")
  response = get_completion(messages, model=model, temperature=temperature, max_tokens=max_tokens, tools=tools,tool_choice=tool_choice)

  if response.get("tool_calls"): # 或用 response 裡面的 finish_reason 判斷也行
    messages.append(response)

    # ------ 呼叫函數，這裡改成執行多 tool_calls (可以改成平行處理，目前用簡單的迴圈)
    for tool_call in response["tool_calls"]:
      function_name = tool_call["function"]["name"]
      function_args = json.loads(tool_call["function"]["arguments"])
      function_to_call = available_tools[function_name]

      print(f"   called function {function_name} with {function_args}")
      function_response = function_to_call(**function_args)
      messages.append(
          {
              "tool_call_id": tool_call["id"], # 多了 toll_call_id
              "role": "tool",
              "name": function_name,
              "content": function_response,
          }
      )

    # 進行遞迴呼叫
    return get_completion_with_function_execution(messages, available_tools, model=model, temperature=temperature, max_tokens=max_tokens, tools=tools,tool_choice=tool_choice)

  else:
    return response

from pypdf import PdfReader

reader = PdfReader("1130219.pdf")
number_of_pages = len(reader.pages)
page = reader.pages[0]
text = page.extract_text()
text


# sitecustomize.py
import pysqlite3
import sys
# 將標準 sqlite3 module 全部改為 pysqlite3
sys.modules['sqlite3'] = pysqlite3


import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="collection2")

for index, page in enumerate(reader.pages):
  chunk = page.extract_text()

  collection.add(
    documents = [chunk],
    embeddings = [ get_embeddings(chunk) ] ,
    metadatas = [ { "date": "2024年2月20日" } ],
    ids=[f"doc-1-{index}" ]
  )


def search_knowledgebase(query, n_results=2):
  """Performs a RAG search on the ChromaDB collection."""
  print(f"   Executing RAG search for: '{query}'")
  results = collection.query(
      query_embeddings=get_embeddings(query),
      n_results=n_results
  )
  context = '\n'.join('* ' + doc for doc in results['documents'][0])
  return context


from pydantic import BaseModel, Field
from typing import List


class Question(BaseModel):
    subquestion: str = Field(description="The question decomposited as much as possible")
    dependency: bool = Field(description="Does this subquestion depend on previous subquestion") # 這個子問題是否有依賴性

class QueryPlan(BaseModel):
    root_question: str = Field(description="The root question that the user asked")
    plan: List[Question] = Field(description="The plan to answer the root question and its sub-questions")


def execute_rag_plan(root_question: str, plan: List[dict]):
    """
    Executes a decomposed query plan to answer a complex question with dependencies.
    It sequentially runs RAG for each sub-question, using LLM calls to rewrite
    dependent questions and summarize intermediate answers.
    """
    print(f"--- Executing RAG Plan for: '{root_question}' ---")
    
    all_contexts = []
    previous_step_answer = ""

    for i, step in enumerate(plan):
        question_text = step["subquestion"]
        is_dependent = step["dependency"]
        
        print(f"\n- Step {i+1}/{len(plan)}: {question_text} (Dependent: {is_dependent})")

        # 1. Rewrite the query if it's dependent on the previous answer
        if is_dependent and previous_step_answer:
            rewrite_messages = [
                {"role": "system", "content": "You are a query rewriter. Your task is to rewrite a follow-up question into a standalone question based on the answer to the previous question. Output ONLY the rewritten question."},
                {"role": "user", "content": f"Previous Answer: '{previous_step_answer}'.\n\nFollow-up Question: '{question_text}'.\n\nStandalone Question:"}
            ]
            print(f"   Rewriting dependent query...")
            # Use a faster model for sub-tasks and the simple get_completion to avoid loops.
            rewritten_response = get_completion(rewrite_messages, model="gpt-3.5-turbo", temperature=0, max_tokens=150)
            question_text = rewritten_response["content"]
            print(f"   Rewritten to: '{question_text}'")

        # 2. Search the knowledge base with the (potentially rewritten) query
        retrieved_context = search_knowledgebase(question_text, n_results=2)
        all_contexts.append(f"### Context for sub-question '{question_text}':\n{retrieved_context}")

        # 3. Summarize the answer for the current step to feed into the next one
        summarize_messages = [
            {"role": "system", "content": "You are a reading comprehension assistant. Based ONLY on the provided context, provide a concise answer to the question. If the context does not contain the answer, state that the information is not available."},
            {"role": "user", "content": f"Context:\n{retrieved_context}\n\nQuestion: '{question_text}'\n\nConcise Answer:"}
        ]
        print(f"   Summarizing answer for this step...")
        summary_response = get_completion(summarize_messages, model="gpt-3.5-turbo", temperature=0, max_tokens=250)
        previous_step_answer = summary_response["content"]
        print(f"   Intermediate Answer: '{previous_step_answer}'")

    print("--- RAG Plan Execution Finished ---")
    # Combine all retrieved contexts to be used for the final answer synthesis
    return "\n\n".join(all_contexts)

available_tools = {
  "execute_rag_plan": execute_rag_plan,
}

user_messages = [
{"role": "system", "content": """
You are a finance/investment specialist. You will use the available tools to find relavent knowlege articles to create the answer.

First, analyze the user's question. If it is complex and contains multiple parts or dependencies, you MUST decompose it into a series of simple, sequential sub-questions. Then, you MUST call the `execute_rag_plan` function with this plan.

For example, if the user asks "What is the most popular product of company X and how did its sales perform last quarter?", you should decompose it into:
1. (dependency=False) What is the most popular product of company X?
2. (dependency=True) How did the sales of this product perform last quarter?

After the tool returns the combined context from all steps, synthesize a final, comprehensive answer based on ALL the provided context.

Answer ONLY with the facts from the retrieved context. If there isn't enough information, say you don't know. Do not generate answers that don't use the sources below.
"""},
   {"role": "user", "content": "請問台灣最具戰略價值的是什麼供應鏈? 此供應鏈在中國市場將表現如何?"}
]
tools = [{ "type": "function",
           "function": {
              "name": "execute_rag_plan",
              "description": "Decomposes a complex question into a multi-step plan and executes it to retrieve knowledge. Use this for questions with multiple parts or dependencies.",
              "parameters": QueryPlan.model_json_schema(),
            }
         }]

response = get_completion_with_function_execution(user_messages, available_tools, model="gpt-4-turbo-preview", tools=tools)

print("------ 最後結果:")
pp(response)


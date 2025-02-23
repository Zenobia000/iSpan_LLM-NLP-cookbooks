import os
import logging
from typing import List
import PyPDF2
import faiss
import numpy as np
from markdownify import markdownify as mdify
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# LangChain 0.3+ Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------------
# Document 結構
# ---------------------------------------------------------------------------------------
class Document(BaseModel):
    text: str
    id: str = ""

# ---------------------------------------------------------------------------------------
# 1. FAISS Retriever
# ---------------------------------------------------------------------------------------
class FAISSRetriever(BaseRetriever):
    """FAISS 向量檢索"""
    embeddings: OpenAIEmbeddings
    texts: List[str]
    top_k: int
    index: faiss.IndexFlatL2

    @classmethod
    def from_chunks(cls, embeddings: OpenAIEmbeddings, chunks: List[str], top_k: int = 3):

        """透過 @classmethod 建立 FAISS 索引"""
        logger.info("初始化 FAISS 檢索索引...")
        vector_dim = len(embeddings.embed_query("test"))  # 確保維度一致
        index = faiss.IndexFlatL2(vector_dim)
        vectors = np.array(embeddings.embed_documents(chunks), dtype="float32")
        index.add(vectors)

        return cls(
            embeddings=embeddings,
            texts=chunks,
            top_k=top_k,
            index=index
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """根據查詢檢索最相關的文檔"""
        query_vec = np.array([self.embeddings.embed_query(query)], dtype="float32")
        _, indices = self.index.search(query_vec, self.top_k)
        for idx in indices[0]:
            print(Document(text=self.texts[idx], id=f"chunk_{idx}"))

        return [Document(text=self.texts[idx], id=f"chunk_{idx}") for idx in indices[0]]

# ---------------------------------------------------------------------------------------
# 2. PDF -> Markdown -> chunk
# ---------------------------------------------------------------------------------------
def pdf_to_markdown_chunks(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"找不到檔案: {pdf_path}")

    chunks = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages, 1):
            text = mdify(page.extract_text() or "")
            chunks += [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
            print(f"page: {i}")
            print(f"chunks: {[text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]}")
            print("--------------------------------")
    return chunks

# ---------------------------------------------------------------------------------------
# 3. Chat Prompt Template (包含 agent_scratchpad)
# ---------------------------------------------------------------------------------------
def get_prompt_template():
    """建立基礎的 ChatPromptTemplate"""
    return ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. When you need to answer questions about specific documents or content, use the retriever_tool to get relevant information first."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{question}"),
        MessagesPlaceholder(variable_name="agent_scratchpad", optional=True),
    ])

# ---------------------------------------------------------------------------------------
# 4. 選擇 LLM
# ---------------------------------------------------------------------------------------
def get_llm(provider: str = "openai"):
    """選擇 LLM：支援 OpenAI / Anthropic Claude"""
    if provider == "openai":
        return ChatOpenAI(model="gpt-4", temperature=0.7)
    elif provider == "anthropic":
        return ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.7)
    else:
        raise ValueError("目前僅支援 OpenAI 或 Anthropic")


# ---------------------------------------------------------------------------------------
# 5. 建立 AgentExecutor (確保傳遞 question & context)
# ---------------------------------------------------------------------------------------
def create_agent(llm: ChatOpenAI | ChatAnthropic, retriever: FAISSRetriever):
    """
    建立 AgentExecutor：
    - 支援 OpenAI 和 Claude 的工具調用
    - 根據 LLM 類型自動選擇合適的工具格式
    """
    
    def query_function(input_text: str) -> str:
        """RAG tool: 檢索相關文檔並生成回答"""
        docs = retriever._get_relevant_documents(input_text)
        context = "\n\n".join([doc.text for doc in docs])
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based on the following context. If the context doesn't contain relevant information, please answer with 'I don't know'"),
            ("system", "Context: {context}"),
            ("human", "{question}")
        ])
        
        formatted_prompt = rag_prompt.format(
            context=context,
            question=input_text
        )
        
        return llm.invoke(formatted_prompt)

    # 根據 LLM 類型配置工具
    tool_config = {
        "name": "retriever",
        "func": query_function,
        "description": "Use this tool to search and get information from documents. Input should be a specific question.",
        "return_direct": True
    }

    # 如果是 Claude，添加特殊配置
    if isinstance(llm, ChatAnthropic):
        tool_config.update({
            "args_schema": {
                "type": "custom",
                "input": "string",
                "description": "The question to search for in the documents"
            }
        })
    
    # 建立 Tool
    retriever_tool = Tool(**tool_config)


    # Claude 特殊處理
    agent = create_openai_tools_agent(
        llm=llm,
        tools=[retriever_tool],
        prompt=get_prompt_template()
    )


    return AgentExecutor(agent=agent, tools=[retriever_tool], verbose=True)

# ---------------------------------------------------------------------------------------
# 6. 主程式
# ---------------------------------------------------------------------------------------
def main():
    """完整 RAG 工作流程，確保 retriever 融入查詢。"""
    pdf_path = r"D:\python_workspace\github\iSpan_LLM-NLP-cookbooks\Langchain_scratch\langchain_framework\Course\Module3\samples\AI模型產業分析.pdf"

    # 1. 讀取 PDF 並轉為 Markdown chunks
    chunks = pdf_to_markdown_chunks(pdf_path, chunk_size=500, overlap=50)

    # 2. 建立 OpenAI Embeddings
    embeddings = OpenAIEmbeddings()

    # 3. 透過 from_chunks() 創建 FAISS 檢索器
    retriever = FAISSRetriever.from_chunks(embeddings, chunks, top_k=3)

    # 4. 選擇 LLM (可以是 OpenAI 或 Anthropic)
    llm = get_llm(provider="openai")  # 或 "anthropic"

    # 5. 創建 AgentExecutor
    agent_executor = create_agent(llm, retriever)

    # 6. 測試查詢
    query = "数据标注⾏业的⾓⾊与全球格局？"
    response = agent_executor.invoke({"question": query, "agent_scratchpad": "", "chat_history": [], "context": ""})

    # 7. 輸出結果
    print("\n=== 查詢結果 ===")
    print(response)


if __name__ == "__main__":
    main()

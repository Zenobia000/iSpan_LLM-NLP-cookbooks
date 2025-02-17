"""
LangChain 自定義模型範例 - 實作

本模組實現了一個技術文檔查詢的 RAG Agent，整合了：
1. Claude 作為基礎 LLM
2. OpenAI Ada 002 作為 Embedding 模型
3. 自定義的文檔檢索器
4. 自定義的文檔處理工具
5. 自定義的 Agent 決策邏輯
"""

import os
import logging
from typing import (
    Any, Dict, List, Optional, Tuple, 
    Union, Annotated, Type
)
from pathlib import Path
import json
import time
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 載入環境變數
load_dotenv()

# 檢查必要的環境變數
required_env_vars = [
    'OPENAI_API_KEY',
    'ANTHROPIC_API_KEY'
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(
        f"缺少必要的環境變數: {', '.join(missing_vars)}\n"
        f"請確保 .env 文件中包含這些變數"
    )

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TechDocsLLM(ChatAnthropic):
    """技術文檔 LLM"""
    
    def __init__(self):
        super().__init__(
            model="claude-3-5-sonnet-20241022",
            temperature=0.2,
            max_tokens=2000,
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        )
    
    @property
    def _llm_type(self) -> str:
        return "tech_docs_llm"

class TechDocsEmbedding(OpenAIEmbeddings):
    """技術文檔 Embedding"""
    
    def __init__(self):
        super().__init__(
            model="text-embedding-3-small",
            chunk_size=1000,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )

class TechDocsRetriever:
    """技術文檔檢索器"""
    
    def __init__(self, docs_dir: Path, embeddings: OpenAIEmbeddings):
        self.docs_dir = docs_dir
        self.embeddings = embeddings
        self.persist_dir = self.docs_dir / "vectorstore"
        self.persist_dir.mkdir(exist_ok=True)
        
        # 設定 Chroma 客戶端
        import chromadb
        self.client_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=str(self.persist_dir)
        )
        
        try:
            self.vectorstore = self._init_vectorstore()
        except Exception as e:
            logger.error(f"初始化向量資料庫時發生錯誤: {str(e)}")
            raise
    
    def _init_vectorstore(self) -> Chroma:
        """初始化向量資料庫"""
        try:
            # 載入所有文檔
            docs = self._load_docs()
            if not docs:
                raise ValueError("沒有找到可用的文檔")
            
            # 建立向量資料庫
            return Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings,
                client_settings=self.client_settings
            )
        except Exception as e:
            logger.error(f"建立向量資料庫時發生錯誤: {str(e)}")
            raise
    
    def _load_docs(self) -> List[Document]:
        """載入文檔"""
        docs = []
        for file_path in self.docs_dir.glob("**/*.*"):
            if file_path.suffix in ['.txt', '.md', '.json']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    docs.append(Document(
                        page_content=content,
                        metadata={"source": str(file_path)}
                    ))
        return docs
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """檢索相關文檔"""
        return self.vectorstore.similarity_search(query, k=1)

class ToolInput(BaseModel):
    """工具輸入模型"""
    query: str = Field(..., description="查詢內容")

class ToolOutput(BaseModel):
    """工具輸出模型"""
    result: str = Field(..., description="處理結果")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="額外元數據")

class DocProcessTool(BaseTool):
    """文檔處理工具"""
    name: str = "doc_processor"
    description: str = "處理和分析技術文檔的工具"
    args_schema: Optional[Type[BaseModel]] = ToolInput
    
    def _run(self, query: str) -> str:
        """執行文檔處理"""
        # 實現文檔處理邏輯
        result = ToolOutput(
            result=f"已處理查詢: {query}",
            metadata={"timestamp": time.time()}
        )
        return result.result
    
    async def _arun(self, query: str) -> str:
        """非同步執行"""
        return self._run(query)

class QueryResult(BaseModel):
    """查詢結果模型"""
    answer: str = Field(description="回答內容")
    sources: List[str] = Field(description="使用的資料來源")
    confidence: float = Field(description="答案的置信度", ge=0, le=1)

class TechDocsAgent:
    """技術文檔 Agent"""
    
    def __init__(
        self,
        llm: TechDocsLLM,
        retriever: TechDocsRetriever,
        tools: List[BaseTool]
    ):
        self.llm = llm
        self.retriever = retriever
        self.tools = tools
        self.output_parser = PydanticOutputParser(pydantic_object=QueryResult)
        
        # 修改提示模板，使其更明確
        self.prompt_template = PromptTemplate(
            template="""基於以下文檔回答問題。如果無法找到答案，請說明原因。

文檔內容：
{context}

問題：{query}

請使用以下 JSON 格式回答，確保回答符合 JSON 格式：
{{
    "answer": "你的詳細回答",
    "sources": [],  # 來源將由系統自動填充
    "confidence": 0.8  # 請根據回答的確定性給出 0-1 之間的置信度
}}

回答：""",
            input_variables=["context", "query"]
        )
    
    def run(self, query: str) -> QueryResult:
        """執行查詢"""
        try:
            # 1. 檢索相關文檔
            docs = self.retriever.get_relevant_documents(query)
            
            # 2. 準備上下文
            context = "\n\n".join(doc.page_content for doc in docs)
            
            # 3. 生成提示
            prompt = self.prompt_template.format(
                context=context,
                query=query
            )
            
            # 4. 呼叫 LLM
            response = self.llm.invoke(prompt)
            
            # 5. 解析結果
            try:
                # 從 AIMessage 中提取內容
                response_text = response.content if hasattr(response, 'content') else str(response)
                result = self.output_parser.parse(response_text)
                result.sources = [doc.metadata["source"] for doc in docs]
                return result
            except Exception as e:
                logger.error(f"解析結果時發生錯誤: {str(e)}")
                logger.debug(f"原始回應: {response_text}")
                return QueryResult(
                    answer="無法解析回答",
                    sources=[doc.metadata["source"] for doc in docs],
                    confidence=0.0
                )
                
        except Exception as e:
            logger.error(f"查詢處理過程發生錯誤: {str(e)}")
            return QueryResult(
                answer=f"處理查詢時發生錯誤: {str(e)}",
                sources=[],
                confidence=0.0
            )

def main():
    """主程序"""
    try:
        # 初始化組件
        llm = TechDocsLLM()
        embeddings = TechDocsEmbedding()
        retriever = TechDocsRetriever(
            docs_dir=Path(__file__).parent / "samples",
            embeddings=embeddings
        )
        tools = [DocProcessTool()]
        
        # 建立 Agent
        agent = TechDocsAgent(
            llm=llm,
            retriever=retriever,
            tools=tools
        )
        
        # 測試查詢
        queries = [
            "LangChain 的核心組件有哪些？",  # 從 langchain_tutorial.md 可以回答
            "目前最準確的 AI 模型是哪個？",  # 從 model_comparison.csv 可以回答
            "台積電在日本的最新發展是什麼？",  # 從 tech_news.txt 可以回答
            "開發 AI 應用需要哪些基礎要求？",  # 從 dev_guide.html 可以回答
            "人工智能目前面臨哪些主要挑戰？"  # 從 ai_introduction.txt 可以回答
        ]
        
        # 執行查詢並輸出結果
        for query in queries:
            logger.info(f"\n=== 處理查詢: {query} ===")
            result = agent.run(query)
            logger.info(f"答案: {result.answer}")
            logger.info(f"來源: {result.sources}")
            logger.info(f"置信度: {result.confidence}")
        
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
"""
LangChain 自定義模型範例 - 模板

本模組展示如何建立自定義的 LangChain 組件，包括：
1. 自定義 LLM 包裝器
2. 自定義 Embedding 模型
3. 自定義 Chain
4. 自定義 Agent
5. 自定義 Tool
6. 自定義 Retriever
"""

from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings.base import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_core.agents import AgentExecutor, BaseAgent
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.prompts import PromptTemplate

class CustomLLM(BaseLanguageModel):
    """自定義 LLM 包裝器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化參數
    
    def _call(self, prompt: str, **kwargs) -> str:
        """實現調用邏輯"""
        pass
    
    @property
    def _llm_type(self) -> str:
        """返回 LLM 類型"""
        return "custom_llm"

class CustomEmbedding(Embeddings):
    """自定義 Embedding 模型"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化參數
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """將文本轉換為向量"""
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """將查詢轉換為向量"""
        pass

class CustomRetriever(BaseRetriever):
    """自定義檢索器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        # 初始化參數
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """實現檢索邏輯"""
        pass

class CustomTool(BaseTool):
    """自定義工具"""
    name: str = "custom_tool"
    description: str = "自定義工具描述"
    
    def _run(self, query: str) -> str:
        """實現工具邏輯"""
        pass
    
    async def _arun(self, query: str) -> str:
        """實現非同步工具邏輯"""
        pass

class CustomAgent(BaseAgent):
    """自定義 Agent"""
    
    @property
    def _agent_type(self) -> str:
        """返回 Agent 類型"""
        return "custom_agent"
    
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """實現規劃邏輯"""
        pass

class RAGAgentTemplate:
    """RAG Agent 模板"""
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        retriever: BaseRetriever,
        tools: List[BaseTool],
        agent: BaseAgent,
        callback_manager: Optional[BaseCallbackManager] = None
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.retriever = retriever
        self.tools = tools
        self.agent = agent
        self.callback_manager = callback_manager
        
        # 初始化 Agent 執行器
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            callback_manager=self.callback_manager,
            verbose=True
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """處理查詢"""
        # 1. 檢索相關文檔
        docs = self.retriever.get_relevant_documents(query)
        
        # 2. 構建增強的查詢
        enhanced_query = self._enhance_query(query, docs)
        
        # 3. 執行 Agent
        result = self.agent_executor.run(enhanced_query)
        
        return {
            "query": query,
            "enhanced_query": enhanced_query,
            "relevant_docs": docs,
            "result": result
        }
    
    def _enhance_query(self, query: str, docs: List[Document]) -> str:
        """增強查詢"""
        # 實現查詢增強邏輯
        pass

def main():
    """主程序"""
    # 初始化組件
    llm = CustomLLM()
    embeddings = CustomEmbedding()
    retriever = CustomRetriever()
    tools = [CustomTool()]
    agent = CustomAgent()
    
    # 建立 RAG Agent
    rag_agent = RAGAgentTemplate(
        llm=llm,
        embeddings=embeddings,
        retriever=retriever,
        tools=tools,
        agent=agent
    )
    
    # 測試查詢
    query = "如何使用 LangChain 建立 RAG 系統？"
    result = rag_agent.process_query(query)
    
    # 輸出結果
    print(f"Query: {result['query']}")
    print(f"Enhanced Query: {result['enhanced_query']}")
    print(f"Relevant Docs: {len(result['relevant_docs'])}")
    print(f"Result: {result['result']}")

if __name__ == "__main__":
    main() 
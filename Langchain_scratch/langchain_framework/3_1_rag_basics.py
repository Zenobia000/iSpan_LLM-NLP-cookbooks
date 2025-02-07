"""
LangChain 0.3+ RAG (Retrieval Augmented Generation) 基礎範例
展示如何實現基本的檢索增強生成系統

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- langchain-community>=0.0.1
- chromadb>=0.4.0
- python-dotenv>=0.19.0
- tiktoken>=0.5.0
"""

# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import os
from dotenv import load_dotenv
import shutil
from datetime import datetime
import hashlib
import json
from dataclasses import dataclass

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數 
load_dotenv()


class SearchResult(BaseModel):
    """搜尋結果模型"""
    query: str = Field(description="搜尋查詢")
    relevant_chunks: List[str] = Field(description="相關的文本片段")
    answer: str = Field(description="根據檢索內容生成的回答")
    sources: List[str] = Field(description="資訊來源")


@dataclass
class Document:
    """文檔資料結構"""
    content: str
    metadata: Dict[str, Any]
    doc_hash: str
    created_at: str
    updated_at: str


class DocumentManager:
    """文檔管理器：負責文檔的資料治理"""
    def __init__(self, persist_dir: str = "document_store"):
        self.persist_dir = persist_dir
        self.metadata_file = Path(persist_dir) / "metadata.json"
        self.documents: Dict[str, Document] = {}
        self._load_metadata()

    def _load_metadata(self):
        """載入文檔元數據"""
        try:
            if self.metadata_file.exists():
                data = json.loads(self.metadata_file.read_text(encoding="utf-8"))
                self.documents = {
                    doc_hash: Document(**doc_data)
                    for doc_hash, doc_data in data.items()
                }
                logger.info(f"已載入 {len(self.documents)} 個文檔的元數據")
        except Exception as e:
            logger.error(f"載入元數據失敗: {str(e)}")
            self.documents = {}

    def _save_metadata(self):
        """保存文檔元數據"""
        try:
            self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                doc_hash: {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "doc_hash": doc.doc_hash,
                    "created_at": doc.created_at,
                    "updated_at": doc.updated_at
                }
                for doc_hash, doc in self.documents.items()
            }
            self.metadata_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            logger.info(f"已保存 {len(self.documents)} 個文檔的元數據")
        except Exception as e:
            logger.error(f"保存元數據失敗: {str(e)}")

    def _compute_hash(self, content: str) -> str:
        """計算文檔雜湊值"""
        return hashlib.sha256(content.encode()).hexdigest()

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Document]:
        """添加新文檔"""
        try:
            doc_hash = self._compute_hash(content)
            now = datetime.now().isoformat()

            # 檢查是否為重複文檔
            if doc_hash in self.documents:
                logger.info(f"文檔已存在 (hash: {doc_hash})")
                return None

            # 建立文檔物件
            document = Document(
                content=content,
                metadata=metadata or {},
                doc_hash=doc_hash,
                created_at=now,
                updated_at=now
            )

            # 儲存文檔
            self.documents[doc_hash] = document
            self._save_metadata()

            return document
        except Exception as e:
            logger.error(f"添加文檔失敗: {str(e)}")
            return None

    def get_document(self, doc_hash: str) -> Optional[Document]:
        """獲取文檔"""
        return self.documents.get(doc_hash)

    def get_all_documents(self) -> List[Document]:
        """獲取所有文檔"""
        return list(self.documents.values())

    def update_document(
        self,
        doc_hash: str,
        content: str = None,
        metadata: Dict[str, Any] = None
    ) -> Optional[Document]:
        """更新文檔"""
        try:
            if doc_hash not in self.documents:
                logger.warning(f"文檔不存在 (hash: {doc_hash})")
                return None

            document = self.documents[doc_hash]
            now = datetime.now().isoformat()

            if content is not None:
                new_hash = self._compute_hash(content)
                if new_hash != doc_hash:
                    logger.warning("文檔內容變更將產生新的文檔")
                    return self.add_document(content, metadata or document.metadata)
                document.content = content

            if metadata is not None:
                document.metadata.update(metadata)

            document.updated_at = now
            self._save_metadata()

            return document
        except Exception as e:
            logger.error(f"更新文檔失敗: {str(e)}")
            return None


class DocumentProcessor:
    """文件處理器"""
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        document_manager: Optional[DocumentManager] = None
    ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        self.document_manager = document_manager or DocumentManager()

    def process_documents(self, documents: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        """處理文件並分割成小塊"""
        try:
            chunks = []
            for i, doc in enumerate(documents):
                # 獲取或添加文檔
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                document = self.document_manager.get_document(
                    self.document_manager._compute_hash(doc)
                ) or self.document_manager.add_document(doc, doc_metadata)

                if document:  # 無論是新文檔還是既有文檔
                    # 分割文檔
                    doc_chunks = self.text_splitter.split_text(document.content)
                    # 為每個分塊添加元數據
                    for chunk in doc_chunks:
                        chunks.append({
                            "content": chunk,
                            "metadata": {
                                "doc_hash": document.doc_hash,
                                "created_at": document.created_at,
                                **document.metadata
                            }
                        })

            if not chunks:
                logger.warning("沒有文檔可處理")
                return []

            logger.info(f"已處理 {len(documents)} 個文檔，生成 {len(chunks)} 個文本塊")
            return [chunk["content"] for chunk in chunks]
        except Exception as e:
            logger.error(f"文件處理失敗: {str(e)}")
            raise


class RAGSystem:
    """RAG 系統實現"""
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo"
        )

    def initialize_vectorstore(self, texts: List[str], persist_dir: str = "vectorstore"):
        """初始化向量存儲"""
        try:
            # 檢查是否已存在向量存儲且有內容
            if os.path.exists(persist_dir):
                logger.info(f"載入既有向量存儲: {persist_dir}")
                self.vectorstore = Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings
                )
                # 檢查是否有內容
                if self.vectorstore._collection.count() == 0 and texts:
                    logger.info("既有向量存儲為空，添加新文檔")
                    self.vectorstore = Chroma.from_texts(
                        texts=texts,
                        embedding=self.embeddings,
                        persist_directory=persist_dir
                    )
                    self.vectorstore.persist()
            else:
                # 只有在有文檔時才創建新的向量存儲
                if texts:
                    logger.info(f"創建新的向量存儲: {persist_dir}")
                    self.vectorstore = Chroma.from_texts(
                        texts=texts,
                        embedding=self.embeddings,
                        persist_directory=persist_dir
                    )
                    self.vectorstore.persist()
                else:
                    raise ValueError("沒有文檔可供初始化向量存儲")

            logger.info(f"向量存儲包含 {self.vectorstore._collection.count()} 個文檔")
        except Exception as e:
            logger.error(f"向量存儲初始化失敗: {str(e)}")
            raise

    def create_retrieval_chain(self) -> Any:
        """建立檢索鏈"""
        # 定義提示詞模板
        template = """根據以下資訊回答問題。如果無法從提供的資訊中找到答案，請說明無法回答。

        資訊:
        {context}

        問題: {question}

        請提供詳細且準確的回答。
        """

        prompt = ChatPromptTemplate.from_template(template)

        # 建立檢索鏈
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return chain

    def query(self, question: str) -> SearchResult:
        """執行查詢"""
        try:
            # 獲取相關文檔
            docs = self.vectorstore.similarity_search(question, k=3)
            chunks = [doc.page_content for doc in docs]
            sources = [getattr(doc.metadata, 'source', 'unknown') for doc in docs]

            # 生成回答
            chain = self.create_retrieval_chain()
            answer = chain.invoke(question)

            return SearchResult(
                query=question,
                relevant_chunks=chunks,
                answer=answer,
                sources=sources
            )

        except Exception as e:
            logger.error(f"查詢執行失敗: {str(e)}")
            raise


def demonstrate_rag(clean: bool = True):
    """展示 RAG 系統的使用"""
    persist_dir = "vectorstore"
    document_store = "document_store"

    # 如果需要清理
    if clean:
        clean_vectorstore(persist_dir)
        clean_vectorstore(document_store)  # 同時清理文檔存儲

    # 準備示例文檔
    documents = [
        """人工智慧（AI）是指由人製造出來的機器所表現出來的智慧。
        通常人工智慧是指通過普通電腦程式來呈現人類智慧的技術。
        該領域的研究包括機器人、語言識別、圖像識別、自然語言處理和專家系統等。""",

        """機器學習是人工智慧的一個分支，主要特點是可以從數據中學習規律，
        而無需明確編程。深度學習是機器學習的一個子領域，
        使用多層神經網絡來學習數據的層次化表示。"""
    ]

    documents += [
    """自然語言處理（NLP）是人工智慧的一個領域，研究計算機如何理解、
    生成和處理人類語言。常見應用包括機器翻譯、文本分類、語音識別和聊天機器人等。""",

    """計算機視覺是一種讓機器能夠理解和解釋視覺信息的技術。
    它利用圖像處理、深度學習和模式識別等方法來分析圖像或影片，
    主要應用於人臉識別、醫學影像分析和自動駕駛等領域。""",

    """專家系統是一種基於知識庫和推理機制的人工智慧系統，
    旨在模擬人類專家的決策能力。這類系統廣泛應用於醫療診斷、
    工程設計、金融分析和其他需要專業知識的領域。"""
]


    try:
        # 初始化文件處理器
        processor = DocumentProcessor()
        chunks = processor.process_documents(documents)

        if not chunks:
            logger.warning("沒有可用的文檔，無法進行演示")
            return

        # 初始化 RAG 系統
        rag_system = RAGSystem()
        rag_system.initialize_vectorstore(chunks, persist_dir)

        # 測試查詢
        test_questions = [
            "什麼是人工智慧？",
            "機器學習和深度學習有什麼關係？",
            "請解釋神經網絡的原理",
            "計算機視覺的常見應用有哪些？",
            "專家系統的常見應用有哪些？"
        ]




        for question in test_questions:
            print(f"\n問題: {question}")
            result = rag_system.query(question)
            print(f"回答: {result.answer}")
            print("\n相關文本片段:")
            for i, chunk in enumerate(result.relevant_chunks, 1):
                print(f"{i}. {chunk[:100]}...")

    except Exception as e:
        logger.error(f"示範執行失敗: {str(e)}")


def clean_vectorstore(persist_dir: str):
    """清理存儲目錄"""
    try:
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
            logger.info(f"已清理目錄: {persist_dir}")
    except Exception as e:
        logger.error(f"清理目錄失敗: {str(e)}")


def main():
    """主程式：展示 RAG 基礎功能"""
    print("=== LangChain 0.3+ RAG 基礎展示 ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return

    try:
        demonstrate_rag(clean=False)
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}")



if __name__ == "__main__":
    main()

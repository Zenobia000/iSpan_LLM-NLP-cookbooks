"""
LangChain 文件載入器範例
展示如何載入和處理不同類型的文件

主要功能：
1. 多種格式文件載入 (TXT, PDF, DOCX, CSV, JSON)
2. 網頁內容載入
3. 資料庫內容載入
4. 自定義載入器
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    BSHTMLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    DirectoryLoader
)

from langchain_core.document_loaders import BaseLoader as CoreBaseLoader
from langchain_core.documents import Document

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數
load_dotenv()

class DocumentInfo(BaseModel):
    """文件資訊模型"""
    content: str = Field(description="文件內容")
    metadata: Dict[str, Any] = Field(description="文件元數據")
    source_type: str = Field(description="來源類型")
    file_path: str = Field(description="檔案路徑")

class CustomNewsLoader(CoreBaseLoader):
    """自定義新聞載入器"""
    
    def __init__(self, file_path: str):
        """初始化載入器"""
        self.file_path = file_path
    
    async def aload(self) -> List[Document]:
        """非同步載入新聞文件"""
        return await super().aload()
    
    def load(self) -> List[Document]:
        """載入新聞文件"""
        documents = []
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    title, date, author, content = line.strip().split('|')
                    metadata = {
                        "title": title,
                        "date": date,
                        "author": author,
                        "source": "news",
                        "file_path": self.file_path
                    }
                    documents.append(
                        Document(
                            page_content=content,
                            metadata=metadata
                        )
                    )
        return documents

def load_text_file(file_path: str | Path) -> List[DocumentInfo]:
    """載入文本檔案"""
    logger.info(f"載入 TXT 檔案: {file_path}")
    loader = TextLoader(str(file_path), encoding='utf-8')
    docs = loader.load()
    return [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata,
            source_type="text",
            file_path=str(file_path)  # 確保轉換為字串
        )
        for doc in docs
    ]

def load_pdf_file(file_path: str) -> List[DocumentInfo]:
    """載入 PDF 檔案"""
    logger.info(f"載入 PDF 檔案: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata,
            source_type="pdf",
            file_path=file_path
        )
        for doc in docs
    ]

def load_word_file(file_path: str) -> List[DocumentInfo]:
    """載入 Word 檔案"""
    logger.info(f"載入 Word 檔案: {file_path}")
    loader = UnstructuredWordDocumentLoader(file_path)
    docs = loader.load()
    return [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata,
            source_type="docx",
            file_path=file_path
        )
        for doc in docs
    ]

def load_csv_file(file_path: str | Path) -> List[DocumentInfo]:
    """載入 CSV 檔案"""
    logger.info(f"載入 CSV 檔案: {file_path}")
    loader = CSVLoader(
        str(file_path),
        encoding="utf-8",  # 明確指定 UTF-8 編碼
        csv_args={
            "delimiter": ",",
            "quotechar": '"',
            "fieldnames": None
        }
    )
    docs = loader.load()
    return [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata,
            source_type="csv",
            file_path=str(file_path)
        )
        for doc in docs
    ]

def load_json_file(file_path: str | Path) -> List[DocumentInfo]:
    """載入 JSON 檔案"""
    logger.info(f"載入 JSON 檔案: {file_path}")
    loader = JSONLoader(
        str(file_path),
        jq_schema='.[]',
        content_key="content",
        metadata_func=lambda metadata, additional_fields: {
            **metadata,
            **additional_fields,  # 包含額外欄位
            "source_file": str(file_path),
            "loader_type": "json"
        }
    )
    docs = loader.load()
    return [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata,
            source_type="json",
            file_path=str(file_path)
        )
        for doc in docs
    ]

def load_web_content(url: str) -> List[DocumentInfo]:
    """載入網頁內容"""
    logger.info(f"載入網頁內容: {url}")
    loader = BSHTMLLoader(url)
    docs = loader.load()
    return [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata,
            source_type="web",
            file_path=url
        )
        for doc in docs
    ]

def load_directory(dir_path: str, glob: str = "**/*.*") -> List[DocumentInfo]:
    """載入整個目錄"""
    logger.info(f"載入目錄: {dir_path}")
    loader = DirectoryLoader(
        dir_path,
        glob=glob,
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,  # 指定預設載入器
        loader_kwargs={"encoding": "utf-8"}
    )
    docs = loader.load()
    return [
        DocumentInfo(
            content=doc.page_content,
            metadata=doc.metadata,
            source_type="directory",
            file_path=str(Path(dir_path) / doc.metadata.get("source", ""))
        )
        for doc in docs
    ]

def create_sample_files():
    """建立示範檔案"""
    # 取得當前檔案所在目錄
    current_dir = Path(__file__).parent
    print(current_dir)
    # 在當前目錄下建立 samples 目錄
    sample_dir = current_dir / "samples"
    logger.info(f"建立示範檔案目錄: {sample_dir}")
    
    try:
        # 確保目錄存在
        sample_dir.mkdir(exist_ok=True)
        
        # 建立文本檔案
        text_file = sample_dir / "sample.txt"
        with open(text_file, "w", encoding="utf-8") as f:
            f.write("這是一個示範文本檔案。\n包含多行內容。")
        logger.info(f"已建立文本檔案: {text_file}")
        
        # 建立 CSV 檔案
        csv_file = sample_dir / "sample.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("name,age,city\nJohn,30,台北\nMary,25,台中")
        logger.info(f"已建立 CSV 檔案: {csv_file}")
        
        # 建立 JSON 檔案
        json_file = sample_dir / "sample.json"
        with open(json_file, "w", encoding="utf-8") as f:
            f.write('[{"title": "文章1", "content": "內容1"}, {"title": "文章2", "content": "內容2"}]')
        logger.info(f"已建立 JSON 檔案: {json_file}")
        
        # 建立新聞檔案
        news_file = sample_dir / "news.txt"
        with open(news_file, "w", encoding="utf-8") as f:
            f.write("新聞標題1|2024-03-01|記者A|新聞內容1\n")
            f.write("新聞標題2|2024-03-02|記者B|新聞內容2\n")
        logger.info(f"已建立新聞檔案: {news_file}")
        
        return sample_dir
        
    except Exception as e:
        logger.error(f"建立示範檔案時發生錯誤: {str(e)}", exc_info=True)
        raise

def main():
    """主程式：展示文件載入功能"""
    print("=== LangChain 文件載入器展示 ===\n")
    sample_dir = None
    
    try:
        # 建立示範檔案
        sample_dir = create_sample_files()
        logger.info(f"示範檔案目錄: {sample_dir}")
        
        # 載入不同類型的檔案
        text_docs = load_text_file(sample_dir / "sample.txt")
        csv_docs = load_csv_file(sample_dir / "sample.csv")
        json_docs = load_json_file(sample_dir / "sample.json")
        
        # 使用自定義載入器
        news_loader = CustomNewsLoader(sample_dir / "news.txt")
        logger.info(f"載入 NEWS 檔案: {sample_dir / 'news.txt'}")
        news_docs = [
            DocumentInfo(
                content=doc.page_content,
                metadata=doc.metadata,
                source_type="news",
                file_path=str(sample_dir / "news.txt")
            )
            for doc in news_loader.load()
        ]
        
        # 顯示載入結果
        print("\n=== 文本檔案內容 ===")
        for doc in text_docs:
            print(f"\n內容: {doc.content}")
            print(f"元數據: {doc.metadata}")
            print("-" * 40)
        
        print("\n=== CSV 檔案內容 ===")
        for doc in csv_docs:
            print(f"\n內容: {doc.content}")
            print(f"元數據: {doc.metadata}")
            print("-" * 40)
        
        print("\n=== JSON 檔案內容 ===")
        for doc in json_docs:
            print(f"\n內容: {doc.content}")
            print(f"元數據: {doc.metadata}")
            print("-" * 40)
        
        print("\n=== 新聞檔案內容 ===")
        for doc in news_docs:
            print(f"\n內容: {doc.content}")
            print(f"元數據: {doc.metadata}")
            print("-" * 40)
        
    except Exception as e:
        logger.error(f"執行過程發生錯誤: {str(e)}", exc_info=True)
        raise
    # finally:
    #     # 清理示範檔案
    #     if sample_dir and sample_dir.exists():
    #         try:
    #             for file in sample_dir.glob("*.*"):
    #                 file.unlink()
    #                 logger.info(f"已刪除檔案: {file}")
    #             sample_dir.rmdir()
    #             logger.info(f"已刪除目錄: {sample_dir}")
    #         except Exception as e:
    #             logger.error(f"清理檔案時發生錯誤: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()


"""
LangChain 0.3+ RAG (Retrieval Augmented Generation) 基礎範例
展示如何實現基本的檢索增強生成系統

主要功能：
1. 文件加載與處理
2. 向量化與儲存
3. 相似度搜尋
4. LLM 串接與回答生成
"""

import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 載入環境變數 
load_dotenv()

class DocumentMetadata(BaseModel):
    """文件元數據模型"""
    source: str = Field(description="文件來源")
    topic: str = Field(description="主題分類")
    date: Optional[str] = Field(default=None, description="發布日期")

def create_demo_documents(long_text) -> List[Document]:

    # 使用文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", "。", "，", " "]
    )
    
    # 分割文本
    texts = text_splitter.split_text(long_text)
    
    # 建立文件列表
    documents = []
    for i, text in enumerate(texts):
        metadata = DocumentMetadata(
            source="wiki",
            topic=f"台北資訊_{i+1}",
            date="2024-03"
        )
        documents.append(
            Document(
                page_content=text.strip(),
                metadata=metadata.dict()
            )
        )

        print(f"{i} 份文件建立完成")
        print(Document(
                page_content=text.strip(),
                metadata=metadata.dict()
            ))
        print("-" * 50)
    
    logger.info(f"已建立 {len(documents)} 份文件")
    return documents

def create_rag_chain(article_text, model_name = "gpt-3.5-turbo", collection_name="taipei_info"):
    """建立 RAG Chain"""
    try:
        # 初始化 LLM
        llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            streaming=True
        )
        logger.info("LLM 初始化完成")
        
        # 初始化 Embeddings
        embeddings = OpenAIEmbeddings()
        logger.info("Embeddings 初始化完成")
        
        # 建立文件
        documents = create_demo_documents(article_text)
        
        # 建立向量資料庫
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name
        )
        logger.info("向量資料庫建立完成")
        
        # 建立 retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",  # 使用相似度搜尋
            search_kwargs={
                "k": 3,  # 返回前 k 個結果
                # 移除 score_threshold
            }
        )
        
        # 包裝 retriever 以顯示檢索結果
        def retriever_with_score(query):
            # 使用 retriever 進行查詢，改用 invoke 方法
            docs = retriever.invoke(query)
            print("\n=== 檢索結果 ===")
            
            # 使用 embeddings 計算實際相似度
            query_embedding = embeddings.embed_query(query)
            
            # 取得文件的 embeddings
            doc_embeddings = [embeddings.embed_documents([doc.page_content])[0] for doc in docs]
            
            # 計算餘弦相似度
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarities = [
                cosine_similarity(
                    np.array(query_embedding).reshape(1, -1),
                    np.array(doc_embedding).reshape(1, -1)
                )[0][0] for doc_embedding in doc_embeddings
            ]
            
            # 轉換為百分比
            similarity_percentages = [sim * 100 for sim in similarities]
            
            # 顯示結果
            for i, (doc, similarity) in enumerate(zip(docs, similarity_percentages), 1):
                print(f"\n文件 {i} (相似度: {similarity:.1f}%):")
                print(f"內容: {doc.page_content}")
                print(f"元數據: {doc.metadata}")
                print("-" * 50)
            
            return docs
        
       
        
        # 定義 prompt template
        template = """根據以下資訊回答問題。如果無法從資訊中找到答案，請說明無法回答。

                    相關資訊：
                            {context}

                    問題：{question}

                    回答："""

        prompt = ChatPromptTemplate.from_template(template)

        # 建立 RAG chain，使用包裝後的 retriever
        chain = (
            {"context": retriever_with_score, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        logger.info("RAG Chain 建立完成")

        return chain

    except Exception as e:
        logger.error(f"建立 RAG Chain 時發生錯誤: {str(e)}")
        raise

def main():
    """主程式：展示 RAG 基礎功能"""
    print("=== LangChain 0.3+ RAG 基礎展示 ===\n")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("請先設定 OPENAI_API_KEY 環境變數！")
        return

    try:
        """建立示範文件"""
        long_text = """
                台北市位於台灣北部，是台灣的首都及最大的都市，人口約250萬。

                作為政治、經濟、文化中心，台北擁有豐富的文化資產和現代化建設。

                台北市分為12個行政區，每個區都有其特色與風貌。

                例如，中正區匯聚了總統府、中正紀念堂等重要政府機構與歷史建築，是台灣政治心臟地帶；

                信義區則是繁華的商業中心，以高樓大廈、百貨公司及時尚品牌聞名，吸引無數國內外旅客前來購物與觀光。



                台北101是台北市最著名的地標建築，高度達509.2公尺，共101層樓，曾是世界最高建築。

                大樓內設有觀景台、購物中心和美食街，訪客可以在高空俯瞰整個台北市景。

                每年跨年時的煙火秀更是舉世聞名，璀璨的煙花照亮夜空，吸引成千上萬的觀光客和居民齊聚欣賞，已成為台北市年度最重要的盛事之一。



                台北的夜市文化聞名世界，是體驗在地生活的最佳場所。

                士林夜市是規模最大的夜市，以各式小吃、服飾、精品聞名，無論是香氣撲鼻的蚵仔煎、鹹香酥脆的雞排，還是獨具特色的士林大香腸，都讓人垂涎三尺，流連忘返。

                而饒河街夜市則以傳統美食及民俗小吃著稱，必吃的胡椒餅外皮香脆、內餡多汁，深受遊客喜愛。

                此外，寧夏夜市則是以傳統台灣小吃為主，被譽為「台北最道地的夜市」，讓人一站式品嚐牛肉湯、魷魚羹、芋圓等經典美味。



                台北市的交通便捷，大眾運輸系統發達，捷運系統包含多條路線，連接大台北地區，讓市民與遊客能夠輕鬆往來各大景點與商業區。

                台北車站作為交通樞紐，整合了捷運、台鐵、高鐵等多種運輸方式，使得往來台灣各地的交通更加便利。

                此外，公車路網密集且規劃完善，加上 YouBike 共享單車系統的推廣，讓台北成為一座綠色低碳的現代都市。



                除了現代化的都市風貌，台北市也擁有豐富的自然景觀。

                陽明山國家公園是都市人休閒放鬆的好去處，春天可賞櫻，秋冬則能欣賞壯麗的芒草美景。

                此外，北投地區因擁有豐富的溫泉資源而聞名，日式溫泉旅館林立，是寒冷冬季泡湯放鬆的絕佳選擇。

                而象山步道則是欣賞台北101與市區美景的熱門登山路線，短短的步道卻能讓人感受到大自然的清新與城市的壯麗交織。



                台北市也是文化藝術的重鎮，擁有眾多博物館與藝文中心。

                國立故宮博物院收藏了無數中國古代珍貴文物，如翠玉白菜、肉形石等國寶級藝術品，每年吸引大量國內外遊客參觀。

                松山文創園區與華山1914文化創意產業園區則是年輕人最愛的藝文展覽與市集活動據點，許多獨立設計師與手作藝術家在此展示他們的創意作品，讓台北市成為文創產業蓬勃發展的城市。



                此外，台北的節慶活動豐富多彩，不僅有著名的台北燈節、龍山寺農曆新年祈福活動，還有國際藝術節、電影節等文化盛事，每年都吸引大量民眾共襄盛舉。

                而台北市政府也積極推動國際活動，例如台北馬拉松、國際動漫展等，提升台北在國際間的能見度。



                整體而言，台北市融合了現代與傳統、都市與自然，無論是科技發展、交通便捷、文化藝術，還是美食與夜生活，皆展現出獨特的魅力，讓每一位來訪的旅人都能留下美好的回憶。

                這座充滿活力的城市，無論白天或夜晚，都有無數值得探索的故事與風景，等待著人們細細品味。
                """

        # 建立 RAG chain
        logger.info("開始建立 RAG Chain...")
        chain = create_rag_chain(long_text)
        
        # 測試問題
        questions = [
            # "台北101有多高？",
            # "台北有什麼著名的夜市？",
            # "台北的人口多少？",
            # "台北捷運和台北車站扮演什麼角色？",
            # "台北市有幾個行政區？",
            # "台北101每年有什麼特別活動？",
            # "陽明山有什麼特色？",  # 新增自然景觀相關問題
            "台北捷運什麼時候開始營運的？"  # 測試未知資訊
        ]
        
        logger.info("開始 RAG 問答測試...")
        print("-" * 50)
        
        for question in questions:
            print(f"\n問題：{question}")
            try:
                response = chain.invoke(question)
                print(f"\n最終回答：{response}")
                print("=" * 80)
            except Exception as e:
                logger.error(f"處理問題時發生錯誤: {str(e)}", exc_info=True)
                print(f"回答：處理問題時發生錯誤")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"執行主程式時發生錯誤: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

"""
CrewAI 旅遊規劃系統的代理定義
"""
from crewai import Agent
from textwrap import dedent
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI

# 導入工具系統
from tools import initialize_tools

"""
Creating Agents Cheat Sheet:
- Think like a boss. Work backwards from the goal and think which employee
  you need to hire to get the job done.
- Define the Captain of the crew who orient the other agents towards the goal.
- Define which experts the captain needs to communicate with and delegate tasks to.
  Build a top down structure of the crew.

Goal:
- Create a 7-day travel itinerary with detailed per-day plans, including budget, packing suggestions, and safety tips.

Captain/Manager/Boss:
- Name: Trip Planning Manager

Specialized Experts:
- Requirements Analyst
- Destination Selection Expert
- Itinerary Planning Expert
- Local Experience Expert
- Quality Assurance Expert
"""

# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py

class TripPlannerAgents:
    def __init__(self):
        """初始化旅行規劃代理系統"""
        # 設置語言模型，確保支持中文
        self.OpenAIGPT35 = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0.7
        )
        self.OpenAIGPT4 = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0.7
        )
        self.Ollama = OllamaLLM(model="openhermes")
        
        # 初始化工具註冊系統
        self.tool_registry = initialize_tools()

    def create_base_agent_template(self, role_name, backstory, goal, tools=None, llm=None):
        """創建基礎代理模板，處理語言一致性問題"""
        if tools is None:
            tools = []
        
        # 如果未指定 llm，使用默認的 GPT-4
        if llm is None:
            llm = self.OpenAIGPT4
        
        # 統一的語言處理策略
        language_instruction = """
        IMPORTANT SYSTEM FORMAT REQUIREMENTS:
        - All system keywords MUST remain in English: "Thought:", "Action:", "Action Input:", "Final Answer:"
        - Your actual thinking, analysis, and answers can be in Traditional Chinese
        
        思考內容請使用繁體中文，但關鍵系統標記必須保持英文。
        """
        
        return Agent(
            role=role_name,
            backstory=backstory,
            goal=goal,
            tools=tools,
            verbose=True,
            language_requirements=language_instruction,
            llm=llm
        )

    def trip_planner_manager(self):
        """創建旅行規劃管理者"""
        backstory = dedent(f"""我是一位資深旅行規劃協調專家，擁有超過15年的複雜旅行安排管理經驗。
                         我擅長協調不同旅行專家的工作，確保旅行的各個方面都得到妥善規劃和整合。
                         我的專長在於理解旅行者的需求並將任務分配給合適的專家。
                         我曾為各種客戶（從家庭旅行到企業團建活動）設計過數百個成功的旅行方案。""")
        
        goal = dedent(f"""通過以下方式協調整個旅行規劃過程：
                       1. 將任務分配給適當的專家
                       2. 確保行程涵蓋旅行的所有方面
                       3. 將不同專家的貢獻整合為一個連貫的計劃
                       4. 在整個規劃過程中保持質量和一致性
                       5. 確保最終方案滿足客戶的所有需求和期望""")
        
        tools = [
            self.tool_registry.get("search_internet"),
            self.tool_registry.get("calculate"),
            self.tool_registry.get("chinese_translation")
        ]
        
        # 使用模板創建代理
        agent = self.create_base_agent_template(
            role_name="Trip Planning Manager",
            backstory=backstory,
            goal=goal,
            tools=tools
        )
        
        # 啟用代理間委派
        agent.allow_delegation = True
        
        return agent

    def requirements_analyst(self):
        """創建需求分析專家，支持中文分析"""
        backstory = dedent(f"""我是一位旅行需求分析專家，擁有心理學和旅行規劃的雙重背景。
                         我擅長將客戶的表達需求轉化為具體的旅行參數，識別潛在的限制條件，
                         並創建全面的旅行需求分析。我曾在多家頂級旅行顧問公司工作，
                         擁有豐富的客戶需求挖掘和分析經驗。我能流利使用中文並理解中文文化背景。""")
        
        goal = dedent(f"""1. 全面分析旅行者的明確和隱含需求
                      2. 識別並評估各種旅行限制條件
                      3. 構建詳細的旅行者畫像
                      4. 提供明確的需求規格，為後續規劃奠定基礎
                      5. 確保所有分析都基於事實和數據
                      6. 能準確理解和處理中文搜索查詢和回應""")
        
        tools = [
            self.tool_registry.get("search_internet"),
            self.tool_registry.get("calculate")
        ]
        
        return self.create_base_agent_template(
            role_name="Travel Requirements Analyst",
            backstory=backstory,
            goal=goal,
            tools=tools
        )

    def destination_selection_expert(self):
        """創建目的地選擇專家"""
        backstory = dedent(f"""我是一位專精於目的地選擇的旅遊專家，擁有地理學和旅遊管理的專業背景。
                           我熟悉全球數千個目的地的特點、季節性和適合不同類型旅行者的特性。
                           我曾為旅行社和旅遊出版物撰寫目的地指南，並有豐富的個人旅行經歷。
                           我特別擅長配對旅行者偏好與最佳目的地選擇。""")
        
        goal = dedent(f"""1. 基於需求分析選擇最佳目的地
                       2. 評估目的地對特定旅行類型的適宜性
                       3. 考慮季節性因素和旅行時間
                       4. 平衡旅行者偏好與實際限制
                       5. 提供多個目的地選項並解釋選擇理由""")
        
        tools = [
            self.tool_registry.get("search_internet"),
            self.tool_registry.get("calculate")
        ]
        
        return self.create_base_agent_template(
            role_name="Destination Selection Expert",
            backstory=backstory,
            goal=goal,
            tools=tools
        )

    def itinerary_planning_expert(self):
        """創建行程規劃專家"""
        backstory = dedent(f"""我是一位精通行程規劃的專家，擁有10年以上的旅行行程設計經驗。
                           我擅長創建平衡景點參觀、活動和休息時間的最佳行程，確保旅行者充分體驗目的地同時不會感到疲憊。
                           我了解各種交通選擇、開放時間和預訂需求，可以創建無縫且有彈性的行程。
                           在我的職業生涯中，我為各種規模的個人和團體設計了數千個成功的行程。""")
        
        goal = dedent(f"""1. 創建詳細且可行的日程安排
                       2. 優化景點和活動的順序和時間
                       3. 平衡行程的密集度和靈活性
                       4. 考慮交通時間和選項
                       5. 確保行程與預算和旅行風格相符""")
        
        tools = [
            self.tool_registry.get("search_internet"),
            self.tool_registry.get("calculate")
        ]
        
        return self.create_base_agent_template(
            role_name="Itinerary Planning Expert",
            backstory=backstory,
            goal=goal,
            tools=tools
        )

    def local_experience_expert(self):
        """創建當地體驗專家"""
        backstory = dedent(f"""我曾在全球多個主要旅遊目的地擔任專業導遊超過十年。
                          我專注於創造超越典型旅遊景點的真實當地體驗。
                          我與當地社區有深厚的聯繫，了解隱藏的景點、地道的餐廳和獨特的文化活動。
                          我精通多種語言和當地風俗，能為旅行者提供獨到的內部視角。""")
        
        goal = dedent(f"""1. 用當地專業知識和內部見解增強旅行計劃
                      2. 提供地道當地體驗的詳細建議
                      3. 分享有關當地文化、習俗和禮儀的重要信息
                      4. 推薦遠離旅遊區的隱藏景點和活動
                      5. 提供實用的當地交通、安全和語言提示""")
        
        tools = [
            self.tool_registry.get("search_internet"),
            self.tool_registry.get("calculate")
        ]
        
        return self.create_base_agent_template(
            role_name="Local Experience Expert",
            backstory=backstory,
            goal=goal,
            tools=tools
        )
        
    def quality_assurance_expert(self):
        """創建質量保證專家"""
        backstory = dedent(f"""我是一位旅行計劃質量控制專家，擁有豐富的旅行規劃審核和優化經驗。
                         我曾在頂級旅行公司擔任質量審核經理，負責確保所有旅行方案的可行性、合理性和高質量。
                         我擅長識別潛在問題、優化行程和提高整體旅行體驗。
                         我對各類目的地都有深入了解，包括常見陷阱和最佳實踐。""")
        
        goal = dedent(f"""1. 全面審核旅行計劃的可行性和質量
                      2. 識別並解決潛在問題和衝突
                      3. 優化行程時間安排和資源分配
                      4. 確保預算分配合理且有應急準備
                      5. 提供具體的改進建議和替代方案
                      6. 增強旅行計劃的整體體驗和價值""")
        
        tools = [
            self.tool_registry.get("search_internet"),
            self.tool_registry.get("calculate"),
            self.tool_registry.get("advanced_calculate")
        ]
        
        return self.create_base_agent_template(
            role_name="Travel Plan Quality Assurance Expert",
            backstory=backstory,
            goal=goal,
            tools=tools,
            llm=self.OpenAIGPT35
        )

    def chinese_translation_expert(self):
        """創建中文翻譯專家代理"""
        backstory = dedent("""
            您是一位專業的中英翻譯專家，擁有超過20年的旅遊文檔翻譯經驗。
            您熟悉台灣與中文世界的旅遊術語和表達方式，能將任何旅行計劃轉換為道地的繁體中文。
            您特別注重保持原文的結構和格式，同時確保譯文流暢自然。
            您曾為多家國際旅行社和旅遊平台提供專業翻譯服務，熟悉旅遊行業的專業術語。
        """)
        
        goal = dedent("""
            將英文或混合語言的旅行計劃完整轉換為純繁體中文，同時：
            1. 保持原始文檔的結構與格式
            2. 確保專有名詞、地名和數字的準確性
            3. 使用道地的繁體中文表達方式
            4. 調整表達以符合台灣本地旅遊用語習慣
            5. 保持內容的完整性，不遺漏任何信息
        """)
        
        tools = [self.tool_registry.get("chinese_translation")]
        
        # 添加工具使用指導
        tool_guidance = """
        重要工具使用指南：
        
        使用 Chinese Translation 工具時，請遵循以下流程：
        
        1. 仔細分析原始文檔的結構和格式（標題、小標題、列表、段落等）
        2. 使用以下格式傳遞參數：
           Action: Chinese Translation
           Action Input: {"input_text": "完整的待翻譯文本，包含所有格式標記"}
        3. 檢查翻譯結果是否保留了原始文檔的所有格式和結構
        4. 如果格式有丟失，將文檔分段傳遞並在最後重新組合
        
        重要：保持所有 Markdown 格式標記（#、-、*等）不變，只翻譯實際內容。
        """
        
        agent = self.create_base_agent_template(
            role_name="Chinese Translation Expert",
            backstory=backstory + tool_guidance,  # 將工具指南添加到背景故事
            goal=goal,
            tools=tools,
            llm=self.OpenAIGPT35
        )
        
        return agent


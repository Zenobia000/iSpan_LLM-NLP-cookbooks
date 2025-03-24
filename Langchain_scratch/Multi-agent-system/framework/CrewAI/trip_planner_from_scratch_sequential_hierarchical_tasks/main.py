import os
import shutil
from pathlib import Path
from crewai import Crew, Process
from textwrap import dedent
from agents import TripPlannerAgents
from tasks import TripPlannerTasks
from dotenv import load_dotenv
from tools.output_interceptor import ChineseOutputInterceptor  # 導入攔截器

# 設置輸出攔截器 - 將 Unicode 編碼轉換為可讀中文
interceptor = ChineseOutputInterceptor.setup()

# 清理 CrewAI 快取（僅運行一次）
cache_dir = Path.home() / ".crewai"
if cache_dir.exists():
    try:
        shutil.rmtree(cache_dir)
        print(f"已刪除 CrewAI 快取目錄：{cache_dir}")
    except Exception as e:
        print(f"無法刪除快取目錄：{e}")

# 或者，設置不同的資料目錄
os.environ["CREWAI_DATADIR"] = "D:\python_workspace\Sunny_NLP_Learning\crewai_data"  # 請修改為您有足夠空間的位置

# 加載 .env 文件中的環境變量
load_dotenv()

# 檢查環境變量是否存在，若不存在則提示用戶
if not os.environ.get("OPENAI_API_KEY"):
    print("警告: 未設置 OPENAI_API_KEY 環境變量。請確保在 .env 文件中設置或直接在環境中設置。")
    api_key = input("請輸入您的 OpenAI API 密鑰: ")
    os.environ["OPENAI_API_KEY"] = api_key

class TripPlannerCrew:
    def __init__(self, destination_type, budget, duration, interests, currency="USD"):
        """初始化旅行規劃團隊"""
        self.destination_type = destination_type
        self.budget = budget
        self.duration = duration
        self.interests = interests
        self.currency = currency  # 添加幣值選項，預設為 USD
        
        # 初始化代理和任務
        self.agents = TripPlannerAgents()
        self.tasks = TripPlannerTasks()
        
    def run(self):
        """執行旅行規劃流程"""
        # 初始化代理和任務
        agents = self.agents
        tasks = self.tasks

        # 創建代理實例
        manager = agents.trip_planner_manager()
        requirements_analyst = agents.requirements_analyst()
        destination_expert = agents.destination_selection_expert()
        itinerary_planner = agents.itinerary_planning_expert()
        local_guide = agents.local_experience_expert()
        quality_controller = agents.quality_assurance_expert()
        
        # 添加翻譯專家
        # translation_expert = agents.chinese_translation_expert()
        
        # 創建任務
        analyze_requirements_task = tasks.analyze_requirements(
            agent=requirements_analyst,
            destination_type=self.destination_type,
            budget=self.budget,
            duration=self.duration,
            interests=self.interests,
            currency=self.currency
        )
        
        select_destination_task = tasks.select_destination(
            agent=destination_expert,
            destination_type=self.destination_type,
            budget=self.budget,
            duration=self.duration,
            requirements_analysis="{{analyze_requirements_task.output}}",
            currency=self.currency
        )
        
        create_itinerary_task = tasks.create_itinerary(
            agent=itinerary_planner,
            destination="{{select_destination_task.output}}",
            duration=self.duration,
            interests=self.interests,
            budget=self.budget,
            currency=self.currency
        )
        
        enhance_local_experience_task = tasks.provide_local_insights(
            agent=local_guide,
            destination="{{select_destination_task.output}}",
            itinerary="{{create_itinerary_task.output}}",
            interests=self.interests
        )
        
        optimize_plan_task = tasks.optimize_plan(
            agent=quality_controller,
            initial_itinerary="{{create_itinerary_task.output}}",
            initial_insights="{{enhance_local_experience_task.output}}",
            budget=self.budget,
            duration=self.duration,
            currency=self.currency
        )
        
        final_plan_task = tasks.coordinate_planning(
            agent=manager,
            destination="{{select_destination_task.output}}",
            itinerary="{{create_itinerary_task.output}}",
            insights="{{enhance_local_experience_task.output}}",
            optimization="{{optimize_plan_task.output}}",
            budget=self.budget,
            duration=self.duration,
            currency=self.currency
        )
        
        # 將所有任務添加到執行列表
        execution_tasks = [
            analyze_requirements_task, 
            select_destination_task, 
            create_itinerary_task, 
            enhance_local_experience_task, 
            optimize_plan_task, 
            final_plan_task
        ]
        
        # 創建 crew 實例
        crew = Crew(
            agents=[
                manager, 
                requirements_analyst, 
                destination_expert, 
                itinerary_planner, 
                local_guide, 
                quality_controller,
                # translation_expert  # 添加翻譯專家
            ],
            tasks=execution_tasks,
            verbose=True,
            process=Process.sequential,
            memory=False  # 禁用持久化記憶體存儲
        )
        
        # 執行任務並獲取結果
        initial_result = crew.kickoff()
        
        # # 添加翻譯任務處理
        # translate_task = tasks.translate_final_plan(
        #     agent=translation_expert,
        #     plan_document=initial_result
        # )
        
        # # 創建只包含翻譯任務的 crew
        # translation_crew = Crew(
        #     agents=[translation_expert],
        #     tasks=[translate_task],
        #     verbose=True
        # )
        
        # # 執行翻譯並獲取最終結果
        # final_result = translation_crew.kickoff()
        
        # 返回翻譯後的結果
        return self._format_final_output(initial_result)
        # return self._format_final_output(final_result)
    
    def _format_final_output(self, plan):
        """格式化最終輸出，確保使用繁體中文"""
        # 繁體中文格式已由翻譯專家處理，這裡僅添加元數據
        
        # 根據幣值選擇顯示適當的符號
        currency_symbol = "NT$" if self.currency == "TWD" else "$"
        
        return f"""
########################
## 以下是您的旅行行程:
########################

{plan}

## 旅行規劃元數據
- 目的地類型: {self.destination_type}
- 預算: {currency_symbol}{self.budget} {self.currency}
- 行程天數: {self.duration}天
- 興趣: {', '.join(self.interests) if isinstance(self.interests, list) else self.interests}
"""

# 主函數，用於執行旅行規劃 crew
if __name__ == "__main__":
    print("## 歡迎使用旅行行程規劃系統")
    print("-------------------------------")
    
    # 獲取用戶輸入
    destination_type = input(dedent("""您對什麼類型的目的地感興趣？(例如：海灘、城市、山脈、文化): """))
    
    # 幣值選擇
    currency_choice = input(dedent("""您希望使用哪種貨幣？(USD/TWD，預設為USD): """)).strip().upper()
    currency = "TWD" if currency_choice == "TWD" else "USD"
    currency_symbol = "NT$" if currency == "TWD" else "$"
    
    # 預算輸入
    budget = input(dedent(f"""您整個旅行的預算是多少？({currency_symbol}，{currency}): """))
    
    duration = input(dedent("""您的行程將持續多少天？: """))
    interests = input(dedent("""您的主要興趣是什麼？請列出所有興趣，用逗號分隔(例如：歷史、美食、冒險、放鬆): """))
    
    # 處理interests，將其轉換為列表
    interests_list = [interest.strip() for interest in interests.split(',')]

    # 創建旅行規劃 crew 實例
    trip_planner = TripPlannerCrew(
        destination_type=destination_type,
        budget=budget,
        duration=duration,
        interests=interests_list,
        currency=currency
    )
    
    # 運行旅行規劃流程並獲取結果
    result = trip_planner.run()
    
    # 輸出結果
    print(result)


    

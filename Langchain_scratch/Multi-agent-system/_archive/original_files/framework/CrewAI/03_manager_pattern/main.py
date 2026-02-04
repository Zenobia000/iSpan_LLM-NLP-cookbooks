import os
from crewai import Crew, Process
from textwrap import dedent
from agents import TripPlannerAgents
from tasks import TripPlannerTasks
from dotenv import load_dotenv

# 加載 .env 文件中的環境變量
load_dotenv()

# 檢查環境變量是否存在，若不存在則提示用戶
if not os.environ.get("OPENAI_API_KEY"):
    print("警告: 未設置 OPENAI_API_KEY 環境變量。請確保在 .env 文件中設置或直接在環境中設置。")
    api_key = input("請輸入您的 OpenAI API 密鑰: ")
    os.environ["OPENAI_API_KEY"] = api_key

class TripPlannerCrew:
    def __init__(self, destination_type, budget, duration, interests):
        """初始化旅行規劃團隊"""
        self.destination_type = destination_type
        self.budget = budget
        self.duration = duration
        self.interests = interests
        
        # 初始化代理和任務
        self.agents = TripPlannerAgents()
        self.tasks = TripPlannerTasks()
        
    def run(self):
        """運行旅行規劃流程 - 完整的多代理協作模式"""
        # 初始化所有代理
        manager = self.agents.trip_planner_manager()
        city_expert = self.agents.city_selection_expert()
        travel_agent = self.agents.expert_travel_agent()
        local_guide = self.agents.local_tour_guide()
        
        # 初始化所有任務，並建立任務間的依賴關係
        select_destination_task = self.tasks.select_destination(
            agent=city_expert,
            destination_type=self.destination_type,
            budget=self.budget,
            duration=self.duration
        )
        
        # 創建行程需要依賴目的地選擇的結果
        create_itinerary_task = self.tasks.create_itinerary(
            agent=travel_agent,
            destination="{{select_destination_task.output}}",
            duration=self.duration,
            interests=self.interests
        )
        
        # 提供當地見解需要依賴目的地選擇的結果
        provide_insights_task = self.tasks.provide_local_insights(
            agent=local_guide,
            destination="{{select_destination_task.output}}"
        )
        
        # 最終整合由管理者完成
        coordinate_task = self.tasks.coordinate_planning(
            agent=manager,
            destination="{{select_destination_task.output}}",
            itinerary="{{create_itinerary_task.output}}",
            insights="{{provide_insights_task.output}}",
            budget=self.budget,
            duration=self.duration
        )
        
        # 創建包含所有代理的Crew
        crew = Crew(
            agents=[manager, city_expert, travel_agent, local_guide],
            tasks=[select_destination_task, create_itinerary_task, provide_insights_task, coordinate_task],
            verbose=True,
            process=Process.sequential  # 確保任務按順序執行，以處理依賴關係
        )
        
        # 執行整個流程
        result = crew.kickoff()
        
        # 根據 CrewAI 最新版本的 CrewOutput 格式提取結果
        try:
            # 嘗試使用新版 CrewAI API 獲取結果
            if hasattr(result, 'raw'):
                final_plan = result.raw  # 某些版本使用 raw 屬性
            elif hasattr(result, 'outputs') and len(result.outputs) > 0:
                final_plan = result.outputs[-1]  # 獲取最後一個輸出
            elif hasattr(result, 'output'):
                final_plan = result.output  # 直接獲取輸出
            elif isinstance(result, str):
                final_plan = result  # 結果本身就是字符串
            else:
                # 如果都不成功，將結果轉為字符串
                final_plan = str(result)
        except Exception as e:
            print(f"獲取結果時出現錯誤: {e}")
            final_plan = "無法提取最終計劃，但所有任務已執行完成。請檢查控制台輸出以獲取詳細信息。"
              
        # 返回最終計劃
        return self._format_final_output(final_plan)
    
    def _format_final_output(self, plan):
        """格式化最終輸出"""
        return f"""
# 旅行計劃摘要

{plan}

## 旅行規劃元數據
- 目的地類型: {self.destination_type}
- 預算: ${self.budget}
- 行程天數: {self.duration}天
- 興趣: {', '.join(self.interests) if isinstance(self.interests, list) else self.interests}
"""


# 主函數，用於執行旅行規劃 crew
if __name__ == "__main__":
    print("## Welcome to Travel Itinerary Planner")
    print("-------------------------------")
    
    # 獲取用戶輸入
    destination_type = input(dedent("""What type of destination are you interested in? (e.g., beach, city, mountains, cultural): """))
    budget = input(dedent("""What's your budget for the entire trip? (in USD): """))
    duration = input(dedent("""How many days will your trip last?: """))
    interests = input(dedent("""What are your main interests? list all of them, separated by commas (e.g., history, food, adventure, relaxation): """))
    
    # 處理interests，將其轉換為列表
    interests_list = [interest.strip() for interest in interests.split(',')]

    # 創建旅行規劃 crew 實例
    trip_planner = TripPlannerCrew(
        destination_type=destination_type,
        budget=budget,
        duration=duration,
        interests=interests_list
    )
    
    # 運行旅行規劃流程並獲取結果
    result = trip_planner.run()
    
    # 輸出結果
    print("\n\n########################")
    print("## Here is your travel itinerary:")
    print("########################\n")
    print(result)


    

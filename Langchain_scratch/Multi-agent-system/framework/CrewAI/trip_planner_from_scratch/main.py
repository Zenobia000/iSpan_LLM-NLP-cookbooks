import os
from crewai import Crew
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
        self.destination_type = destination_type
        self.budget = budget
        self.duration = duration
        self.interests = interests

    def run(self):
        # 初始化代理和任務
        agents = TripPlannerAgents()
        tasks = TripPlannerTasks()

        # 創建代理實例  
        city_expert = agents.city_selection_expert()
        travel_agent = agents.expert_travel_agent()
        local_guide = agents.local_tour_guide()

        # 定義任務
        select_destination_task = tasks.select_destination(
            agent=city_expert,
            destination_type=self.destination_type,
            budget=self.budget,
            duration=self.duration
        )       
        
        create_itinerary_task = tasks.create_itinerary(     
            agent=travel_agent,     
            destination="{{select_destination_task.output}}"        ,  # 使用前一個任務的輸出
            duration=self.duration,     
            interests=self.interests        
        )       
        
        provide_insights_task = tasks.provide_local_insights        (
            agent=local_guide,      
            destination="{{select_destination_task.output}}"          # 使用前一個任務的輸出
        )       
        
        # 定義並啟動 crew
        crew = Crew(
            agents=[city_expert, travel_agent, local_guide],
            tasks=[select_destination_task, create_itinerary_task, provide_insights_task],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# 主函數，用於執行旅行規劃 crew
if __name__ == "__main__":
    print("## Welcome to Travel Itinerary Planner")
    print("-------------------------------")
    
    # 獲取用戶輸入
    destination_type = input(dedent("""What type of destination are you interested in? (e.g., beach, city, mountains, cultural): """))
    budget = input(dedent("""What's your budget for the entire trip? (in USD): """))
    duration = input(dedent("""How many days will your trip last?: """))
    interests = input(dedent("""What are your main interests? (e.g., history, food, adventure, relaxation): """))

    # 創建旅行規劃 crew 實例
    trip_planner = TripPlannerCrew(
        destination_type=destination_type,
        budget=budget,
        duration=duration,
        interests=interests
    )
    
    # 運行旅行規劃流程並獲取結果
    result = trip_planner.run()
    
    # 輸出結果
    print("\n\n########################")
    print("## Here is your travel itinerary:")
    print("########################\n")
    print(result)

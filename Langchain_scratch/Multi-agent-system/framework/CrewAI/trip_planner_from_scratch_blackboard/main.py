import os
from crewai import Crew
from textwrap import dedent
from agents import TripPlannerAgents
from tasks import TripPlannerTasks
from tools.blackboard import BlackboardManager
from tools.result_aggregator import ResultAggregator
from dotenv import load_dotenv

# 加載 .env 文件中的環境變量
load_dotenv()

# 檢查環境變量是否存在，若不存在則提示用戶
if not os.environ.get("OPENAI_API_KEY"):
    print("警告: 未設置 OPENAI_API_KEY 環境變量。請確保在 .env 文件中設置或直接在環境中設置。")
    api_key = input("請輸入您的 OpenAI API 密鑰: ")
    os.environ["OPENAI_API_KEY"] = api_key

class TripPlannerCrew:
    def __init__(self):
        self.agents = TripPlannerAgents()
        self.tasks = TripPlannerTasks()
        self.blackboard = BlackboardManager()
        self.result_aggregator = ResultAggregator()

    def plan_trip(self, destination_type, budget, duration, interests):
        """
        使用中央調度的方式規劃旅程
        
        Args:
            destination_type (str): 目的地類型 (e.g., beach, city, mountains)
            budget (int): 預算金額
            duration (int): 旅程天數
            interests (list): 興趣列表
        
        Returns:
            dict: 完整的旅行計劃，包含目的地、行程和在地建議
        """
        # 初始化所有代理人
        manager = self.agents.trip_planner_manager()
        city_expert = self.agents.city_selection_expert()
        travel_expert = self.agents.expert_travel_agent()
        local_guide = self.agents.local_tour_guide()

        # 第一階段：需求分析與初步規劃
        initial_tasks = [
            self.tasks.analyze_requirements(manager, destination_type, budget, duration, interests)
        ]
        
        planning_crew = Crew(
            agents=[manager, city_expert, travel_expert, local_guide],
            tasks=initial_tasks,
            process="sequential",
            verbose=True
        )
        initial_results = planning_crew.kickoff()
        
        # 將初步分析結果寫入黑板
        result_keys = ['requirements']
        for i, (task, agent) in enumerate(zip(initial_tasks, [manager])):
            try:
                result = initial_results[task.description]
            except (KeyError, AttributeError):
                result = "No result available"
                
            self.blackboard.write(
                key=f"initial_{result_keys[i]}",
                value=result,
                agent_id=agent.role
            )
            self.result_aggregator.add_result(
                key=f"initial_{result_keys[i]}",
                value=result,
                agent_id=agent.role
            )

        # 第二階段：目的地選擇與評估
        destination_tasks = [
            self.tasks.research_destinations(city_expert, destination_type, budget)
        ]
        
        destination_crew = Crew(
            agents=[city_expert, travel_expert, local_guide, manager],
            tasks=destination_tasks,
            process="sequential",
            verbose=True
        )
        destination_results = destination_crew.kickoff()
        
        # 將目的地評估結果寫入黑板
        result_keys = ['destinations']
        for i, (task, agent) in enumerate(zip(destination_tasks, [city_expert])):
            try:
                result = destination_results[task.description]
            except (KeyError, AttributeError):
                result = "No result available"
                
            self.blackboard.write(
                key=f"destination_{result_keys[i]}",
                value=result,
                agent_id=agent.role
            )
            self.result_aggregator.add_result(
                key=f"destination_{result_keys[i]}",
                value=result,
                agent_id=agent.role
            )

        # 第三階段：行程規劃與在地體驗整合
        selected_destination = self.blackboard.read("destination_destinations")
        planning_tasks = [
            self.tasks.provide_local_insights(local_guide, selected_destination)
        ]
        
        final_crew = Crew(
            agents=[manager, travel_expert, local_guide, city_expert],
            tasks=planning_tasks,
            process="sequential",
            verbose=True
        )
        final_results = final_crew.kickoff()
        
        # 將規劃結果寫入黑板
        result_keys = ['insights']
        for i, (task, agent) in enumerate(zip(planning_tasks, [local_guide])):
            try:
                result = final_results[task.description]
            except (KeyError, AttributeError):
                result = "No result available"
                
            self.blackboard.write(
                key=f"planning_{result_keys[i]}",
                value=result,
                agent_id=agent.role
            )
            self.result_aggregator.add_result(
                key=f"planning_{result_keys[i]}",
                value=result,
                agent_id=agent.role
            )
        
        # 第四階段：最終整合與優化
        optimization_task = self.tasks.optimize_plan(
            manager,
            self.blackboard.read("planning_routes"),
            self.blackboard.read("planning_activities"),
            self.blackboard.read("planning_insights"),
            self.blackboard.read("planning_budget"),
            budget,
            duration
        )
        
        optimization_crew = Crew(
            agents=[manager, travel_expert, local_guide, city_expert],
            tasks=[optimization_task],
            process="sequential",
            verbose=True
        )
        optimized_result = optimization_crew.kickoff()
        
        # 獲取優化結果
        try:
            optimization_output = optimized_result[optimization_task.description]
            optimization_summary = optimization_output.get("optimization_summary", {})
            critical_issues = optimization_summary.get("issues", {}).get("critical", [])
        except (KeyError, AttributeError):
            optimization_output = {}
            optimization_summary = {}
            critical_issues = []
        
        if critical_issues:
            print("\nDetected critical issues that need attention:")
            for issue in critical_issues:
                print(f"- {issue}")
            
            # 根據問題類型重新執行相應階段
            for issue in critical_issues:
                if "destination" in issue.lower():
                    # 重新執行目的地選擇
                    print("\nRe-evaluating destination selection...")
                    destination_task = self.tasks.research_destinations(city_expert, destination_type, budget)
                    destination_crew = Crew(
                        agents=[city_expert, manager, travel_expert],
                        tasks=[destination_task],
                        process="sequential",
                        verbose=True
                    )
                    new_destination_results = destination_crew.kickoff()
                    result = new_destination_results.get(destination_task.description, "No result available")
                    self.blackboard.write(
                        key="destination_destinations",
                        value=result,
                        agent_id=city_expert.role
                    )
                    self.result_aggregator.add_result(
                        key="destination_destinations",
                        value=result,
                        agent_id=city_expert.role
                    )
                    
                elif "itinerary" in issue.lower():
                    # 重新規劃行程
                    print("\nRe-planning itinerary...")
                    selected_destination = self.blackboard.read("destination_destinations")
                    planning_tasks = [
                        self.tasks.design_route_optimization(travel_expert, selected_destination, duration),
                        self.tasks.create_activity_schedule(city_expert, selected_destination, interests),
                        self.tasks.provide_local_insights(local_guide, selected_destination),
                        self.tasks.monitor_budget_compliance(
                            manager,
                            self.blackboard.read("destination_transportation"),
                            budget
                        )
                    ]
                    final_crew = Crew(
                        agents=[manager, travel_expert, local_guide, city_expert],
                        tasks=planning_tasks,
                        process="sequential",
                        verbose=True
                    )
                    new_results = final_crew.kickoff()
                    
                    result_keys = ['routes', 'activities', 'insights', 'budget']
                    for i, (task, agent) in enumerate(zip(planning_tasks, [travel_expert, city_expert, local_guide, manager])):
                        result = new_results.get(task.description, "No result available")
                        self.blackboard.write(
                            key=f"planning_{result_keys[i]}",
                            value=result,
                            agent_id=agent.role
                        )
                        self.result_aggregator.add_result(
                            key=f"planning_{result_keys[i]}",
                            value=result,
                            agent_id=agent.role
                        )
                    
                elif "local" in issue.lower():
                    # 重新獲取本地建議
                    print("\nUpdating local insights...")
                    selected_destination = self.blackboard.read("destination_destinations")
                    local_tasks = [
                        self.tasks.create_activity_schedule(city_expert, selected_destination, interests),
                        self.tasks.provide_local_insights(local_guide, selected_destination)
                    ]
                    local_crew = Crew(
                        agents=[city_expert, local_guide],
                        tasks=local_tasks,
                        process="sequential",
                        verbose=True
                    )
                    new_local_results = local_crew.kickoff()
                    
                    for i, (task, agent, key) in enumerate(zip(local_tasks, [city_expert, local_guide], ['activities', 'insights'])):
                        result = new_local_results.get(task.description, "No result available")
                        self.blackboard.write(
                            key=f"planning_{key}",
                            value=result,
                            agent_id=agent.role
                        )
                        self.result_aggregator.add_result(
                            key=f"planning_{key}",
                            value=result,
                            agent_id=agent.role
                        )
            
            # 再次進行最終優化
            print("\nPerforming final optimization after addressing critical issues...")
            optimization_task = self.tasks.optimize_plan(
                manager,
                self.blackboard.read("planning_routes"),
                self.blackboard.read("planning_activities"),
                self.blackboard.read("planning_insights"),
                self.blackboard.read("planning_budget"),
                budget,
                duration
            )
            optimization_crew = Crew(
                agents=[manager, travel_expert, local_guide, city_expert],
                tasks=[optimization_task],
                process="sequential",
                verbose=True
            )
            optimized_result = optimization_crew.kickoff()
            optimization_output = optimized_result.get(optimization_task.description, {})
        
        # 將最終優化結果寫入黑板和聚合器
        result_keys = ['routes', 'activities', 'insights', 'budget']
        final_results = {
            key: self.blackboard.read(f"planning_{key}")
            for key in result_keys
        }
        
        for key in result_keys:
            optimized_value = optimization_output.get(f"optimized_{key}", final_results[key])
            self.blackboard.write(
                key=f"planning_{key}",
                value=optimized_value,
                agent_id=manager.role
            )
            self.result_aggregator.add_result(
                key=f"planning_{key}",
                value=optimized_value,
                agent_id=manager.role
            )
            
        self.result_aggregator.add_result(
            key="optimization_summary",
            value=optimization_summary,
            agent_id=manager.role
        )
        
        # 返回完整的旅行計劃
        return {
            # 初始階段資訊
            "initial_analysis": {
                "requirements": self.blackboard.read("initial_requirements"),
                "feasibility": self.blackboard.read("initial_feasibility"),
                "constraints": self.blackboard.read("initial_constraints"),
                "experiences": self.blackboard.read("initial_experiences")
            },
            # 目的地評估資訊
            "destination_evaluation": {
                "selected_destinations": self.blackboard.read("destination_destinations"),
                "transportation_options": self.blackboard.read("destination_transportation"),
                "seasonal_considerations": self.blackboard.read("destination_seasonal")
            },
            # 詳細規劃資訊
            "detailed_planning": {
                "routes": self.blackboard.read("planning_routes"),
                "activities": self.blackboard.read("planning_activities"),
                "local_insights": self.blackboard.read("planning_insights"),
                "budget_allocation": self.blackboard.read("planning_budget")
            },
            # 優化結果
            "optimization_results": {
                "summary": self.result_aggregator.get_result("optimization_summary"),
                "optimized_routes": self.blackboard.read("planning_routes"),
                "optimized_activities": self.blackboard.read("planning_activities"),
                "optimized_insights": self.blackboard.read("planning_insights"),
                "optimized_budget": self.blackboard.read("planning_budget")
            },
            # 完整歷史記錄
            "planning_history": self.blackboard.get_history()
        }


# 主函數，用於執行旅行規劃 crew
if __name__ == "__main__":
    print("## Welcome to Travel Itinerary Planner")
    print("-------------------------------")
    
    # 獲取用戶輸入
    destination_type = input(dedent("""What type of destination are you interested in? (e.g., beach, city, mountains, cultural): """))
    budget = int(input(dedent("""What's your budget for the entire trip? (in USD): """)))
    duration = int(input(dedent("""How many days will your trip last?: """)))
    interests = input(dedent("""What are your main interests? (comma separated, e.g., history, food, adventure): """)).split(",")
    interests = [interest.strip() for interest in interests]  # 清理每個興趣的空白

    # 創建旅行規劃 crew 實例
    crew = TripPlannerCrew()
    
    # 運行旅行規劃流程並獲取結果
    trip_plan = crew.plan_trip(
        destination_type=destination_type,
        budget=budget,
        duration=duration,
        interests=interests
    )
    
    # 輸出結果
    print("\n\n########################")
    print("## Final Travel Plan")
    print("########################\n")
    
    # 獲取優化結果
    optimization_results = trip_plan["optimization_results"]
    
    print("1. Optimization Summary:")
    print("----------------------")
    print(optimization_results["summary"])
    
    print("\n2. Optimized Routes:")
    print("------------------")
    print(optimization_results["optimized_routes"])
    
    print("\n3. Optimized Activities:")
    print("----------------------")
    print(optimization_results["optimized_activities"])
    
    print("\n4. Local Insights:")
    print("---------------")
    print(optimization_results["optimized_insights"])
    
    print("\n5. Budget Allocation:")
    print("------------------")
    print(optimization_results["optimized_budget"])


    

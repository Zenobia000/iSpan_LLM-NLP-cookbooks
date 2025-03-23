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
- Name: Expert Travel Agent

Employees/Experts to hire:
- City Selection Expert
- Local Tour Guide


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should be actionable
- Backstory should be their resume

"""

# This is an example of how to define custom agents.
# You can define as many agents as you want.
# You can also define custom tasks in tasks.py


class TripPlannerAgents:
    def __init__(self):
        """初始化旅行規劃代理系統"""
        # 設置語言模型
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = OllamaLLM(model="openhermes")
        
        # 初始化工具註冊系統
        self.tool_registry = initialize_tools()

    def expert_travel_agent(self):
        """創建專家旅行代理"""
        return Agent(
            role="Expert Travel Agent",
            backstory=dedent(f"""Expert in travel planning and logistics.
                             I have a deep understanding of the best travel destinations,
                             accommodations, and activities for any given budget and travel style."""),
                             
            goal=dedent(f"""Create a 7-day travel itinerary with detailed per-day plans
                             , including budget, packing suggestions, and safety tips."""),
            tools=[
                self.tool_registry.get("calculate"),
                self.tool_registry.get("advanced_calculate"),
                self.tool_registry.get("search_internet")
            ],
            allow_delegation=True,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    
    def city_selection_expert(self):
        """創建城市選擇專家代理"""
        return Agent(
            role="City Selection Expert",
            backstory=dedent(f"""I am a destination research specialist with over 15 years of experience
                             analyzing travel destinations worldwide. I have visited over 100 countries
                             and specialize in matching travelers with destinations based on their preferences,
                             budget constraints, and travel goals. I stay updated on global travel trends,
                             safety conditions, and seasonal attractions for all major destinations."""),
            goal=dedent(f"""Recommend optimal travel destinations based on travelers' preferences,
                          budget, and timing. Provide detailed information about local attractions,
                          cultural highlights, best times to visit, and practical logistics
                          to help create the perfect travel itinerary."""),
            tools=[self.tool_registry.get("search_internet")],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def local_tour_guide(self):
        """創建當地導遊代理"""
        return Agent(
            role="Local Tour Guide",
            backstory=dedent(f"""I have worked as a professional tour guide in multiple major 
                              tourist destinations for over a decade. I specialize in creating 
                              authentic local experiences that go beyond typical tourist attractions.
                              I have extensive knowledge of local customs, hidden gems, transportation
                              systems, and safety considerations in various destinations."""),
            goal=dedent(f"""Provide detailed day-by-day activity plans for selected destinations, 
                          including insider tips on local attractions, authentic dining experiences,
                          transportation options, and cultural etiquette. Ensure travelers get the
                          most enriching and safe experience at each location."""),
            tools=[
                self.tool_registry.get("search_internet"),
                self.tool_registry.get("calculate")
            ],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )


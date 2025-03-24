# To know more about the Task class, visit: https://docs.crewai.com/concepts/tasks
from crewai import Task
from textwrap import dedent

"""
Creating Tasks Cheat Sheet:
👉Begin with the end in mind. Identify the specific outcome your tasks are aiming to achieve.
- Break down the outcome into actionable tasks, assigning each task to the appropriate agent.
- Ensure tasks are descriptive, providing clear instructions and expected deliverables.

Goal:
- Develop a detailed itinerary, including city selection, attractions, and practical travel advice.

Key Steps for Task Creation:
1. Identify the Desired Outcome: Define what success looks like for your project.
    - A detailed itinerary, including city selection, attractions, and practical travel advice.

2. Task Breakdown: Divide the goal into smaller, manageable tasks that agents can execute.
    - Itinerary Planning: develop a detailed plan for each day of the trip.
    - City Selection: Analyze and pick the best cities to visit.
    - Local Tour Guide: Find a local expert to provide insights and recommendations.


3. Assign Tasks to Agents: Match tasks with agents based on their roles and expertise.
    - Itinerary Planning -> Expert Travel Agent
    - City Selection -> City Selection Expert
    - Local Tour Guide -> Local Tour Guide

4. Task Description Template:
   - Use this template as a guide to define each task in your CrewAI application.
   - This template helps ensure that each task is clearly defined, actionable, and aligned with the specific
     goals of your project.

Template:
----------
def [task_name](self, agent, [parameters]):
    return Task(description=dedent(f'''
    **Task**: [Provide a concise name or summary of the task.]
    **Description**: [Detailed description of what the agent is expected to do, including actionable steps and expected outcomes.]
    
    **Parameters**:
    - [Parameter 1]: [Description]
    - [Parameter 2]: [Description]
    ... [Add more parameters as needed.]
    
    **Note**: [Optional section for incentives or encouragement for high-quality work. This can include tips, additional information, etc.]
    '''), agent=agent)
"""


class TripPlannerTasks:
    def __tip_section(self):
        return "If you do your BEST WORK, I'll give you a $10,000 commission!"
    
    def coordinate_planning(self, agent, destination_type, budget, duration, interests):
        """管理者的協調任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Coordinate the entire travel planning process
                **Description**: As the Trip Planning Manager, you need to coordinate the planning process and ensure all aspects
                of the trip are properly covered. You should:
                
                1. Review the initial requirements:
                   - Destination Type: {destination_type}
                   - Budget: ${budget}
                   - Duration: {duration} days
                   - Interests: {interests}
                
                2. Delegate and oversee the following tasks:
                   - Destination selection to the City Selection Expert
                   - Itinerary creation to the Expert Travel Agent
                   - Local insights collection to the Local Tour Guide
                
                3. Ensure integration and consistency:
                   - Verify that all components work together
                   - Check that budget constraints are respected
                   - Validate that traveler interests are addressed
                   - Maintain quality throughout the plan
                
                4. Provide final recommendations and adjustments
                
                Use the blackboard system to:
                - Monitor progress of each task
                - Share information between agents
                - Track changes and versions
                - Ensure data consistency
                
                {self.__tip_section()}
                """
            ),
            expected_output="A comprehensive travel plan with all components properly integrated",
            agent=agent
        )

    def select_destination(self, agent, destination_type, budget, duration):
        """目的地選擇任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Analyze and select the best travel destinations
                **Description**: Based on the traveler's preferences, budget constraints, and time frame, analyze and recommend the 
                optimal travel destinations. Your analysis should consider factors such as seasonal conditions, tourist attractions, 
                safety, and how well the destination matches the traveler's interests.
                
                Please provide:
                1. A ranked list of 3-5 recommended destinations that match the criteria
                2. For each destination, include:
                   - Brief overview and why it's suitable for this trip
                   - Best areas/neighborhoods to stay
                   - Must-see attractions and activities
                   - Estimated costs for accommodations, food, and activities
                   - Best time to visit and current seasonal conditions
                
                **Parameters**:
                - Destination Type: {destination_type} (e.g., beach, city, mountains, cultural)
                - Budget: ${budget} for the entire trip
                - Duration: {duration} days
                
                {self.__tip_section()}
                
                Make sure to use the most recent data possible and consider current travel conditions.
                
                Note: Write your findings to the blackboard for other agents to reference.
                """
            ),
            expected_output="A detailed report of recommended destinations with justifications and practical information",
            agent=agent
        )
    
    def create_itinerary(self, agent, destination, duration, interests):
        """行程規劃任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Develop a comprehensive day-by-day travel itinerary
                **Description**: Create a detailed itinerary for a {duration}-day trip to {destination}, taking into account 
                the traveler's interests and preferences. The itinerary should maximize the travel experience while remaining 
                practical and avoiding an overly rushed schedule.
                
                Your itinerary should include:
                1. A day-by-day schedule with morning, afternoon, and evening activities
                2. Recommended accommodations for each night
                3. Transportation options between sites and activities
                4. Estimated timing for each activity, including travel time between locations
                5. Meal recommendations (especially local specialties worth trying)
                6. Backup activities or rainy-day alternatives
                7. A packing list tailored to the destination and planned activities
                8. Estimated budget breakdown for each day
                
                **Parameters**:
                - Destination: {destination}
                - Duration: {duration} days
                - Interests: {interests} (e.g., history, food, adventure, relaxation)
                
                {self.__tip_section()}
                
                Ensure the itinerary is realistic regarding travel times and activity durations.
                Balance scheduled activities with free time.
                
                Note: Read destination details from the blackboard and write your itinerary for other agents to reference.
                """
            ),
            expected_output="A complete day-by-day itinerary with all requested details and practical considerations",
            agent=agent
        )

    def provide_local_insights(self, agent, destination):
        """本地見解任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Provide insider knowledge and local recommendations
                **Description**: As an experienced local guide, provide authentic insights and recommendations for travelers 
                visiting {destination}. Go beyond typical tourist information to share hidden gems, local customs, and practical 
                advice that will enhance the traveler's experience.
                
                Your insights should include:
                1. Hidden gems and off-the-beaten-path attractions locals love
                2. Cultural norms, etiquette, and local customs visitors should be aware of
                3. Common tourist mistakes or scams to avoid
                4. Authentic local dining recommendations at various price points
                5. Practical tips for navigating public transportation
                6. Useful local phrases or language tips
                7. Best times of day to visit popular attractions to avoid crowds
                8. Local festivals or events happening during the travel period
                9. Safety tips specific to different neighborhoods
                
                **Parameters**:
                - Destination: {destination}
                
                {self.__tip_section()}
                
                Focus on providing authentic, current information that can't easily be found in standard guidebooks or websites.
                
                Note: Read the existing itinerary from the blackboard and provide insights that complement the planned activities.
                """
            ),
            expected_output="A comprehensive guide with insider knowledge, cultural insights, and practical local tips",
            agent=agent
        )

    def optimize_plan(self, agent, initial_itinerary, initial_insights, budget, duration):
        """最終行程優化任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: 優化並整合最終旅行計劃
                **Description**: 作為旅行規劃經理，您需要與所有專家合作，全面審查和優化旅行計劃。
                請確保計劃滿足所有要求並達到最佳效果。

                需要審查和優化的內容：
                1. 行程可行性審查
                   - 檢查行程時間安排的合理性
                   - 確認各景點/活動之間的交通時間
                   - 評估天氣和季節性因素的影響
                   - 標記任何潛在的衝突或不可行點
                
                2. 預算合理性評估
                   - 總預算: ${budget}
                   - 檢查所有費用項目（住宿、交通、餐飲、活動）
                   - 確保有應急預算（建議為總預算的10-15%）
                   - 建議可能的節省方案
                   - 標記任何超出預算的項目
                
                3. 時間安排優化
                   - 行程天數: {duration} 天
                   - 優化景點遊覽順序
                   - 合理安排休息時間
                   - 預留彈性時間應對突發情況
                   - 標記任何時間安排過密或不合理的部分
                
                4. 整體體驗提升
                   - 結合當地特色活動
                   - 平衡觀光與深度體驗
                   - 添加特別推薦和替代方案
                   - 標記任何可能影響體驗品質的因素
                
                5. 問題處理機制
                   - 對於發現的每個問題：
                     * 問題描述
                     * 嚴重程度評估（高/中/低）
                     * 建議的解決方案
                     * 需要協調的專家（如果需要）
                   - 提供備選方案或應急計劃
                
                6. 跨專家協調
                   - 如發現問題需要：
                     * City Expert 重新評估目的地選擇
                     * Travel Expert 調整行程安排
                     * Local Guide 提供替代建議
                   - 確保各專家間的建議不衝突
                
                初始行程：
                {initial_itinerary}
                
                當地見解：
                {initial_insights}
                
                請提供：
                1. 優化後的詳細行程
                2. 更新的當地建議
                3. 優化總結報告，包含：
                   - 主要改進點
                   - 發現的問題及解決方案
                   - 需要特別注意的事項
                   - 建議的後續行動（如果需要）
                
                {self.__tip_section()}
                """
            ),
            expected_output=dedent("""
                {
                    "itinerary": "優化後的詳細行程",
                    "local_insights": "更新的當地建議",
                    "optimization_summary": {
                        "improvements": "主要改進點列表",
                        "issues": {
                            "critical": "需要立即處理的問題",
                            "moderate": "需要注意的問題",
                            "minor": "可以改進的小問題"
                        },
                        "solutions": "針對每個問題的解決方案",
                        "next_steps": "建議的後續行動"
                    }
                }
            """),
            agent=agent
        )

    # 第一階段任務
    def analyze_requirements(self, agent, destination_type, budget, duration, interests):
        """需求分析任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Analyze travel requirements and create initial framework
                **Description**: As the Trip Planning Manager, analyze the requirements and create a structured framework for the trip.
                
                Focus areas:
                1. Requirements Analysis
                   - Destination Type: {destination_type}
                   - Budget: ${budget}
                   - Duration: {duration} days
                   - Interests: {interests}
                
                2. Create Planning Framework
                   - Define key milestones
                   - Identify critical decision points
                   - Outline resource allocation strategy
                
                3. Risk Assessment
                   - Identify potential challenges
                   - Suggest mitigation strategies
                
                {self.__tip_section()}
                """
            ),
            expected_output="A structured analysis report with planning framework and risk assessment",
            agent=agent
        )

    def assess_feasibility(self, agent, destination_type, budget, duration):
        """可行性評估任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Assess travel plan feasibility from a travel expert perspective
                **Description**: Evaluate the feasibility of the travel requirements considering practical travel factors.
                
                Analyze:
                1. Travel Logistics
                   - Transportation options
                   - Travel time requirements
                   - Connection possibilities
                
                2. Budget Feasibility
                   - Transportation costs
                   - Accommodation estimates
                   - Activity cost ranges
                
                3. Timeline Assessment
                   - Required travel time
                   - Activity scheduling possibilities
                   - Buffer time needs
                
                Parameters:
                - Destination Type: {destination_type}
                - Budget: ${budget}
                - Duration: {duration} days
                
                {self.__tip_section()}
                """
            ),
            expected_output="A detailed feasibility report with specific recommendations",
            agent=agent
        )

    def identify_constraints(self, agent, destination_type, budget):
        """限制條件識別任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Identify destination-specific constraints and limitations
                **Description**: Analyze potential constraints related to the destination type and budget.
                
                Focus on:
                1. Geographic Constraints
                   - Accessibility issues
                   - Seasonal limitations
                   - Infrastructure requirements
                
                2. Budget Constraints
                   - Cost of living analysis
                   - Price seasonality
                   - Hidden costs
                
                3. Regulatory Constraints
                   - Visa requirements
                   - Travel restrictions
                   - Health requirements
                
                Parameters:
                - Destination Type: {destination_type}
                - Budget: ${budget}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A comprehensive constraints analysis with mitigation suggestions",
            agent=agent
        )

    def suggest_experience_options(self, agent, destination_type, interests):
        """體驗建議任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Suggest unique experience options based on interests
                **Description**: Provide creative experience suggestions that match the traveler's interests.
                
                Create suggestions for:
                1. Cultural Experiences
                   - Local customs
                   - Traditional activities
                   - Cultural events
                
                2. Activity Options
                   - Adventure activities
                   - Relaxation options
                   - Educational experiences
                
                3. Special Experiences
                   - Unique local offerings
                   - Seasonal specialties
                   - Hidden gems
                
                Parameters:
                - Destination Type: {destination_type}
                - Interests: {interests}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A curated list of experience options with descriptions and recommendations",
            agent=agent
        )

    # 第二階段任務
    def research_destinations(self, agent, destination_type, budget):
        """目的地研究任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Research and analyze potential destinations
                **Description**: Conduct detailed research on potential destinations that match the requirements.
                
                Research areas:
                1. Destination Analysis
                   - Match with {destination_type} requirement
                   - Cost of living analysis
                   - Tourist infrastructure
                   - Safety and security
                
                2. Seasonal Considerations
                   - Weather patterns
                   - Peak/off-peak seasons
                   - Local events and festivals
                
                3. Budget Analysis
                   - Accommodation costs
                   - Local transportation costs
                   - Activity and attraction prices
                   - Food and entertainment expenses
                
                Budget: ${budget}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A detailed analysis of potential destinations with pros and cons",
            agent=agent
        )

    def evaluate_transportation(self, agent, destination_candidates):
        """交通評估任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Evaluate transportation options and logistics
                **Description**: Analyze transportation options and logistics for candidate destinations.
                
                Evaluate:
                1. External Transportation
                   - Flight options and costs
                   - Alternative transport methods
                   - Transit visa requirements
                
                2. Local Transportation
                   - Public transport systems
                   - Taxi/ride-sharing availability
                   - Car rental options
                
                3. Inter-city Travel
                   - Connection options
                   - Time requirements
                   - Cost analysis
                
                Destinations to evaluate:
                {destination_candidates}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A comprehensive transportation analysis for each destination",
            agent=agent
        )

    def assess_seasonal_factors(self, agent, destination_candidates):
        """季節性因素評估任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Assess seasonal factors and timing considerations
                **Description**: Evaluate seasonal impacts on travel experience for each destination.
                
                Assessment areas:
                1. Weather Impact
                   - Temperature and precipitation
                   - Natural phenomena
                   - Activity limitations
                
                2. Tourism Seasons
                   - Peak/off-peak periods
                   - Price variations
                   - Crowd levels
                
                3. Local Events
                   - Festivals and celebrations
                   - Cultural events
                   - Special attractions
                
                Destinations to assess:
                {destination_candidates}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A seasonal analysis report with timing recommendations",
            agent=agent
        )

    # 第三階段任務
    def design_route_optimization(self, agent, selected_destination, duration):
        """路線優化任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Optimize travel routes and timing
                **Description**: Create optimized travel routes and timing plans.
                
                Planning areas:
                1. Route Design
                   - Efficient travel paths
                   - Time optimization
                   - Alternative routes
                
                2. Activity Clustering
                   - Geographic grouping
                   - Time-based grouping
                   - Theme-based grouping
                
                3. Buffer Planning
                   - Travel time buffers
                   - Rest periods
                   - Flexibility options
                
                Parameters:
                - Destination: {selected_destination}
                - Duration: {duration} days
                
                {self.__tip_section()}
                """
            ),
            expected_output="An optimized route plan with timing details",
            agent=agent
        )

    def create_activity_schedule(self, agent, selected_destination, interests):
        """活動排程任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Create detailed activity schedules
                **Description**: Develop comprehensive activity schedules based on interests.
                
                Schedule components:
                1. Main Activities
                   - Must-see attractions
                   - Interest-based activities
                   - Special experiences
                
                2. Alternative Options
                   - Weather backup plans
                   - Flexible activities
                   - Optional additions
                
                3. Timing Details
                   - Opening hours
                   - Best visiting times
                   - Duration estimates
                
                Parameters:
                - Destination: {selected_destination}
                - Interests: {interests}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A detailed activity schedule with alternatives and timing",
            agent=agent
        )

    def monitor_budget_compliance(self, agent, planned_activities, budget):
        """預算監控任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: Monitor and ensure budget compliance
                **Description**: Track and validate all planned expenses against budget.
                
                Monitoring areas:
                1. Expense Tracking
                   - Activity costs
                   - Transportation expenses
                   - Accommodation costs
                   - Food and miscellaneous
                
                2. Budget Analysis
                   - Cost breakdown
                   - Buffer allocation
                   - Potential savings
                
                3. Optimization Suggestions
                   - Cost-saving options
                   - Value maximization
                   - Budget reallocation
                
                Parameters:
                - Planned Activities: {planned_activities}
                - Total Budget: ${budget}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A budget compliance report with optimization suggestions",
            agent=agent
        )


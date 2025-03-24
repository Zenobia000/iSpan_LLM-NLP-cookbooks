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
    
    def coordinate_planning(self, agent, destination, itinerary, insights, optimization, budget, duration, currency="USD"):
        """最終計劃協調與整合任務"""
        # 根據幣值選擇顯示適當的符號
        currency_symbol = "NT$" if currency == "TWD" else "$"
        
        return Task(
            description=dedent(f"""
                作為旅行規劃管理專家，您需要協調各個專家提供的信息，並創建最終的旅行計劃文檔。
                
                ## 輸入信息
                - 目的地選擇與分析: {destination}
                - 初步行程規劃: {itinerary}
                - 當地體驗與洞察: {insights}
                - 優化建議: {optimization}
                - 預算: {currency_symbol}{budget} {currency}
                - 行程天數: {duration}天
                
                ## 輸出要求
                請整合上述所有信息，創建一份全面、連貫且實用的旅行計劃文檔。最終文檔必須使用繁體中文呈現，包括所有標題、小標題和內容。
                
                文檔應包含：
                1. 行程摘要與亮點（全繁體中文）
                2. 目的地概述與實用信息（全繁體中文）
                3. 詳細的日程安排（全繁體中文）
                4. 預算細分與消費指南（全繁體中文）
                5. 實用提示與資源（全繁體中文）
                6. 備選方案與靈活性建議（全繁體中文）
                
                請確保內容實用、具體且有指導性，幫助旅行者輕鬆執行這個計劃。
                **重要提示：整個文檔必須完全使用繁體中文，不得使用任何英文標題或內容。**
            """),
            agent=agent,
            expected_output="一份全面且實用的旅行計劃文檔，完全使用繁體中文呈現，包含所有必要細節和建議。"
        )

    def select_destination(self, agent, destination_type, budget, duration, requirements_analysis, currency="USD"):
        """目的地選擇任務"""
        # 根據幣值選擇顯示適當的符號
        currency_symbol = "NT$" if currency == "TWD" else "$"
        
        return Task(
            description=dedent(
                f"""
                **Task**: 分析並選擇最佳旅行目的地
                **Description**: 基於旅行者的偏好、預算限制和時間框架，分析並推薦最佳的旅行目的地。您的分析應考慮季節性條件、旅遊景點、安全因素，以及目的地與旅行者興趣的匹配度。
                
                需求分析報告：
                ```
                {requirements_analysis}
                ```
                
                請提供：
                1. 3-5個符合條件的推薦目的地排名清單
                2. 每個目的地包含：
                   - 簡要概述及其適合此次旅行的原因
                   - 最佳住宿區域/社區
                   - 必看景點和活動
                   - 住宿、餐飲和活動的估計成本 (以{currency}為單位)
                   - 最佳旅行時間和當前季節性條件
                   - 安全考量和特殊提示
                3. 最終目的地建議及選擇理由
                
                **參數**:
                - 目的地類型: {destination_type}
                - 預算: {currency_symbol}{budget} {currency}
                - 行程天數: {duration}天
                
                確保使用最新的數據並考慮當前旅行條件。您的推薦應詳細、具體且貼合需求分析結果。
                若選擇的貨幣為TWD，請確保在預算考慮中適當換算。
                
                {self.__tip_section()}
                """
            ),
            expected_output="一份詳細的推薦目的地報告，包含目的地分析、比較和最終建議，提供充分的理由和實用信息",
            agent=agent
        )
    
    def create_itinerary(self, agent, destination, duration, interests, budget, currency="USD"):
        """行程規劃任務"""
        # 根據幣值選擇顯示適當的符號
        currency_symbol = "NT$" if currency == "TWD" else "$"
        
        return Task(
            description=dedent(
                f"""
                **Task**: 開發全面的逐日旅行行程
                **Description**: 為{destination}的{duration}天行程創建詳細的行程安排，考慮旅行者的興趣和偏好。行程應最大化旅行體驗，同時保持實用性並避免過於緊湊的安排。
                
                您的行程應包括：
                1. 逐日安排，包含上午、下午和晚上的活動
                2. 每晚的住宿建議，包含價格範圍和特點
                3. 景點間的交通選擇和方式
                4. 每項活動的預計時間，包括景點間的交通時間
                5. 餐飲建議（特別是值得嘗試的當地特色）
                6. 備選活動或雨天備案
                7. 根據目的地和計劃活動定制的打包清單
                8. 每天的預算細分 (以{currency}為單位)
                9. 住宿、交通、餐飲、活動、雜項的總體預算分配
                
                **參數**:
                - 目的地: {destination}
                - 行程天數: {duration}天
                - 興趣: {interests}
                - 總預算: {currency_symbol}{budget} {currency}
                
                確保行程在旅行時間和活動時長方面是現實可行的。
                平衡安排活動和自由時間。
                考慮當地交通狀況和景點開放時間。
                根據旅行者的興趣優先安排活動。
                若使用TWD作為貨幣單位，請適當考慮匯率因素。
                
                {self.__tip_section()}
                """
            ),
            expected_output="一份完整的逐日行程，包含所有請求的細節和實用考量，並附有預算分配明細",
            agent=agent
        )

    def provide_local_insights(self, agent, destination, itinerary, interests):
        """本地體驗強化任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: 提供內部知識和當地推薦
                **Description**: 作為經驗豐富的當地導遊，為前往{destination}的旅行者提供真實的見解和建議。超越典型的旅遊信息，分享隱藏寶藏、當地風俗和實用建議，以提升旅行者的體驗。
                
                現有行程：
                ```
                {itinerary}
                ```
                
                旅行者興趣：{interests}
                
                您的見解應包括：
                1. 當地人喜愛的隱藏景點和非傳統景點
                2. 遊客應該了解的文化規範、禮儀和當地習俗
                3. 避免常見的遊客錯誤或騙局
                4. 不同價位的正宗當地餐廳推薦
                5. 使用公共交通的實用技巧
                6. 有用的當地短語或語言提示
                7. 避開人群的最佳時間參觀熱門景點
                8. 旅行期間的當地節日或活動
                9. 針對不同社區的安全提示
                10. 根據旅行者興趣的特殊體驗或活動
                11. 與現有行程互補的具體建議和增強點
                
                {self.__tip_section()}
                
                專注於提供真實、最新的信息，這些信息在標準旅遊指南或網站上不容易找到。您的建議應增強而不是取代現有行程，提供額外的深度和真實性。
                """
            ),
            expected_output="一份詳盡的指南，包含內部知識、文化見解和針對現有行程的實用當地提示",
            agent=agent
        )

    def optimize_plan(self, agent, initial_itinerary, initial_insights, budget, duration, currency="USD"):
        """行程優化任務"""
        # 根據幣值選擇顯示適當的符號
        currency_symbol = "NT$" if currency == "TWD" else "$"
        
        return Task(
            description=dedent(
                f"""
                **Task**: 優化並完善旅行計劃
                **Description**: 作為質量控制專家，全面審查和優化旅行計劃。確保行程在時間安排、預算和整體體驗方面達到最佳效果。
                
                初始行程：
                ```
                {initial_itinerary}
                ```
                
                當地見解：
                ```
                {initial_insights}
                ```
                
                審查和優化重點：
                
                1. 實用性和可行性
                   - 評估每日行程的實際可行性
                   - 檢查行程點之間的交通時間是否合理
                   - 確認開放時間和季節性限制
                   - 識別並解決任何行程衝突或過度安排
                
                2. 預算優化
                   - 總預算: {currency_symbol}{budget} {currency}
                   - 評估預算分配的合理性
                   - 提出優化建議以增加價值
                   - 標記任何超出預算或不平衡的方面
                   - 確保包含5-10%的緊急預算
                
                3. 體驗增強
                   - 整合當地見解到主要行程中
                   - 平衡熱門景點與獨特體驗
                   - 確保行程節奏適宜（避免疲勞）
                   - 加入"緩衝時間"以應對延誤或自發探索
                
                4. 風險管理
                   - 識別潛在問題並提供解決方案
                   - 提供備選方案（天氣、閉館等）
                   - 包含安全提示和緊急聯繫信息
                
                請提供：
                1. 優化建議摘要
                2. 具體的日程調整（如需要）
                3. 預算優化建議 (以{currency}為單位)
                4. 體驗增強建議
                5. 實用提示和備選方案
                
                您的最終輸出應平衡實用性、預算考量和體驗質量，確保創造一個順暢且令人難忘的旅行計劃。
                
                {self.__tip_section()}
                """
            ),
            expected_output="一份全面的優化報告，包含具體調整建議、預算優化、體驗增強和風險管理策略",
            agent=agent
        )

    # 第一階段任務
    def analyze_requirements(self, agent, destination_type, budget, duration, interests, currency="USD"):
        """需求分析任務"""
        return Task(
            description=dedent(
                f"""
                **Task**: 全面分析旅行需求和限制條件
                **Description**: 作為需求分析專家，您需要徹底分析旅行者的需求、限制條件和偏好，為後續規劃奠定基礎。
                
                您需要完成：
                
                1. 需求細分與分析
                   - 目的地類型分析：{destination_type} - 該類型目的地的典型特點、常見活動和體驗
                   - 預算分析：{budget} - 預算適合度、如何最佳分配、預算限制下的優化建議
                   - 時間框架分析：{duration}天 - 合理行程安排、活動密度建議、行程節奏規劃
                   - 興趣偏好分析：{interests} - 如何將這些興趣與目的地類型和活動相匹配
                
                2. 限制條件識別
                   - 預算限制因素 (考慮選定的貨幣：{currency})
                   - 時間限制因素
                   - 季節性考量
                   - 潛在的特殊需求
                
                3. 旅行者畫像構建
                   - 根據提供的信息推斷旅行者類型（探險型、文化型、休閒型等）
                   - 推斷旅行風格偏好（奢華、經濟、平衡等）
                   - 推斷活動強度偏好（密集探索型、輕鬆休閒型等）
                
                4. 初步建議
                   - 預算分配建議（住宿、交通、餐飲、活動等）
                   - 時間分配建議
                   - 目的地類型中的優先地區或國家
                   - 根據興趣的重點活動類型
                
                若選擇的貨幣為TWD，請在分析中考慮其國際換算性和適用區域。
                
                請提供一份全面的分析報告，包括上述所有方面，以及您認為重要的任何其他考量。報告應當客觀、詳細、有洞察力，並為後續的目的地選擇和行程規劃提供明確指導。
                
                {self.__tip_section()}
                """
            ),
            expected_output="一份全面詳細的旅行需求分析報告，包括需求解析、限制條件、旅行者畫像和初步建議",
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
                - Budget: {budget}
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
                - Budget: {budget}
                
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
                
                Budget: {budget}
                
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
                - Total Budget: {budget}
                
                {self.__tip_section()}
                """
            ),
            expected_output="A budget compliance report with optimization suggestions",
            agent=agent
        )

    def translate_final_plan(self, agent, plan_document):
        """創建翻譯最終旅行計劃的任務"""
        return Task(
            description=dedent(f"""
                您的任務是將以下旅行計劃文檔完整翻譯成繁體中文，包括所有標題、小標題和內容。
                
                ## 翻譯要求
                1. 將所有英文或混合語言內容轉換為純繁體中文
                2. 保持原文檔的結構和格式
                3. 確保所有日期、價格、時間等信息的準確性
                4. 使用台灣當地常用的旅遊術語和表達方式
                5. 保持專有名詞的正確翻譯（如地名、景點名稱等）
                
                ## 待翻譯的原始文檔
                {plan_document}
                
                ## 輸出格式
                請提供一份完整的繁體中文旅行計劃文檔，保持原有的格式結構。
            """),
            agent=agent,
            expected_output="一份完全使用繁體中文表達的旅行計劃文檔，保持原有的格式結構和內容完整性。"
        )


from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from crewai.tools import tool
from dotenv import load_dotenv

# --- 環境設定 ---
# 為了方便執行，我們從 .env 檔案加載 OpenAI API 金鑰
# 請確保您的專案根目錄下有一個 .env 檔案，且內容為：
# OPENAI_API_KEY="sk-..."
load_dotenv()

# 如果沒有 .env 檔案，您也可以直接在這裡設定，但這不是一個好習慣
# os.environ["OPENAI_API_KEY"] = "sk-..."

# --- 工具設定 ---
# 在這個版本的 CrewAI 中，我們使用 `@tool` 裝飾器來定義工具。
# 我們定義一個函數，並用 `@tool` 標記它。函數的 docstring 會被用作工具的描述。
@tool("DuckDuckGo Search")
def search_tool(query: str) -> str:
    """A wrapper around DuckDuckGo Search. Use this to search for up-to-date information on the web. The query should be a search string."""
    return DuckDuckGoSearchRun().run(query)

# =====================================================================================
# 1. 定義我們的代理 (Agents)
# =====================================================================================
# 在多代理系統中，每個代理都有一個明確的角色 (role)、目標 (goal) 和背景故事 (backstory)。
# 這能幫助大型語言模型 (LLM) 更好地「扮演」這個角色，從而執行更精準的任務。

# 代理1：城市選擇專家
city_selection_expert = Agent(
    role="旅遊目的地專家",
    goal="根據用戶的偏好（例如天氣、季節、興趣），利用網路搜尋功能，從全球範圍內選擇一個最適合的城市。", # <-- 已更新，說明會使用網路搜尋
    backstory="您是一位經驗豐富的旅行顧問，擁有豐富的地理和文化知識，並且精通網路搜尋，總能為旅客找到他們夢想中的完美度假地點。", # <-- 已更新，說明會使用網路搜尋
    verbose=True,  # 設定為 True，可以看到此代理的詳細執行過程
    allow_delegation=False, # 在這個簡單範例中，我們不允許代理之間互相委派任務
    tools=[search_tool], # <-- 新增！讓這個代理可以使用網路搜尋工具
)

# 代理2：當地行程規劃師
local_tour_guide = Agent(
    role="當地導遊",
    goal="為一個給定的城市，設計一個引人入勝且可行的三天行程大綱。",
    backstory="您就是這個城市的在地通！從熱門景點到當地人才知道的私房美食，您都能安排得妥妥當當，讓旅客體驗最道地的文化。",
    verbose=True,
    allow_delegation=False,
)

# 代理3：旅行報告整合員
travel_concierge = Agent(
    role="旅行行程整合員",
    goal="將城市選擇的理由和三天的行程規劃，整合成一份格式優美、易於閱讀的最終旅行建議報告。",
    backstory="您是一位細心且有條理的旅行秘書，擅長將零散的資訊片段，轉化成一份精美、完整且令人心動的行程單。",
    verbose=True,
    allow_delegation=False,
)


# =====================================================================================
# 2. 定義他們需要執行的任務 (Tasks)
# =====================================================================================
# 任務是代理需要完成的具體工作。
# - `description`：描述任務的具體要求，可以使用 `{}` 傳入變數。
# - `expected_output`：明確告知代理，我們期望它產出什麼樣的結果。
# - `agent`：指定由哪一個代理來執行此任務。
# - `context`：定義任務的依賴關係，讓一個任務可以利用其他任務的成果。

# 任務1：選擇城市
# 這個任務是整個流程的起點。
select_city_task = Task(
    description="分析用戶對於 '{topic}' 的旅行偏好，並從全球選擇一個最符合這些偏好的城市。",
    expected_output="一個城市的名稱，並附上一段簡短的說明，解釋為什麼推薦這個城市。",
    agent=city_selection_expert,
)

# 任務2：規劃行程
# 這個任務依賴於 `select_city_task` 的結果。
# 我們使用 `context` 參數來告訴它，你需要參考「選擇城市」任務的產出。
plan_itinerary_task = Task(
    description="根據前一個任務選出的城市，為其設計一個為期三天的行程亮點，每天至少包含 2-3 個活動建議。",
    expected_output="一個條列式的三日行程表，清楚說明每天的活動安排。",
    agent=local_tour_guide,
    context=[select_city_task],  # <-- 關鍵！定義了此任務的依賴
)

# 任務3：整合報告
# 這個任務是流程的終點，它依賴前兩個任務的成果，進行最終的彙總。
format_report_task = Task(
    description="將城市推薦的理由和詳細的三日行程整合成一份完整的旅行建議報告。報告需使用 Markdown 格式，包含主標題、城市介紹和每日行程列表。",
    expected_output="一份格式化的 Markdown 文件，包含最終的旅行建議。",
    agent=travel_concierge,
    context=[select_city_task, plan_itinerary_task], # <-- 關鍵！它需要參考前兩個任務的結果
)


# =====================================================================================
# 3. 組建團隊 (Crew) 並設定執行流程
# =====================================================================================
# Crew 是代理和任務的組合。
# - `agents`：定義團隊中有哪些成員。
# - `tasks`：定義團隊需要完成的所有任務。
# - `process`：定義任務的執行方式。`Process.sequential` 表示任務將按照 `tasks` 列表中的順序，一個接一個地執行。
crew = Crew(
    agents=[city_selection_expert, local_tour_guide, travel_concierge],
    tasks=[select_city_task, plan_itinerary_task, format_report_task],
    process=Process.sequential,  # 設定為循序執行流程
    verbose=True # 設定為 True，可以看到詳細的執行日誌
)


# =====================================================================================
# 4. 啟動任務
# =====================================================================================
# 這是程式的進入點。
if __name__ == "__main__":
    print("##################################################")
    print("## 歡迎使用 CrewAI 簡易旅行規劃器 ##")
    print("##################################################")
    
    # 獲取用戶輸入，這個輸入會被傳遞到任務的 `{topic}` 變數中
    topic = input("您好！請告訴我，您這次想去哪一種類型的地方旅行？（例如：一個充滿歷史感的古城、陽光與沙灘的熱帶島嶼...）\n> ")
    
    # 使用 .kickoff() 方法啟動 Crew，並傳入需要的變數
    # CrewAI 會自動將 `topic` 的值填入 `select_city_task` 的 `description` 中
    result = crew.kickoff(inputs={'topic': topic})
    
    print("\n\n##################################################")
    print("##            您的專屬旅行計畫已完成            ##")
    print("##################################################\n")
    print(result)

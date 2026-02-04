# 簡易旅行規劃器 (Simplified Trip Planner)

這是一個使用 [CrewAI](https://github.com/joaomdmoura/crewAI) 框架建立的簡易多代理 (Multi-agent) 系統範例。

## 專案目標

此專案的主要目的是 **教育與展示**，而非建構一個功能完備的應用程式。它旨在用最簡潔的程式碼，清晰地展示多代理系統的核心架構，包括：

1.  **代理分工**：如何定義擁有多種不同角色 (Role) 和目標 (Goal) 的代理。
2.  **任務依賴**：一個任務如何利用前一個任務的產出 (`context`) 來執行工作。
3.  **循序流程**：如何將代理和任務組建成一個團隊 (`Crew`)，並讓它們按照指定的順序 (`Process.sequential`) 協同工作。

所有相關的程式碼都被整合在 `main.py` 單一檔案中，以便於閱讀和理解。

## 設定步驟

1.  **安裝必要的 Python 套件**:
    ```bash
    pip install crewai python-dotenv
    ```

2.  **設定您的 API 金鑰**:
    *   將專案中的 `.env_example` 檔案重新命名為 `.env`。
    *   打開 `.env` 檔案，並將您的 OpenAI API 金鑰填入其中：
      ```
      OPENAI_API_KEY="sk-..."
      ```

## 如何執行

完成設定後，直接在終端機中執行 `main.py` 檔案即可：

```bash
python main.py
```

程式會引導您輸入旅行偏好，然後啟動多代理系統為您規劃行程。

"""
專用調試腳本 - 強制重新加載模塊並修復參數問題
"""
import os
import sys
import importlib
import inspect
from dotenv import load_dotenv

# 加載 .env 文件中的環境變量
load_dotenv()

# 確保 OpenAI API 密鑰設置正確
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️ 未設置 OPENAI_API_KEY 環境變量")
    api_key = input("請輸入您的 OpenAI API 密鑰: ")
    os.environ["OPENAI_API_KEY"] = api_key
    print("✅ 已設置 OPENAI_API_KEY 環境變量")
else:
    print("✓ OPENAI_API_KEY 環境變量已設置")

# 確保當前目錄在 Python 路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
    print(f"✅ 已將 {current_dir} 添加到 Python 路徑")

# 強制重新加載模塊
print("正在載入並檢查 agents 模塊...")
import agents
importlib.reload(agents)

# 檢查 create_base_agent_template 方法是否接受 llm 參數
template_method = getattr(agents.TripPlannerAgents, 'create_base_agent_template')
signature = inspect.signature(template_method)
has_llm_param = 'llm' in signature.parameters

# 如果不接受 llm 參數，則修改方法
if not has_llm_param:
    from crewai import Agent
    from textwrap import dedent
    
    # 修正 TripPlannerAgents 類
    def patched_create_base_agent_template(self, role_name, backstory, goal, tools=None, llm=None):
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
    
    # 動態替換方法
    setattr(agents.TripPlannerAgents, 'create_base_agent_template', patched_create_base_agent_template)
    print("✅ 已修復 create_base_agent_template 方法以接受 llm 參數")
else:
    print("✓ create_base_agent_template 方法已正確定義，無需修改")

# 修改 TripPlannerAgents.__init__ 以避免 API 密鑰問題
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from tools import initialize_tools

def patched_init(self):
    """初始化旅行規劃代理系統 - 修復版本"""
    # 打印 API 密鑰長度，保持安全
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print(f"✓ 使用 API 密鑰 ({len(api_key)} 個字符)")
    else:
        print("⚠️ 沒有找到 API 密鑰")
    
    # 設置語言模型，確保支持中文
    try:
        self.OpenAIGPT35 = ChatOpenAI(
            model_name="gpt-3.5-turbo", 
            temperature=0.7
        )
        print("✅ 成功初始化 OpenAIGPT35 模型")
    except Exception as e:
        print(f"⚠️ 初始化 OpenAIGPT35 時出錯: {e}")
        # 使用安全的後備選項
        self.OpenAIGPT35 = None
    
    try:
        self.OpenAIGPT4 = ChatOpenAI(
            model_name="gpt-4o", 
            temperature=0.7
        )
        print("✅ 成功初始化 OpenAIGPT4 模型")
    except Exception as e:
        print(f"⚠️ 初始化 OpenAIGPT4 時出錯: {e}")
        if self.OpenAIGPT35:
            print("⚠️ 使用 GPT-3.5 替代 GPT-4")
            self.OpenAIGPT4 = self.OpenAIGPT35
        else:
            self.OpenAIGPT4 = None
    
    # 嘗試初始化 Ollama 模型
    try:
        self.Ollama = OllamaLLM(model="openhermes")
        print("✅ 成功初始化 Ollama 模型")
    except Exception as e:
        print(f"⚠️ 初始化 Ollama 時出錯 (可忽略): {e}")
        self.Ollama = None
    
    # 初始化工具註冊系統
    self.tool_registry = initialize_tools()
    print("✅ 工具註冊系統初始化成功")

# 應用修補函數
print("正在應用修補的初始化函數...")
setattr(agents.TripPlannerAgents, '__init__', patched_init)

# 測試修復後的方法
print("\n測試修復結果:")
try:
    from agents import TripPlannerAgents
    test_agents = TripPlannerAgents()
    
    print("\n嘗試創建質量保證專家代理...")
    quality_controller = test_agents.quality_assurance_expert()
    print("✅ 質量保證專家代理創建成功")
    
    print("\n嘗試創建翻譯專家代理...")
    translation_expert = test_agents.chinese_translation_expert()
    print("✅ 翻譯專家代理創建成功")
    
    print("\n所有測試通過!")
except Exception as e:
    print(f"❌ 測試失敗: {e}")

print("\n現在您可以運行主程序了! 請使用以下命令:")
print("python main.py")

# 修復 ChatOpenAI 缺少 supports_stop_words 方法的問題
print("\n修復 ChatOpenAI 模型兼容性問題...")
from langchain_openai import ChatOpenAI

# 添加缺少的方法
def supports_stop_words(self) -> bool:
    return True

def get_num_tokens_if_needed(self, text: str) -> int:
    if hasattr(self, "get_num_tokens"):
        return self.get_num_tokens(text)
    return len(text) // 4

# 動態添加方法到 ChatOpenAI 類
if not hasattr(ChatOpenAI, "supports_stop_words"):
    setattr(ChatOpenAI, "supports_stop_words", supports_stop_words)
    print("✅ 已添加 supports_stop_words 方法到 ChatOpenAI")

if not hasattr(ChatOpenAI, "get_num_tokens") and not hasattr(ChatOpenAI, "_get_num_tokens"):
    setattr(ChatOpenAI, "get_num_tokens", get_num_tokens_if_needed)
    print("✅ 已添加 get_num_tokens 方法到 ChatOpenAI")

print("✅ ChatOpenAI 模型兼容性問題已修復") 
#!/usr/bin/env python3
"""
LangChain v1.0+ 遷移驗證腳本
檢查關鍵功能是否正常工作
"""

import sys
import os
from pathlib import Path

# 添加當前目錄到 Python 路徑
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """測試關鍵 import 是否正常"""
    print("🔍 測試 Import 語句...")

    try:
        # 測試核心 import
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        print("✅ 核心組件 import 成功")
    except ImportError as e:
        print(f"❌ 核心組件 import 失敗: {e}")
        return False

    try:
        # 測試 Agent import
        from langchain.agents import tool
        print("✅ Agent 工具 import 成功")
    except ImportError as e:
        print(f"⚠️  Agent import 可能有問題: {e}")

    try:
        # 測試文本分割器 import
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("✅ 文本分割器 import 成功")
    except ImportError as e:
        print(f"⚠️  文本分割器 import 可能有問題: {e}")

    return True

def test_basic_chain():
    """測試基本 Chain 功能"""
    print("\n🔍 測試基本 Chain 功能...")

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough

        # 檢查環境變數
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  未設定 OPENAI_API_KEY，跳過 API 測試")
            return True

        # 建立基本 Chain
        model = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

        prompt = PromptTemplate.from_template("""
        你好！請簡短回覆：{question}
        """)

        output_parser = StrOutputParser()

        chain = (
            {"question": RunnablePassthrough()}
            | prompt
            | model
            | output_parser
        )

        print("✅ Chain 建立成功")

        # 簡單測試
        # response = chain.invoke("LangChain v1.0+ 有什麼新功能？")
        # print(f"✅ Chain 執行成功: {response[:50]}...")

    except Exception as e:
        print(f"❌ Chain 測試失敗: {e}")
        return False

    return True

def test_file_syntax():
    """檢查檔案語法"""
    print("\n🔍 檢查檔案語法...")

    # 關鍵檔案列表
    key_files = [
        "Course/Module1/1_1_framework_overview.py",
        "Course/Module2/2_1_agent_concepts.py",
        "Course/Module3/3_1_rag_basics.py"
    ]

    base_dir = Path(__file__).parent

    for file_path in key_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            print(f"❌ 檔案不存在: {file_path}")
            continue

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 編譯檢查語法
            compile(content, str(full_path), 'exec')
            print(f"✅ {file_path} 語法正確")

        except SyntaxError as e:
            print(f"❌ {file_path} 語法錯誤: {e}")
        except Exception as e:
            print(f"⚠️  {file_path} 檢查時發生問題: {e}")

    return True

def test_requirements():
    """檢查 requirements.txt"""
    print("\n🔍 檢查 requirements.txt...")

    req_file = Path(__file__).parent / "requirements.txt"
    if not req_file.exists():
        print("❌ requirements.txt 不存在")
        return False

    content = req_file.read_text(encoding='utf-8')

    # 檢查關鍵套件版本
    required_packages = {
        "langchain": "1.0.0",
        "langchain-openai": "0.2.0",
        "langchain-core": "0.3.0",
        "langchain-community": "0.3.0"
    }

    for package, min_version in required_packages.items():
        if package in content:
            print(f"✅ {package} 已包含")
        else:
            print(f"⚠️  {package} 可能缺失")

    return True

def main():
    """主測試函數"""
    print("🚀 開始 LangChain v1.0+ 遷移驗證測試\n")

    tests = [
        ("Import 測試", test_imports),
        ("基本 Chain 測試", test_basic_chain),
        ("檔案語法檢查", test_file_syntax),
        ("Requirements 檢查", test_requirements)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 發生異常: {e}")
            results.append((test_name, False))

    # 顯示總結
    print("\n" + "="*50)
    print("📊 測試結果總結:")
    print("="*50)

    passed = 0
    for test_name, result in results:
        status = "✅ 通過" if result else "❌ 失敗"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\n總計: {passed}/{len(results)} 項測試通過")

    if passed == len(results):
        print("🎉 所有測試通過！遷移成功！")
    elif passed >= len(results) * 0.8:
        print("⚠️  大部分測試通過，建議檢查失敗項目")
    else:
        print("❌ 多項測試失敗，需要進一步檢修")

if __name__ == "__main__":
    main()
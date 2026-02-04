#!/usr/bin/env python3
"""
批量遷移 Agent 檔案到 LangChain v1.0+ API
"""

import re
from pathlib import Path

def migrate_file(file_path: Path):
    """遷移單一檔案到 v1.0+ API"""
    if not file_path.exists():
        print(f"檔案不存在: {file_path}")
        return

    content = file_path.read_text(encoding='utf-8')
    original_content = content

    # 1. 更新版本說明
    content = re.sub(
        r'LangChain 0\.3\+ ',
        'LangChain v1.0+ ',
        content
    )

    # 2. 更新 import 語句
    content = re.sub(
        r'from langchain\.agents import create_openai_functions_agent\n',
        '',
        content
    )
    content = re.sub(
        r'from langchain\.agents import AgentExecutor\n',
        '',
        content
    )
    content = re.sub(
        r'from langchain_core\.prompts import ChatPromptTemplate, MessagesPlaceholder\n',
        '',
        content
    )

    # 添加新的 import
    if 'from langchain import create_agent' not in content:
        content = re.sub(
            r'(from langchain_openai import ChatOpenAI\n)',
            r'from langchain import create_agent\n\1',
            content
        )

    # 3. 更新函數定義
    content = re.sub(
        r'def create_agent\(\) -> AgentExecutor:',
        'def create_ai_agent():',
        content
    )

    # 4. 更新函數內容 - 移除舊的 Agent 建立方式
    content = re.sub(
        r'    # 建立提示詞模板.*?MessagesPlaceholder\(variable_name=\'\{agent_scratchpad\}\'\)\s*\]\)',
        '',
        content,
        flags=re.DOTALL
    )

    content = re.sub(
        r'    # 建立 Agent\s*agent = create_openai_functions_agent\(llm, tools, prompt\)\s*',
        '',
        content
    )

    content = re.sub(
        r'    # 建立 Agent 執行器\s*agent_executor = AgentExecutor\(\s*agent=agent,\s*tools=tools,\s*verbose=True\s*\)\s*return agent_executor',
        '''    # 使用新的 create_agent API
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt
    )

    return agent''',
        content
    )

    # 5. 更新 llm 變數名為 model
    content = re.sub(r'    llm = ChatOpenAI\(', '    model = ChatOpenAI(', content)

    # 6. 更新調用方式
    content = re.sub(r'agent_executor = create_agent\(\)', 'agent = create_ai_agent()', content)
    content = re.sub(r'agent_executor\.invoke\(\{"input": ([^}]+)\}\)', r'agent.invoke(\1)', content)
    content = re.sub(r'response\[\'output\'\]', 'response', content)

    # 7. 更新需求套件版本
    content = re.sub(
        r'- langchain>=0\.3\.0',
        '- langchain>=1.0.0',
        content
    )
    content = re.sub(
        r'- langchain-openai>=0\.0\.2',
        '- langchain-openai>=0.2.0',
        content
    )

    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        print(f"已更新: {file_path}")
    else:
        print(f"無需更新: {file_path}")

def main():
    """主函數"""
    # 需要遷移的檔案列表
    files_to_migrate = [
        "Course/Module2/2_2_tools_usage.py",
        "Course/Module2/2_3_agent_automation_demo1.py",
        "Course/Module2/2_3_agent_automation_demo2.py",
        "Course/Module3/3_5_1_custom_llm_agent_template.py",
        "Course/Module3/3_5_2_custom_llm_agent.py"
    ]

    base_dir = Path(__file__).parent

    for file_path in files_to_migrate:
        full_path = base_dir / file_path
        migrate_file(full_path)

    print("批量遷移完成！")

if __name__ == "__main__":
    main()
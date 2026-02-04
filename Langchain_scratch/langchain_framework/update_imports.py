#!/usr/bin/env python3
"""
批量更新 LangChain import 路徑到 v1.0+ 標準
"""

import re
from pathlib import Path

def update_imports_in_file(file_path: Path):
    """更新單一檔案的 import 路徑"""
    if not file_path.exists() or file_path.suffix != '.py':
        return False

    content = file_path.read_text(encoding='utf-8')
    original_content = content

    # 1. 文本分割器 import 更新
    content = re.sub(
        r'from langchain\.text_splitter import',
        'from langchain_text_splitters import',
        content
    )

    # 2. 模型 import 路徑更新
    content = re.sub(
        r'from langchain\.chat_models import ChatOpenAI',
        'from langchain_openai import ChatOpenAI',
        content
    )

    # 3. 文件載入器 import 更新
    content = re.sub(
        r'from langchain\.document_loaders import',
        'from langchain_community.document_loaders import',
        content
    )

    # 4. Embeddings import 更新
    content = re.sub(
        r'from langchain\.embeddings\.openai import OpenAIEmbeddings',
        'from langchain_openai import OpenAIEmbeddings',
        content
    )

    # 5. VectorStore import 更新
    content = re.sub(
        r'from langchain\.vectorstores import',
        'from langchain_community.vectorstores import',
        content
    )

    # 6. Chains import 更新
    content = re.sub(
        r'from langchain\.chains import',
        'from langchain_community.chains import',
        content
    )

    # 7. 輸出解析器 import 更新
    content = re.sub(
        r'from langchain\.output_parsers import',
        'from langchain_core.output_parsers import',
        content
    )

    # 8. Schema import 更新
    content = re.sub(
        r'from langchain\.schema import',
        'from langchain_core.messages import',
        content
    )

    # 檢查是否有變更
    if content != original_content:
        file_path.write_text(content, encoding='utf-8')
        return True
    return False

def main():
    """主函數：遍歷所有 Python 檔案並更新 import"""
    base_dir = Path(__file__).parent
    updated_files = []

    # 遍歷所有 .py 檔案
    for py_file in base_dir.rglob('*.py'):
        # 跳過腳本本身和遷移腳本
        if py_file.name in ['update_imports.py', 'migrate_agents.py']:
            continue

        if update_imports_in_file(py_file):
            updated_files.append(py_file)
            print(f"已更新: {py_file.relative_to(base_dir)}")

    print(f"\n總共更新了 {len(updated_files)} 個檔案")

    if updated_files:
        print("\n更新的檔案列表:")
        for file in updated_files:
            print(f"  - {file.relative_to(base_dir)}")

if __name__ == "__main__":
    main()
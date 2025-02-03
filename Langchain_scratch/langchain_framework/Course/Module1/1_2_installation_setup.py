"""
LangChain 0.3+ 安裝與環境設定教學
此範例展示如何設置 LangChain 開發環境並驗證安裝

需求套件:
- langchain>=0.3.0
- langchain-openai>=0.0.2
- python-dotenv>=0.19.0
"""

import sys
import pkg_resources
import subprocess
from pathlib import Path
from typing import Dict, List
import logging

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_python_version() -> bool:
    """
    檢查 Python 版本是否符合要求 (建議 3.9+)
    """
    current_version = sys.version_info
    required_version = (3, 9)

    if current_version >= required_version:
        logger.info(f"Python 版本檢查通過: {sys.version}")
        return True
    else:
        logger.warning(
            f"Python 版本過低: {sys.version}\n"
            f"建議使用 Python {required_version[0]}.{required_version[1]} 或更高版本"
        )
        return False


def get_required_packages() -> Dict[str, str]:
    """
    取得必要套件清單及其版本要求
    """
    return {
        "langchain": ">=0.3.0",
        "langchain-openai": ">=0.0.2",
        "langchain-community": ">=0.0.1",
        "python-dotenv": ">=0.19.0",
        "openai": ">=1.0.0"
    }


def check_installed_packages(required_packages: Dict[str, str]) -> List[str]:
    """
    檢查已安裝的套件版本，返回需要安裝的套件清單
    """
    packages_to_install = []

    for package, version in required_packages.items():
        try:
            installed = pkg_resources.get_distribution(package)
            logger.info(f"已安裝 {package} 版本: {installed.version}")
        except pkg_resources.DistributionNotFound:
            logger.warning(f"未安裝 {package}")
            packages_to_install.append(f"{package}{version}")

    return packages_to_install


def setup_env_file():
    """
    建立並設定 .env 檔案
    """
    env_path = Path(".env")

    if not env_path.exists():
        env_content = """# LangChain 環境設定
                        OPENAI_API_KEY=your-api-key-here
                        ANTHROPIC_API_KEY=your-api-key-here

                        # 可選設定
                        SERPAPI_API_KEY=your-api-key-here
                        GOOGLE_API_KEY=your-api-key-here
                        """
        env_path.write_text(env_content)
        logger.info("已建立 .env 檔案範本")
    else:
        logger.info(".env 檔案已存在")


def main():
    """
    主程式：執行環境檢查與設定
    """
    print("=== LangChain 0.3+ 環境設定精靈 ===\n")

    # 1. 檢查 Python 版本
    if not check_python_version():
        print("\n請更新 Python 版本後再繼續。")
        return

    # 2. 檢查必要套件
    required_packages = get_required_packages()
    packages_to_install = check_installed_packages(required_packages)

    # 3. 安裝缺少的套件
    if packages_to_install:
        print("\n正在安裝缺少的套件...")
        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install"
            ] + packages_to_install)
            logger.info("套件安裝完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"套件安裝失敗: {str(e)}")
            return

    # 4. 設定環境檔案
    setup_env_file()

    print("\n=== 環境設定完成 ===")
    print("""
        接下來您需要:
        1. 編輯 .env 檔案，填入您的 API 金鑰
        2. 確認環境變數已正確載入
            """)


if __name__ == "__main__":
    main()

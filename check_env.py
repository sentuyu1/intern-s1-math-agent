"""
环境检查脚本

运行方式：python check_env.py

它会检查：
1. Python 版本是否合适
2. 必要的包是否安装
3. 是否能找到 lagent
4. 是否能读取 API token（需要你先把 token 放到 .env 文件或环境变量）
"""

import sys
import importlib


def check_python_version():
    """检查 Python 版本。"""
    version = sys.version_info
    print(f"Python 版本: {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 版本过低，建议升级到 3.10 或更高")
        return False
    print("✅ Python 版本符合要求")
    return True


def check_package(package_name):
    """检查某个包是否安装。"""
    try:
        importlib.import_module(package_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"❌ {package_name} 未安装，请运行: pip install {package_name}")
        return False


def check_lagent():
    """检查 lagent 是否安装。"""
    try:
        import lagent
        print(f"✅ lagent 已安装，版本信息: {lagent.__version__ if hasattr(lagent, '__version__') else '未知'}")
        return True
    except ImportError:
        print("❌ lagent 未安装")
        print("   安装方法：")
        print("   git clone https://github.com/InternLM/lagent.git")
        print("   cd lagent")
        print("   pip install -e .")
        return False


def check_api_token():
    """检查 API token 是否配置。"""
    import os
    token = os.environ.get("INTERN_S1_API_KEY")

    # 也尝试从 .env 文件读取
    if not token and os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("INTERN_S1_API_KEY="):
                    token = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if token:
        print(f"✅ API token 已配置: {token[:10]}...")
        return True
    else:
        print("❌ API token 未配置")
        print("   配置方法（二选一）：")
        print("   1. 创建 .env 文件，写入：INTERN_S1_API_KEY=你的token")
        print("   2. 设置环境变量：set INTERN_S1_API_KEY=你的token")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("开始检查开发环境")
    print("=" * 50)

    results = []
    results.append(check_python_version())
    results.append(check_package("openai"))
    results.append(check_lagent())
    results.append(check_api_token())

    print("=" * 50)
    if all(results):
        print("🎉 环境检查全部通过，可以开始开发了！")
    else:
        print("⚠️ 环境检查未通过，请按上方提示修复后再试。")
    print("=" * 50)

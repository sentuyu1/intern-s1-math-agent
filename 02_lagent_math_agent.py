"""
小白入门示例 2：用 Lagent 搭建一个会写 Python 的数学智能体

运行前：
1. 安装 lagent：git clone https://github.com/InternLM/lagent.git && cd lagent && pip install -e .
2. 安装 openai sdk：pip install openai

说明：
- Lagent 是 Agent 框架，帮你组织"思考 -> 调用工具 -> 执行 -> 再思考"的循环
- 这个例子让模型自动生成 Python 代码来解数学题，并执行代码得到答案
"""

import re
import sys
import subprocess
sys.path.insert(0, r"D:\intern-s1项目\lagent")

from lagent.agents import Agent
from lagent.schema import AgentMessage
from lagent.llms import GPTAPI


# 1. 封装一个调用 Intern-S1 的 LLM 类
class InternS1API(GPTAPI):
    """让 Lagent 能调用书生 API 的 Intern-S1 模型。"""

    def __init__(self, api_key, model="intern-s1", **kwargs):
        super().__init__(
            model_type=model,
            key=api_key,
            api_base="https://chat.intern-ai.org.cn/api/v1/chat/completions",
            **kwargs,
        )


# 2. 配置 API token
API_KEY = "sk-6kkTxSeSZw5uoHQOvC4BNfxSrqTtJsz6iK4bwbQsAyiZmGhS"

llm = InternS1API(
    api_key=API_KEY,
    model="intern-s1",
    max_new_tokens=4096,
    temperature=0.2,
)

# 3. 定义系统提示词
system_prompt = """你是一名数学解题专家。请按以下步骤解决用户的数学题：
1. 理解题目，分析已知条件和求解目标；
2. 编写 Python 代码来精确计算或验证答案；
3. 将代码放在 ```python 和 ``` 之间；
4. 根据代码执行结果，给出最终答案和简要解释。
"""

# 4. 创建 Agent
agent = Agent(llm=llm)
agent.template = [dict(role="system", content=system_prompt)]


# 5. 工具函数：提取并执行 Python 代码
def extract_python_code(text: str) -> str | None:
    """从模型输出中提取第一个 Python 代码块。"""
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def execute_code(code: str) -> str:
    """通过 subprocess 执行 Python 代码并返回输出。"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return "代码执行超时"


# 6. 解题函数（最多尝试 3 轮）
def solve_math(problem: str, max_turn: int = 3):
    message = AgentMessage(sender="user", content=problem)

    for turn in range(max_turn):
        # 模型思考/生成代码
        message = agent(message)
        print(f"\n--- 第 {turn + 1} 轮模型输出 ---")
        print(message.content[:500] + "..." if len(message.content) > 500 else message.content)

        # 提取代码并执行
        code = extract_python_code(message.content)
        if code is None:
            # 模型没有生成代码，认为已经得出最终答案
            return message.content

        print(f"\n--- 代码执行结果 ---")
        exec_result = execute_code(code)
        print(exec_result)

        # 把执行结果作为下一轮输入
        message = AgentMessage(
            sender="user",
            content=f"代码执行结果：{exec_result}\n\n请根据这个结果给出最终答案。",
        )

    return message.content


# 7. 测试
if __name__ == "__main__":
    problem = "求 1 到 100 之间所有质数的和。"
    final_answer = solve_math(problem)
    print("\n=== 最终答案 ===")
    print(final_answer)

"""
Lagent 核心概念详解示例

把 Lagent 想象成一个公司：
- LLM（大模型）= 员工的大脑
- Agent = 员工本人，负责接收任务、思考、输出
- Action / Tool = 员工能使用的工具（比如计算器、搜索引擎、Python 解释器）
- ActionExecutor = 工具管理部门，负责找到正确的工具并执行
- Memory = 员工的记事本，记录对话历史
- Parser = 格式解析器，从模型输出里提取结构化信息
- Aggregator = 会议记录员，把 Memory 整理成模型能看懂的对话格式

本示例展示：
1. 如何自定义一个 LLM 类接入 Intern-S1
2. 如何创建一个 Agent
3. 如何让 Agent 使用 Python 工具解数学题
4. 如何查看 Memory
"""

from lagent.agents import Agent
from lagent.schema import AgentMessage
from lagent.llms import GPTAPI
from lagent.actions import ActionExecutor, IPythonInteractive
from lagent.prompts.parsers import ToolParser
from lagent.agents.aggregator import InternLMToolAggregator


# ===================== 第 1 步：定义 LLM =====================
class InternS1API(GPTAPI):
    """
    让 Lagent 能调用书生 Intern-S1 API。
    GPTAPI 是 Lagent 内置的、用于 OpenAI 风格 API 的封装。
    """

    def __init__(self, api_key, model="intern-s1", **kwargs):
        super().__init__(
            model_type=model,  # 模型名称
            key=api_key,       # API token
            url="https://chat.intern-ai.org.cn/api/v1/chat/completions",
            retry=3,           # 请求失败时重试 3 次
            **kwargs,
        )


# ===================== 第 2 步：创建 Agent =====================
# Agent 是 Lagent 的核心，它把 LLM、Memory、Parser、Aggregator 组合在一起。

llm = InternS1API(
    api_key="YOUR_API_TOKEN",  # ← 改成你的 token
    model="intern-s1",
    max_new_tokens=4096,
    temperature=0.2,
)

system_prompt = """你是一名数学解题专家。
请按以下步骤解决用户的数学题：
1. 分析题目；
2. 如果需要计算，编写 Python 代码；
3. 把代码放在 ```python 和 ``` 之间；
4. 给出最终答案。
"""

# ToolParser：从模型输出中提取 ```python ... ``` 里的代码
parser = ToolParser(
    tool_type="code interpreter",
    begin="```python\n",
    end="\n```\n",
)

agent = Agent(
    llm=llm,
    system_prompt=system_prompt,
    output_format=parser,           # 让 Agent 知道如何解析模型输出
    aggregator=InternLMToolAggregator(),  # 组织对话历史，支持工具调用
)


# ===================== 第 3 步：创建 ActionExecutor =====================
# ActionExecutor 负责执行工具。这里我们只给 Agent 配一个 Python 解释器。

executor = ActionExecutor(actions=[IPythonInteractive()])


# ===================== 第 4 步：运行解题循环 =====================
def solve(problem: str, max_turn: int = 3):
    """
    多轮解题循环：
    用户提问 -> Agent 思考/写代码 -> Executor 执行代码 -> Agent 再思考 -> ...
    """
    message = AgentMessage(sender="user", content=problem)

    for turn in range(max_turn):
        # 调用 Agent：模型会根据当前记忆生成回复
        message = agent(message)
        print(f"\n--- 第 {turn + 1} 轮 Agent 输出 ---")
        print(message.content)

        # 如果模型没有生成代码，说明已经给出最终答案
        if message.formatted is None or message.formatted.get("tool_type") is None:
            return message.content

        # 否则，执行代码
        message = executor(message)
        print(f"\n--- Executor 执行结果 ---")
        print(message.content)

    return message.content


# ===================== 第 5 步：查看 Memory =====================
def show_memory():
    """查看 Agent 记住了哪些对话。"""
    print("\n=== Agent 的 Memory ===")
    for msg in agent.memory.get_memory():
        print(f"[{msg.sender}] {msg.content[:100]}...")


# ===================== 第 6 步：测试 =====================
if __name__ == "__main__":
    problem = "求解方程 x^2 - 5x + 6 = 0 的所有实数根。"
    final = solve(problem)
    print("\n=== 最终答案 ===")
    print(final)
    show_memory()

    # 清空记忆，准备下一道题
    agent.reset()

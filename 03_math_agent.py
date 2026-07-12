"""
================================================================================
挑战杯 XH-202627：基于 Intern-S1 的数学智能体系统
================================================================================

使用方式：
  【方式 1】交互模式：   python 03_math_agent.py
  【方式 2】批量模式：   python 03_math_agent.py --batch problems.json
  【方式 3】导入使用：
           from math_agent import MathAgent, InternS1API
           agent = MathAgent(api_key="your-key")
           result = agent.solve("求 1+2+...+100 的和")

依赖安装：
  pip install lagent openai pdfplumber  (lagent 不太容易安装可能要用下面的方式)
  或者 clone lagent 到本地后把路径加到 sys.path
================================================================================
"""

import json
import re
import sys
import subprocess
import time
import argparse
from typing import Optional

# 把 lagent 克隆目录加入搜索路径（如果 pip install -e . 未成功）
sys.path.insert(0, r"D:\intern-s1项目\lagent")

from lagent.agents import Agent
from lagent.schema import AgentMessage
from lagent.llms import GPTAPI


# ============================================================
# ① Intern-S1 模型适配器 —— 对接书生平台 API
# ============================================================
class InternS1API(GPTAPI):
    """
    Intern-S1 通过 OpenAI 兼容接口访问。
    文档: https://internlm.intern-ai.org.cn/api/document
    """

    def __init__(self, api_key: str, model: str = "intern-s1", **kwargs):
        super().__init__(
            model_type=model,
            key=api_key,
            api_base="https://chat.intern-ai.org.cn/api/v1/chat/completions",
            # 以下两个参数是手动传入的，给你清晰的默认值
            max_new_tokens=kwargs.pop("max_new_tokens", 4096),
            temperature=kwargs.pop("temperature", 0.2),
            **kwargs,
        )


# ============================================================
# ② 系统提示词 —— 赛题要求的"启发式解释 + JSON 输出"
# ============================================================
SYSTEM_PROMPT = """你是一位顶尖的数学竞赛教练。请严格按照以下流程解决数学问题：

## 解题流程
1. **问题重述**：用简洁的话概括题目
2. **策略选择**：挑出最优解法并解释为什么
3. **逐步推导**：写出详细的数学推导，关键步骤加说明
4. **计算验证**：如果涉及计算，写 Python 代码（放在 ```python ... ``` 里）
5. **最终答案**：用 \\boxed{答案} 的 LaTeX 格式给出

## 输出格式（必须输出 JSON）
```json
{
  "problem": "原题",
  "solution": {
    "understanding": "对问题的理解",
    "strategy": "选用的解题策略和理由",
    "derivation": "完整的推导过程",
    "code": "执行的 Python 代码（如果没有则为空字符串）",
    "code_output": "代码输出结果（如果没有则为空字符串）",
    "final_answer": "LaTeX 格式的最终答案"
  },
  "educational_insight": "知识点总结、解题技巧、易错点"
}
```
"""


# ============================================================
# ③ 工具函数
# ============================================================
def extract_code_blocks(text: str) -> list[str]:
    """提取文本中所有 ```python ... ``` 代码块。"""
    return re.findall(r"```python\n(.*?)```", text, re.DOTALL)


def execute_code(code: str, timeout: int = 30) -> dict:
    """安全地执行一段 Python 代码，返回结果字典。"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
        )
        return {
            "stdout": r.stdout.strip(),
            "stderr": r.stderr.strip(),
            "ok": r.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": "超时", "ok": False}


def extract_json(text: str) -> dict | None:
    """从文本中提取第一个合法的 JSON 对象（优先找代码块里的）。"""
    # 先找 ```json ... ```
    m = re.search(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    try:
        if m:
            return json.loads(m.group(1))
    except json.JSONDecodeError:
        pass
    # 再找最外层 { ... }
    m = re.search(r"\{[\s\S]*\"problem\"[\s\S]*\}", text)
    try:
        if m:
            return json.loads(m.group(0))
    except json.JSONDecodeError:
        pass
    return None


# ============================================================
# ④ 核心智能体类
# ============================================================
class MathAgent:
    """
    数学解题智能体。

    参数：
        api_key  (str): Intern-S1 API key
        model    (str): 模型名，默认 intern-s1

    用法：
        agent = MathAgent(api_key="sk-xxxx")
        result = agent.solve("求 1 到 100 所有质数的和")
        print(result["solution"]["final_answer"])
    """

    def __init__(self, api_key: str, model: str = "intern-s1"):
        self.api_key = api_key
        self.llm = InternS1API(api_key=api_key, model=model)
        self.agent = Agent(llm=self.llm, name="MathSolver")
        self.agent.template = [{"role": "system", "content": SYSTEM_PROMPT}]

    def solve(self, problem: str, max_rounds: int = 3, verbose: bool = False) -> dict:
        """
        求解一道数学题。

        参数：
            problem    (str): 数学题文本
            max_rounds (int): 最多 Agent 交互轮数
            verbose   (bool): 是否打印中间过程

        返回：
            dict: 包含 solution、educational_insight 的完整结果
        """
        msg = AgentMessage(sender="user", content=problem)
        all_outputs = []

        for round_idx in range(max_rounds):
            msg = self.agent(msg, session_id=0)
            text = msg.content
            if verbose:
                print(f"\n--- 第 {round_idx+1} 轮 ---")
                print(text[:600])

            # 提取并执行代码
            codes = extract_code_blocks(text)
            if codes:
                for code in codes:
                    exec_r = execute_code(code)
                    out = exec_r["stdout"] or exec_r["stderr"]
                    if out:
                        all_outputs.append(out)
                if all_outputs:
                    msg = AgentMessage(
                        sender="user",
                        content=f"代码输出：{all_outputs[-1]}。请给出最终的 JSON 结果。",
                    )
                    continue

            # 尝试解析 JSON
            result = extract_json(text)
            if result:
                result.setdefault("id", "")
                result.setdefault("elapsed_seconds", 0)
                return result

            # 兜底：把纯文本包进 JSON
            return {
                "problem": problem,
                "solution": {
                    "understanding": "",
                    "strategy": "",
                    "derivation": text,
                    "code": "",
                    "code_output": "",
                    "final_answer": "见推导",
                },
                "educational_insight": "",
            }

        return {"error": f"超过 {max_rounds} 轮，未能得到 JSON 结果"}

    def batch_solve(
        self, problems: list[str], verbose: bool = True
    ) -> list[dict]:
        """批量求解，返回结果列表。"""
        results = []
        total = len(problems)
        for i, p in enumerate(problems):
            print(f"\n[{i+1}/{total}] {p[:60]}...")
            t0 = time.time()
            r = self.solve(p, verbose=verbose)
            r["id"] = str(i + 1)
            r["elapsed_seconds"] = round(time.time() - t0, 1)
            results.append(r)
            # 打印简要结果
            ans = r.get("solution", {}).get("final_answer", "?")
            print(f"  → 答案: {ans}  (耗时 {r['elapsed_seconds']}s)")
        return results


# ============================================================
# ⑤ CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Intern-S1 数学智能体 — 挑战杯 XH-202627 参赛方案"
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="JSON 文件路径，包含题目列表"
    )
    parser.add_argument(
        "--api-key", type=str,
        default="sk-6kkTxSeSZw5uoHQOvC4BNfxSrqTtJsz6iK4bwbQsAyiZmGhS",
        help="Intern-S1 API key"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果输出 JSON 文件路径"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="打印详细过程"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Intern-S1 数学智能体系统 (基于 Lagent)")
    print("赛题 XH-202627：基于 Intern-S1 的数学智能体设计与推理创新")
    print("=" * 60)

    agent = MathAgent(api_key=args.api_key)

    if args.batch:
        # 批量模式：从 JSON 文件读题
        with open(args.batch, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            problems = [p if isinstance(p, str) else p.get("problem", str(p)) for p in data]
        else:
            problems = [data["problem"]] if "problem" in data else list(data.values())
        print(f"\n共 {len(problems)} 道题\n")
        results = agent.batch_solve(problems)
    else:
        # 交互模式
        print("\n交互模式：输入数学题，输入 quit 退出\n")
        results = []
        idx = 1
        while True:
            problem = input("题目> ").strip()
            if problem.lower() in ("quit", "q", "exit"):
                break
            if not problem:
                continue
            print("思考中...")
            t0 = time.time()
            result = agent.solve(problem)
            result["id"] = str(idx)
            result["elapsed_seconds"] = round(time.time() - t0, 1)
            results.append(result)

            ans = result.get("solution", {}).get("final_answer", "?")
            insight = result.get("educational_insight", "")
            print(f"\n答案: {ans}")
            if insight:
                print(f"启发: {insight}")
            print("-" * 40)
            idx += 1

    # 保存结果
    output = args.output or f"math_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {output}")
    print(f"共完成 {len(results)} 题")


if __name__ == "__main__":
    main()

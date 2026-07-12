"""
直接攻克运输问题和三次样条问题 —— 用代码优先策略
"""
import sys
sys.path.insert(0, r"D:\intern-s1项目\lagent")

from lagent.agents import Agent
from lagent.schema import AgentMessage
from lagent.llms import GPTAPI


class InternS1API(GPTAPI):
    def __init__(self, api_key, model="intern-s1", **kwargs):
        super().__init__(
            model_type=model, key=api_key,
            api_base="https://chat.intern-ai.org.cn/api/v1/chat/completions",
            max_new_tokens=8192, temperature=0.2, **kwargs,
        )


API_KEY = "sk-6kkTxSeSZw5uoHQOvC4BNfxSrqTtJsz6iK4bwbQsAyiZmGhS"
llm = InternS1API(api_key=API_KEY)


# ====== 问题 1: 运输问题（代码优先） ======
prompt1 = """你是运筹学专家。请直接用 Python 代码完成以下运输问题的求解。只输出代码，不要用手推导。

【问题】
3 个仓库供应量: supply = [20, 30, 25]
4 个零售点需求量: demand = [15, 20, 18, 22]
成本矩阵: cost = [[2,4,5,1],[3,1,6,4],[5,2,3,2]]

请编写 Python 代码：
1. 用最小元素法得到初始解（分配矩阵 x[i][j]）
2. 用位势法（u_i + v_j = c_ij 对基变量）计算检验数
3. 选一个负检验数对应的非基变量，构造闭回路，计算 theta，进行一步优化
4. 输出: 初始总成本、优化后总成本、优化节省了多少元

提示：
- 闭回路找法：从进基变量出发，DFS 交替行进在基变量中找回路
- 也可直接用 scipy.optimize.linprog 或 pulp 验证

只输出 Python 代码即可。"""

agent1 = Agent(llm=llm, name="Transport")
agent1.template = [{"role": "system", "content": prompt1}]
msg1 = agent1(AgentMessage(sender="user", content="Write Python code to solve the transport problem."), session_id=1)
print("=== 运输问题输出 ===")
print(msg1.content[:3000])
print("\n" + "="*60)


# ====== 问题 2: 三次样条（代码优先） ======
prompt2 = """你是数值分析专家。请直接用 Python 代码完成三次样条插值计算。只输出代码，不要手推。

【问题】
数据点: (0,0), (1,1), (2,8), (3,27)
自然边界条件: S''(0) = S''(3) = 0
区间长度: h_i = 1 (统一)

三弯矩方程组: [[4, 1], [1, 4]] · [M1, M2] = [36, 72]

对于区间 [1,2]，样条表达式为 S(x) = a + b*(x-1) + c*(x-1)^2 + d*(x-1)^3

请用 numpy 解出 M1, M2，然后计算 a, b, c, d 的值。
只输出 Python 代码即可。"""

agent2 = Agent(llm=llm, name="Spline")
agent2.template = [{"role": "system", "content": prompt2}]
msg2 = agent2(AgentMessage(sender="user", content="Write Python code to compute the cubic spline coefficients."), session_id=2)
print("=== 三次样条输出 ===")
print(msg2.content[:3000])

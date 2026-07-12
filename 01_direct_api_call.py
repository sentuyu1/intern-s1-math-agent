"""
小白入门示例 1：直接调用 Intern-S1 API 解数学题

运行前：
1. 安装依赖：pip install openai
2. 去 https://studio.intern-ai.org.cn/console/dashboard 获取 API token
3. 把下面的 YOUR_API_TOKEN 换成你的真实 token
"""

from openai import OpenAI

# 1. 配置客户端
client = OpenAI(
    api_key="YOUR_API_TOKEN",  # ← 改成你的 token
    base_url="https://chat.intern-ai.org.cn/api/v1/",
)

# 2. 准备一道数学题
math_problem = """
求函数 f(x) = x^2 - 4x + 3 的最小值。
请用中文给出推理过程和最终答案，最终答案用 JSON 格式输出：
{"answer": "...", "reasoning": "..."}
"""

# 3. 调用 Intern-S1（开启深度思考模式）
response = client.chat.completions.create(
    model="intern-s1",  # 也可换成 intern-s1-pro / intern-s1-mini
    messages=[
        {"role": "system", "content": "你是一位数学专家，擅长严谨的数学推理。"},
        {"role": "user", "content": math_problem},
    ],
    thinking_mode=True,  # 开启长思考，适合复杂数学题
    temperature=0.2,     # 越低越稳定
    max_tokens=4096,
)

# 4. 打印结果
answer = response.choices[0].message.content
print("=== 模型输出 ===")
print(answer)

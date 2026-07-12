"""
自动评分脚本

功能：
1. 读取模型输出的 results.json
2. 读取带标准答案的 sample_18_domains.json
3. 逐题对比，计算准确率
4. 按子领域统计正确率

使用方法：
python evaluate.py --results results.json --answers sample_18_domains.json

评分规则说明：
- 默认使用字符串包含匹配（不区分大小写、忽略空格）
- 例如标准答案是 "2πi"，模型回答 "结果是 2πi" 也算对
- 你可以根据实际需要修改 normalize 函数
"""

import json
import argparse
from collections import defaultdict


def normalize(text: str) -> str:
    """
    标准化答案文本，用于比较。
    去掉空格、换行、标点，统一小写。
    """
    if not text:
        return ""
    text = str(text).lower()
    # 去掉常见标点符号
    for char in " .,;:!?，。；：！？\"'\"`()（）[]【】{}":
        text = text.replace(char, "")
    # 去掉多余空白
    text = "".join(text.split())
    return text


def is_correct(pred: str, gold: str) -> bool:
    """
    判断预测答案是否正确。
    策略：如果 gold 出现在 pred 中，或 pred 出现在 gold 中，则认为正确。
    """
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)

    if not pred_norm or not gold_norm:
        return False

    # 完全相等
    if pred_norm == gold_norm:
        return True

    # 包含匹配
    if gold_norm in pred_norm or pred_norm in gold_norm:
        return True

    return False


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(results, answers):
    """评估模型输出。"""
    # 构建标准答案字典
    answer_dict = {item["id"]: item for item in answers}

    total = 0
    correct = 0
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    details = []

    for result in results:
        problem_id = result["id"]
        if problem_id not in answer_dict:
            print(f"警告：找不到题目 {problem_id} 的标准答案，跳过")
            continue

        gold = answer_dict[problem_id]["answer"]
        pred = result.get("answer", "")
        domain = answer_dict[problem_id].get("domain", "未知")

        ok = is_correct(pred, gold)

        total += 1
        domain_stats[domain]["total"] += 1
        if ok:
            correct += 1
            domain_stats[domain]["correct"] += 1

        details.append({
            "id": problem_id,
            "domain": domain,
            "predicted": pred,
            "gold": gold,
            "correct": ok,
        })

    # 打印总结果
    print("\n" + "=" * 60)
    print(f"总题数：{total}")
    print(f"正确数：{correct}")
    print(f"准确率：{correct / total * 100:.2f}%" if total > 0 else "N/A")
    print("=" * 60)

    # 按领域打印
    print("\n各子领域正确率：")
    for domain in sorted(domain_stats.keys()):
        stat = domain_stats[domain]
        acc = stat["correct"] / stat["total"] * 100 if stat["total"] > 0 else 0
        print(f"  {domain:12s}: {stat['correct']}/{stat['total']} ({acc:.1f}%)")

    # 打印错题
    wrong = [d for d in details if not d["correct"]]
    if wrong:
        print("\n错题详情：")
        for d in wrong:
            print(f"  [{d['id']}] {d['domain']}")
            print(f"    预测：{d['predicted']}")
            print(f"    标准：{d['gold']}")

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "domain_stats": dict(domain_stats),
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="评估模型解题结果")
    parser.add_argument("--results", default="results.json", help="模型输出结果文件")
    parser.add_argument("--answers", default="sample_18_domains.json", help="带标准答案的文件")
    parser.add_argument("--output", default="evaluation_result.json", help="评估结果保存路径")
    args = parser.parse_args()

    results = load_json(args.results)
    answers = load_json(args.answers)

    eval_result = evaluate(results, answers)

    # 保存评估结果
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)

    print(f"\n评估结果已保存到：{args.output}")


if __name__ == "__main__":
    main()

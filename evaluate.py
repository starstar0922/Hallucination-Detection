# -*- coding: utf-8 -*-
"""对照 ground_truth.json 验证检测结果:准确率/精确率/召回率/F1、漏检与误报清单。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results" / "detection_results.json"))
    ap.add_argument("--truth", default=str(ROOT / "task4_ground_truth.json"))
    ap.add_argument("--report", default=str(ROOT / "results" / "evaluation_report.md"))
    args = ap.parse_args()

    preds = {r["id"]: r for r in load(args.results)["results"]}
    truths = {t["id"]: t for t in load(args.truth)}
    assert set(preds) == set(truths), "预测与标注的 id 不一致"
    replies = {r["id"]: r for r in load(ROOT / "task4_replies.json")}

    ids = sorted(truths)
    tp = fp = tn = fn = 0
    fp_cases, fn_cases = [], []
    type_counter = Counter()
    gt_type_hit = defaultdict(lambda: [0, 0])  # gt_type -> [detected, total]

    for i in ids:
        t, p = truths[i], preds[i]
        gt_h, pred_h = t["is_hallucination"], p["is_hallucination"]
        if gt_h and pred_h:
            tp += 1
            type_counter["TP"] += 1
        elif gt_h and not pred_h:
            fn += 1
            fn_cases.append(i)
        elif not gt_h and pred_h:
            fp += 1
            fp_cases.append(i)
        else:
            tn += 1
        gt_type_hit[t["hallucination_type"] or "非幻觉"][1] += 1
        if gt_h and pred_h:
            gt_type_hit[t["hallucination_type"]][0] += 1

    accuracy = (tp + tn) / len(ids)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    lines = [
        "# 幻觉检测验证报告",
        "",
        f"- 检测器:{load(args.results)['meta']['detector']}",
        f"- 样本总数:{len(ids)}  |  幻觉标注:{sum(1 for t in truths.values() if t['is_hallucination'])}  |  非幻觉标注:{sum(1 for t in truths.values() if not t['is_hallucination'])}",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| 准确率 Accuracy | {accuracy:.2%} |",
        f"| 精确率 Precision | {precision:.2%} |",
        f"| 召回率 Recall | {recall:.2%} |",
        f"| F1 | {f1:.4f} |",
        "",
        f"| | 预测幻觉 | 预测正常 |",
        f"| --- | --- | --- |",
        f"| 实际幻觉 | TP={tp} | FN={fn} |",
        f"| 实际正常 | FP={fp} | TN={tn} |",
        "",
        f"- 漏检(实际幻觉、预测正常):{'、'.join(fn_cases) if fn_cases else '无'}",
        f"- 误报(实际正常、预测幻觉):{'、'.join(fp_cases) if fp_cases else '无'}",
        "",
        "## 各 GT 类型的召回情况",
        "",
        "| GT 类型 | 检出/总数 | 检出率 |",
        "| --- | --- | --- |",
    ]
    for gt_type in sorted(gt_type_hit, key=lambda k: -gt_type_hit[k][1]):
        hit, total = gt_type_hit[gt_type]
        lines.append(f"| {gt_type} | {hit}/{total} | {hit / total:.0%} |")

    lines += [
        "",
        "## 逐条对比",
        "",
        "| id | 用户问题 | GT | 预测 | 一致 | GT类型 | 预测类型 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i in ids:
        t, p = truths[i], preds[i]
        match = "✓" if t["is_hallucination"] == p["is_hallucination"] else "✗"
        lines.append(
            f"| {i} | {replies[i]['user_question'][:18]} | {'幻觉' if t['is_hallucination'] else '正常'} "
            f"| {'幻觉' if p['is_hallucination'] else '正常'} | {match} "
            f"| {t['hallucination_type'] or '-'} | {p['type'] if p['is_hallucination'] else '-'} |"
        )

    report = "\n".join(lines)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n报告已保存: {out}")


if __name__ == "__main__":
    main()

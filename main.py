# -*- coding: utf-8 -*-
"""批量检测入口:对 task4_replies.json 的 20 条回复逐条检测并保存结果。

用法:
  python main.py                 # 真实 LLM(DeepSeek)
  python main.py --mock          # 规则兜底模式
"""
import argparse
import json
import sys
import time
from pathlib import Path

from detector import create_detector, load_api_key, MODEL

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="客服回复幻觉批量检测")
    ap.add_argument("--input", default=str(ROOT / "task4_replies.json"))
    ap.add_argument("--output", default=str(ROOT / "results" / "detection_results.json"))
    ap.add_argument("--mock", action="store_true", help="无 API 时的规则兜底模式")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        items = json.load(f)

    detector = create_detector(mock=args.mock)
    mode = "mock" if args.mock else MODEL

    results = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        r = detector.detect(item)
        results.append(r)
        flag = "幻觉" if r["is_hallucination"] else "正常"
        print(f"[{i:2d}/{len(items)}] {item['id']} -> {flag} ({r['type']}, {r['severity']})")
        if i < len(items):
            time.sleep(0.3)  # 避免触发限流

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "detector": mode,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(items),
        "hallucination_count": sum(1 for r in results if r["is_hallucination"]),
    }
    if hasattr(detector, "usage"):
        meta["usage_tokens"] = detector.usage
    payload = {"meta": meta, "results": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n完成:{len(items)} 条,检测出幻觉 {payload['meta']['hallucination_count']} 条,"
          f"耗时 {time.time() - t0:.1f}s")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()

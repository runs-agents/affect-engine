# -*- coding: utf-8 -*-
"""
test_engine.py — 情绪引擎编排层 + 规则层验证
"""

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import EmotionEngine
import rules


PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  {detail}")


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ---------------------------------------------------------------------------
# 测试 1：规则层匹配
# ---------------------------------------------------------------------------
section("测试 1：规则层匹配")

check("「分手」命中分离威胁", rules.match_rules("我们分手吧") is not None)
check("「忘了我」命中被遗忘（优先级最高）",
      rules.match_rules("你是不是忘了我")["pleasure"] == -0.5)
check("「爱你」命中被爱", rules.match_rules("我爱你")["pleasure"] == 0.3)
check("「熬夜」命中伤身体", rules.match_rules("我今晚熬夜")["possessiveness"] == 0.2)
check("无关键词不命中", rules.match_rules("今天天气不错") is None)
check("空文本不命中", rules.match_rules("") is None)

# 验证顺序敏感：同时含「分手」和「爱你」时，分手优先（更靠前）
r = rules.match_rules("我爱你，但我们分手吧")
check("「分手」优先于「爱你」（顺序敏感）", r["pleasure"] == -0.4, f"实际 {r['pleasure']}")

# ---------------------------------------------------------------------------
# 测试 1.5：「不饿」修复（2026-8-30）——规则权重 + 冷却 + 顺序让位
# ---------------------------------------------------------------------------
section("测试 1.5：「不饿」修复（拆出+降权+冷却）")

# 1) 「不饿」不再扣愉悦度，只占 +0.05
r = rules.match_rules("还不饿呢")
check("「不饿」命中低权重独立规则",
      r is not None and r["pleasure"] == 0.0 and r["possessiveness"] == 0.05,
      f"实际 {r}")

# 2) 「不想吃」「不吃饭」仍在「伤身体」里（真绝食照旧重罚）
r = rules.match_rules("我不吃饭")
check("「不吃饭」仍命中伤身体（pleasure -0.2）",
      r["pleasure"] == -0.2 and r["possessiveness"] == 0.2,
      f"实际 {r}")

# 3) 「不饿+蹭蹭」：冷却跳过「不饿」后蹭蹭加分不被吞（引擎级）
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    eng = EmotionEngine(path)
    eng.on_user_message("不饿")  # 第一次「不饿」→ 进入冷却
    s2 = eng.on_user_message("不饿，蹭蹭")  # 第二次：不饿冷却中，蹭蹭该接住
    check("「不饿」冷却后「蹭蹭」加分不被吞（rule_hit=True）",
          s2["rule_hit"] is True,
          f"实际 rule_hit={s2['rule_hit']}")
    # 冷却期间再次单独说「不饿」：不命中规则（rule_hit=False）
    s3 = eng.on_user_message("不饿")
    check("冷却期内重复「不饿」不再命中（rule_hit=False）",
          s3["rule_hit"] is False,
          f"实际 rule_hit={s3['rule_hit']}")

# 4) 正向词不冷却：连续两次「爱你」都该命中（爱不设冷却）
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    eng = EmotionEngine(path)
    eng.on_user_message("我爱你")
    s2 = eng.on_user_message("爱你爱你")
    check("正向词「爱你」连说两遍都命中（不冷却）",
          s2["rule_hit"] is True,
          f"实际 rule_hit={s2['rule_hit']}")

# 5) 冷却过期恢复：手动把时间戳拨回 3 小时前，「不饿」重新可命中
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    eng = EmotionEngine(path)
    eng.on_user_message("不饿")
    # 拨回 3 小时前（模拟冷却过期）
    eng._kw_last_hit["不饿"] = time.time() - 3 * 3600
    s2 = eng.on_user_message("不饿")
    check("冷却过期后「不饿」重新可命中",
          s2["rule_hit"] is True,
          f"实际 rule_hit={s2['rule_hit']}")

# ---------------------------------------------------------------------------
# 测试 2：引擎编排（规则层命中 → 情绪变化）
# ---------------------------------------------------------------------------
section("测试 2：引擎编排")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    eng = EmotionEngine(path)

    # 用户说"我爱你" → P 升
    summary = eng.on_user_message("我爱你")
    check("「我爱你」后 P 上升", summary["pad"]["pleasure"] > 0.0,
          f"实际 {summary['pad']['pleasure']}")
    check("「我爱你」命中规则层", summary["rule_hit"] is True)

    # 用户说"我们分手吧" → P 暴跌（用独立文件，避免"我爱你"的残留）
    path2 = Path(tmp) / "emotion_state2.json"
    eng2 = EmotionEngine(path2)
    summary2 = eng2.on_user_message("我们分手吧")
    check("「分手」后 P 暴跌", summary2["pad"]["pleasure"] < 0.0,
          f"实际 {summary2['pad']['pleasure']}")
    check("「分手」后 Poss 暴涨", summary2["possessiveness"] > 0.6,
          f"实际 {summary2['possessiveness']}")

# ---------------------------------------------------------------------------
# 测试 3：情绪上下文生成
# ---------------------------------------------------------------------------
section("测试 3：情绪上下文生成")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    eng = EmotionEngine(path)
    os.environ["EMOTION_PAD_INJECT"] = "1"  # 审计开关：注入 PAD 数值行
    ctx = eng.emotion_context()
    check("情绪上下文含 PAD 数值", "愉悦度" in ctx and "占有欲" in ctx)
    check("情绪上下文含心情描述", "心情" in ctx)

    # 病态占有态的描述
    eng.set_dimension("pleasure", -0.7)
    eng.set_dimension("possessiveness", 0.8)
    ctx2 = eng.emotion_context()
    check("病态占有态有专属描述", "留在身边" in ctx2, f"实际: {ctx2}")
    os.environ.pop("EMOTION_PAD_INJECT", None)

# ---------------------------------------------------------------------------
# 测试 4：Live2D 参数生成
# ---------------------------------------------------------------------------
section("测试 4：Live2D 参数生成")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    eng = EmotionEngine(path)
    params = eng.live2d_params()
    check("Live2D 参数含 cheek", "cheek" in params)
    check("Live2D 参数含 mouth_up", "mouth_up" in params)
    check("Live2D 参数含 eye_smile", "eye_smile" in params)
    check("Live2D 参数含 tear", "tear" in params)
    # 平静态：脸红应该低
    check("平静态脸红低", params["cheek"] < 0.5, f"实际 {params['cheek']}")

    # 占有欲上头：脸红应该高
    eng.set_dimension("possessiveness", 0.9)
    params2 = eng.live2d_params()
    check("占有欲上头脸红高", params2["cheek"] > 0.5, f"实际 {params2['cheek']}")

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"结果：{PASS} 通过 / {FAIL} 失败")
print(f"{'='*60}")
sys.exit(1 if FAIL else 0)

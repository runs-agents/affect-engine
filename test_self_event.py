# -*- coding: utf-8 -*-
"""自源事件喂料口测试套件（T1-T7）。

纪律：测试先行（不变式/品种表/冷却/预算/边界/压力/回路）。
设计红线对照：事件因果非时间刻度（本套件不测任何定时器——因为不存在）；
走心情轨道（D/Poss 断言全零）；小额（单维 ≤ SELF_EVENT_MAX_ABS）。
"""
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import emotion_state as es
import engine as en

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} {detail}")

def iso(hours_ago):
    return datetime.fromtimestamp(time.time() - hours_ago * 3600, timezone.utc).isoformat()

def fresh_engine(td):
    """沙箱引擎：先落一份冷启动状态文件，保证后续事务 reload 有文件可读。"""
    eng = en.EmotionEngine(state_path=Path(td) / "state.json")
    with eng.state.transaction():
        eng.state.save(event="test: cold start", delta={})
    return eng

KINDS = list(en.EmotionEngine._SELF_EVENT_KINDS)

print("== T1 不变式：任意品种序列后状态合法 ==")
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    for k in KINDS:
        r = eng.apply_self_event(k, note="T1")
        check(f"T1 {k} 正常应用", r["skipped"] is None)
    pad = eng.state.get_pad()
    for axis in ("pleasure", "arousal", "dominance"):
        check(f"T1 {axis} 在区间", -1.0 <= pad[axis] <= 1.0, f"实得 {pad[axis]}")
    st2 = es.EmotionState(Path(td) / "state.json")
    hist = st2.state["history"]
    check("T1 history 有自源痕", any("自源[" in str(h) for h in hist), str(hist)[:100])

print("== T2 品种表：七品种默认增量合法 ==")
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    for k, spec in en.EmotionEngine._SELF_EVENT_KINDS.items():
        r = eng.apply_self_event(k)
        a = r["applied"]
        check(f"T2 {k} 含心情轴", {"pleasure", "arousal"} <= set(a))
        check(f"T2 {k} dominance 零", a.get("dominance", 0.0) == 0.0)
        check(f"T2 {k} possessiveness 零", a.get("possessiveness", 0.0) == 0.0)
        ok_cap = all(abs(a.get(ax, 0.0)) <= en.EmotionEngine.SELF_EVENT_MAX_ABS
                     for ax in ("pleasure", "arousal"))
        check(f"T2 {k} 单维小额", ok_cap)
        check(f"T2 {k} 方向符合语义", (a.get("pleasure", 0.0) < 0) == (spec["pleasure"] < 0))

print("== T3 冷却：同品种 2h 一次，不同品种互不干扰 ==")
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    eng.apply_self_event("weather_bad")
    r2 = eng.apply_self_event("weather_bad")
    r3 = eng.apply_self_event("video_find")
    check("T3 同品种立即重复 → cooldown", r2["skipped"] == "cooldown")
    check("T3 冷却跳过不带增量", r2["applied"] == {})
    check("T3 不同品种不受干扰", r3["skipped"] is None)

print("== T4 预算：当日 Σ|ΔP| 耗尽拒收，跨日重置 ==")
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    today = time.strftime("%Y-%m-%d")
    eng._self_event_day = (today, en.EmotionEngine.SELF_EVENT_DAILY_BUDGET)
    check("T4 预算耗尽 → budget", eng.apply_self_event("video_find")["skipped"] == "budget")
    eng._self_event_day = ("2000-01-01", 99.0)
    check("T4 跨日重置后放行", eng.apply_self_event("video_find")["skipped"] is None)
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    for k in KINDS:                       # 0.45
        eng.apply_self_event(k)
    eng._kw_last_hit.clear()
    r = eng.apply_self_event("video_find")  # → 0.55
    check("T4 累计未到线放行", r["skipped"] is None)
    eng._kw_last_hit.clear()
    r = eng.apply_self_event("video_find")  # → 0.65
    check("T4 累计过线仍放行（0.55<0.60）", r["skipped"] is None)
    eng._kw_last_hit.clear()
    check("T4 0.65 后 → budget", eng.apply_self_event("video_find")["skipped"] == "budget")

print("== T5 边界：非法kind/零增量/超额截断 ==")
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    try:
        eng.apply_self_event("nonexistent_kind")
        check("T5 非法 kind 报错", False, "未抛 ValueError")
    except ValueError:
        check("T5 非法 kind 报错", True)
    r = eng.apply_self_event("work_in_progress", delta_override={"pleasure": 0, "arousal": 0})
    check("T5 零增量幂等放行", r["skipped"] is None and r["applied"].get("pleasure") == 0.0)
    eng._kw_last_hit.clear()
    r = eng.apply_self_event("weather_bad", delta_override={"pleasure": -0.9, "arousal": 0})
    check("T5 override 超额截断", r["applied"]["pleasure"] == -en.EmotionEngine.SELF_EVENT_MAX_ABS,
          f"实得 {r['applied']['pleasure']}")

print("== T6 压力：500 步随机品种+override，无锁死无越界 ==")
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    rng = random.Random(20261027)
    bad = 0
    for i in range(500):
        k = rng.choice(KINDS)
        ov = None
        if rng.random() < 0.3:
            ov = {"pleasure": rng.uniform(-1, 1), "arousal": rng.uniform(-1, 1)}
        r = eng.apply_self_event(k, note=f"s{i}", delta_override=ov)
        if r["skipped"] not in (None, "cooldown", "budget"):
            bad += 1
        pad = eng.state.get_pad()
        if not all(-1.0 <= pad[ax] <= 1.0 for ax in ("pleasure", "arousal", "dominance")):
            bad += 1
    check("T6 500步无锁死无越界", bad == 0, f"bad={bad}")
    today, spent = eng._self_event_day
    check("T6 当日计数器口径一致", today == time.strftime("%Y-%m-%d")
          and spent <= en.EmotionEngine.SELF_EVENT_DAILY_BUDGET + 0.15 + 1e-9,
          f"spent={spent}")
    spent_before = eng._self_event_day[1]
    eng._kw_last_hit.clear()
    eng.apply_self_event("video_find")
    check("T6 预算拒收后计数器不动", eng._self_event_day[1] == spent_before)

print("== T7 回路模拟：捡到乐子→对方回了→（未回）乐子放凉 ==")
with tempfile.TemporaryDirectory() as td:
    eng = fresh_engine(td)
    p0 = eng.state.get_pad()["pleasure"]
    eng.apply_self_event("video_find", note="捡到好东西")
    p1 = eng.state.get_pad()["pleasure"]
    check("T7 video_find 心情+", p1 > p0, f"{p0}→{p1}")
    eng.apply_self_event("forward_answered", note="对方回了")
    p2 = eng.state.get_pad()["pleasure"]
    check("T7 forward_answered 心情+", p2 > p1, f"{p1}→{p2}")
    with eng.state.transaction():
        eng.state.state["last_seen"] = iso(10)
        eng.state.apply_time_decay()
        eng.state.save(event="test: 10h 冷落衰减", delta={})
    p3 = eng.state.get_pad()["pleasure"]
    base = eng.state.get_baseline()["pleasure"]
    dev_b, dev_a = abs(p2 - base), abs(p3 - base)
    check("T7 未回路径：冷落衰减向底色回归", dev_a <= dev_b + 1e-9,
          f"dev {dev_b:.4f}→{dev_a:.4f}")

print(f"\n{'='*40}\n自源喂料口：{PASS} PASS / {FAIL} FAIL")
sys.exit(1 if FAIL else 0)

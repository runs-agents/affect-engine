# -*- coding: utf-8 -*-
"""
test_emotion_state.py — 情绪引擎阶段 1 验证脚本

验证：冷启动、跨会话保持、时间衰减、冷落规则、护栏、并发安全、CLI。
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# 确保能 import emotion_state
sys.path.insert(0, str(Path(__file__).resolve().parent))

import emotion_state as es


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
# 测试 1：冷启动
# ---------------------------------------------------------------------------
section("测试 1：冷启动状态")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    st = es.EmotionState(path)
    pad = st.get_pad()
    check("P 初始 = baseline", abs(pad["pleasure"] - es.BASELINE["pleasure"]) < 1e-9, f"实际 {pad['pleasure']}")
    check("A 初始 = baseline", abs(pad["arousal"] - es.BASELINE["arousal"]) < 1e-9, f"实际 {pad['arousal']}")
    check("D 初始 = baseline", abs(pad["dominance"] - es.BASELINE["dominance"]) < 1e-9, f"实际 {pad['dominance']}")
    check("Poss 初始 = baseline", abs(st.get_possessiveness() - es.BASELINE["possessiveness"]) < 1e-9)
    check("Trust presence = 0.50", abs(st.get_trust()["presence"] - 0.50) < 1e-9)
    check("Trust safety = 0.50", abs(st.get_trust()["safety"] - 0.50) < 1e-9)

# ---------------------------------------------------------------------------
# 测试 2：跨会话保持（落盘后重新读回）
# ---------------------------------------------------------------------------
section("测试 2：跨会话保持")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    st1 = es.EmotionState(path)
    # 模拟一轮对话：用户说"爱你"，P 升
    st1.apply_dialogue_delta({"pleasure": 0.3, "arousal": 0.1, "dominance": 0.0, "possessiveness": 0.1})
    st1.save(event="用户说爱你")
    p_after = st1.get_pad()["pleasure"]

    # 模拟"开新会话"：重新实例化，读回文件
    st2 = es.EmotionState(path)
    check("新会话读回 P 一致（不重置）", abs(st2.get_pad()["pleasure"] - p_after) < 1e-9,
          f"st1={p_after}, st2={st2.get_pad()['pleasure']}")
    check("新会话读回 history 有记录", len(st2.dump()["history"]) == 1)

# ---------------------------------------------------------------------------
# 测试 3：时间衰减（baseline 回归）
# ---------------------------------------------------------------------------
section("测试 3：baseline 回归")

# 直接验证 _regress 公式：value(t) = baseline + (value0 - baseline) * e^(-λt)
# 6 小时后，P 从 0.9 向 baseline 0.30 回归（λ=0.12/h）
v = es.EmotionState._regress(0.9, 0.30, 0.12, 6.0)
expected = 0.30 + 0.6 * (2.71828 ** (-0.72))
check("6h 后 P 向 baseline 回归（公式正确）", abs(v - expected) < 1e-6,
      f"实际 {v:.4f}, 期望 {expected:.4f}")

# 验证向 baseline 回归而非向 0：长期后应趋近 baseline 而非 0
v_long = es.EmotionState._regress(0.9, 0.30, 0.12, 100.0)
check("长期后趋近 baseline 0.30（非 0）", abs(v_long - 0.30) < 0.01,
      f"实际 {v_long:.4f}")

# 验证 D 极慢衰减（λ=0.004/h，7 天后几乎不动）
v_d = es.EmotionState._regress(0.9, 0.50, 0.004, 24.0)
check("D 24h 后几乎不动（极慢衰减）", abs(v_d - 0.9) < 0.05,
      f"实际 {v_d:.4f}")

# ---------------------------------------------------------------------------
# 测试 4：冷落规则（定量分段线性）
# ---------------------------------------------------------------------------
section("测试 4：冷落规则")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    st = es.EmotionState(path)

    # 冷落 1 小时：P 应下降，Poss 应上升
    st.state["last_seen"] = es.datetime.fromtimestamp(
        time.time() - 1 * 3600, tz=es.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    p_before = st.get_pad()["pleasure"]
    poss_before = st.get_possessiveness()
    st.apply_time_decay()
    check("冷落 1h 后 P 下降", st.get_pad()["pleasure"] < p_before,
          f"before={p_before}, after={st.get_pad()['pleasure']}")
    check("冷落 1h 后 Poss 上升", st.get_possessiveness() > poss_before,
          f"before={poss_before}, after={st.get_possessiveness()}")

    # 冷落 24 小时：P 应大幅下降
    st2 = es.EmotionState(path)
    st2.state["last_seen"] = es.datetime.fromtimestamp(
        time.time() - 24 * 3600, tz=es.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    p_before2 = st2.get_pad()["pleasure"]
    st2.apply_time_decay()
    check("冷落 24h 后 P 大幅下降", st2.get_pad()["pleasure"] < p_before2 - 0.2,
          f"before={p_before2}, after={st2.get_pad()['pleasure']}")

# ---------------------------------------------------------------------------
# 测试 5：护栏（delta cap + 硬钳位 + 联合熔断）
# ---------------------------------------------------------------------------
section("测试 5：护栏")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    st = es.EmotionState(path)

    base_p = es.BASELINE["pleasure"]
    st.apply_dialogue_delta({"pleasure": 0.9})  # 想推 0.9，但 cap 到 0.3
    check("delta cap：P 增量被截断到 0.3", abs(st.get_pad()["pleasure"] - (base_p + 0.3)) < 1e-9,
          f"实际 {st.get_pad()['pleasure']}")

    # 硬钳位：P 不能超过 1.0
    st2 = es.EmotionState(path)
    st2.set_dimension("pleasure", 0.95)
    st2.apply_dialogue_delta({"pleasure": 0.3})
    check("硬钳位：P 不超过 1.0", st2.get_pad()["pleasure"] <= 1.0,
          f"实际 {st2.get_pad()['pleasure']}")

    # 联合熔断：病态占有（P < -0.6 且 Poss > 0.7）
    st3 = es.EmotionState(path)
    st3.set_dimension("pleasure", -0.7)
    st3.set_dimension("possessiveness", 0.8)
    triggered = st3.apply_dialogue_delta({"pleasure": 0.0})
    # 病态占有触发后 Poss 应被回拉
    check("联合熔断：病态占有触发后 Poss 回拉", st3.get_possessiveness() < 0.8,
          f"实际 {st3.get_possessiveness()}")

# ---------------------------------------------------------------------------
# 测试 6：损坏恢复
# ---------------------------------------------------------------------------
section("测试 6：损坏恢复")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    # 写入损坏的 JSON
    path.write_text("{ 这不是合法 JSON", encoding="utf-8")
    st = es.EmotionState(path)
    check("损坏 JSON 回退冷启动（不崩溃）", abs(st.get_pad()["pleasure"] - es.BASELINE["pleasure"]) < 1e-9)

    # 写入字段缺失的 JSON
    path.write_text('{"pad": {"pleasure": 0.5}}', encoding="utf-8")
    st2 = es.EmotionState(path)
    check("字段缺失自动补全（arousal 补 baseline）", abs(st2.get_pad()["arousal"] - es.BASELINE["arousal"]) < 1e-9)

# ---------------------------------------------------------------------------
# 测试 7：原子写 + 文件锁（并发安全）
# ---------------------------------------------------------------------------
section("测试 7：原子写 + 文件锁")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    st = es.EmotionState(path)
    st.save(event="测试原子写")
    # 验证落盘文件是合法 JSON
    data = json.loads(path.read_text(encoding="utf-8"))
    check("落盘文件是合法 JSON", isinstance(data, dict))
    check("落盘含 history", len(data["history"]) == 1)
    # 验证没有残留 tmp 文件
    check("无残留 tmp 文件", not path.with_suffix(".json.tmp").exists())

# ---------------------------------------------------------------------------
# 测试 8：时钟漂移防护
# ---------------------------------------------------------------------------
section("测试 8：时钟漂移防护")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "emotion_state.json"
    st = es.EmotionState(path)
    # 未来时间（时钟倒拨）
    future = es.datetime.fromtimestamp(time.time() + 3600, tz=es.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    check("时钟倒拨 elapsed=0", es._elapsed_hours(future) == 0.0)
    # 超 30 天（异常）
    ancient = es.datetime.fromtimestamp(time.time() - 40 * 24 * 3600, tz=es.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    check("超 30 天回退假设冷落 1 天", es._elapsed_hours(ancient) == 24.0)

# ---------------------------------------------------------------------------
# 测试 9：配置覆盖（缺省可跑 + 非法回退 + 越界钳位 + 规则表整表替换）
# ---------------------------------------------------------------------------
section("测试 9：配置覆盖")

import importlib

with tempfile.TemporaryDirectory() as tmp:
    cfg = Path(tmp) / "config.json"
    cfg.write_text(json.dumps({
        "baseline": {"pleasure": 0.5, "arousal": "abc", "possessiveness": 99},
        "lambda": {"pleasure": 0.2, "arousal": -1},
        "rules": [[["你好呀"], {"pleasure": 0.1}]],
    }), encoding="utf-8")
    os.environ["EMOTION_CONFIG_PATH"] = str(cfg)
    es2 = importlib.reload(es)
    check("合法覆盖生效（baseline.pleasure=0.5）", abs(es2.BASELINE["pleasure"] - 0.5) < 1e-9,
          f"实际 {es2.BASELINE['pleasure']}")
    check("非法值回退默认（arousal='abc'→0.0）", abs(es2.BASELINE["arousal"] - 0.0) < 1e-9,
          f"实际 {es2.BASELINE['arousal']}")
    check("越界钳位（possessiveness=99→1.0）", abs(es2.BASELINE["possessiveness"] - 1.0) < 1e-9,
          f"实际 {es2.BASELINE['possessiveness']}")
    check("lambda 合法覆盖（pleasure=0.2）", abs(es2.LAMBDA["pleasure"] - 0.2) < 1e-9)
    check("lambda 负数拒绝（arousal=-1→保持默认）", abs(es2.LAMBDA["arousal"] - 1.4) < 1e-9)
    import rules as rl
    rl2 = importlib.reload(rl)
    check("规则表整表替换生效（「你好呀」命中）",
          rl2.match_rules("你好呀") is not None and rl2.match_rules("你好呀")["pleasure"] == 0.1)
    check("替换后旧示例词不再命中（「爱你」已被替换掉）", rl2.match_rules("我爱你") is None)
    del os.environ["EMOTION_CONFIG_PATH"]
    importlib.reload(es2)
    importlib.reload(rl2)
    check("移除配置后回默认（pleasure=0.0）", abs(es2.BASELINE["pleasure"] - 0.0) < 1e-9)
    check("回默认后示例词表恢复（「爱你」重新命中）", rl2.match_rules("我爱你") is not None)

# 缺省可跑：EMOTION_CONFIG_PATH 指向不存在的文件 → 全默认，不炸
with tempfile.TemporaryDirectory() as tmp:
    os.environ["EMOTION_CONFIG_PATH"] = str(Path(tmp) / "nope.json")
    es3 = importlib.reload(es)
    check("缺省可跑（配置文件不存在→全默认）", abs(es3.BASELINE["pleasure"] - 0.0) < 1e-9)
    del os.environ["EMOTION_CONFIG_PATH"]
    importlib.reload(es3)

# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"结果：{PASS} 通过 / {FAIL} 失败")
print(f"{'='*60}")
sys.exit(1 if FAIL else 0)

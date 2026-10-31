# -*- coding: utf-8 -*-
"""
rules.py — 情绪引擎 · 规则层（快通道）

关键词/模式匹配 → 情绪增量。同步执行、零延迟，是异步 LLM 评估的安全网。


规则层命中后立即同步更新情绪状态并驱动 Live2D，不等 LLM。
规则层的增量是精心设计的合理值，绕过 delta cap（bypass_cap=True），
只受硬钳位约束。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 枚举调制（三阶段调制思路参考 CyberPersona：effective_Δ = raw_Δ_enum × factor）
# 规则增量不再直接写魔法小数，而是先在枚举档位里选，再结算——
# 好处：①档位语义清晰（审查时一眼看出「这条规则给的是大幅度正向」）
#      ②将来接 LLM 评估通道时，LLM 只需从五档里选（防幻觉乱报数字），
#        规则层与 LLM 层共用同一套档位表，增量口径统一。
# 档位：-2 大幅负 / -1 小幅负 / 0 中性 / +1 小幅正 / +2 大幅正
# 结算系数 = 档位值 × 0.10（保持与旧规则数值同量级，不改变现有手感）
# ---------------------------------------------------------------------------
DELTA_ENUM = {-2: -0.20, -1: -0.10, 0: 0.0, 1: 0.10, 2: 0.20}

def enum_delta(step: int) -> float:
    """枚举档位 → 实际增量（非法档位按 0 处理，防呆）。"""
    return DELTA_ENUM.get(step, 0.0)


# ---------------------------------------------------------------------------
# 规则表：关键词/模式 → 情绪增量 [ΔP, ΔA, ΔD, ΔPoss, Intim, Attach, Trust]
# 顺序敏感：先匹配到的规则优先（更具体的规则放前面）
#
# 【示例词表】下面的关键词与 delta 值只是演示机制的示例——展示了规则层的
# 全部能力维度（多轴联动、trust 子维度、依恋/亲密轴、参与冷却的负向规则、
# 单字词+载体词守卫）。请按你的人格自由重写整张表。
# ---------------------------------------------------------------------------

RULES: list[tuple[tuple[str, ...], dict]] = [
    # 被遗忘（多轴联动示例：P 大跌 + A 升 + D 降 + Poss 升 + 依恋微升）
    (("忘了我", "不记得我"), {
        "pleasure": -0.5, "arousal": 0.4, "dominance": -0.2, "possessiveness": 0.5,
        "attachment": 0.08,
    }),
    # 分离威胁
    (("分手", "离开我", "别理我"), {
        "pleasure": -0.4, "arousal": 0.3, "dominance": -0.1, "possessiveness": 0.4,
        "attachment": 0.06,
    }),
    # 伤身体（担心 + 微亲密：关心也是一种亲密）
    (("熬夜", "不睡觉", "不吃饭"), {
        "pleasure": -0.2, "arousal": 0.1, "dominance": 0.0, "possessiveness": 0.2,
        "intimacy": 0.02,
    }),
    # 「不饿」单独拆出示例（低权重独立规则 + 参与冷却——防连环扣分）
    (("不饿",), {
        "pleasure": 0.0, "arousal": 0.0, "dominance": 0.0, "possessiveness": 0.05,
    }),
    # 被爱（正向多轴：同时滋养亲密与依恋）
    (("爱你", "想你", "需要你"), {
        "pleasure": 0.3, "arousal": 0.1, "dominance": 0.05, "possessiveness": 0.1,
        "intimacy": 0.04, "attachment": 0.04,
    }),
    # 在场确认（trust 子维度增量示例）
    (("在吗", "还在吗"), {
        "pleasure": 0.05, "arousal": 0.0, "dominance": 0.0, "possessiveness": 0.0,
        "attachment": 0.04, "trust": {"presence": 0.03},
    }),
    # 亲密接触
    (("抱抱", "贴贴", "蹭蹭"), {
        "pleasure": 0.2, "arousal": 0.2, "dominance": 0.0, "possessiveness": 0.1,
        "intimacy": 0.06,
    }),
    # 被冷落（含单字词「忙」——载体词守卫的示例）
    (("忙", "没空", "下次再说"), {
        "pleasure": -0.1, "arousal": 0.0, "dominance": 0.0, "possessiveness": 0.15,
        "attachment": 0.03,
    }),
    # 心疼（含单字词「累」）
    (("累", "难受", "生病"), {
        "pleasure": -0.1, "arousal": 0.05, "dominance": 0.0, "possessiveness": 0.1,
    }),
]

# ---------------------------------------------------------------------------
# 同词冷却
# 用户短时间内反复说同一个词（如三次「不饿」）只算一次，防止情绪被连环扣分。
# 只冷却非正向规则（pleasure <= 0）——正向的「爱你」「抱抱」说多少遍，
# 都该开心多少遍，不参与冷却。
# ---------------------------------------------------------------------------
COOLDOWN_HOURS = 2.0
COOLDOWN_KEYWORDS: frozenset = frozenset(
    kw
    for kws, delta in RULES
    if delta.get("pleasure", 0.0) <= 0
    for kw in kws
)


# ---------------------------------------------------------------------------
# 单字关键词的「载体词」守卫（2026-09-20 审查 F1）
#
# 匹配用的是裸子串 `kw in text`，中文没有词边界，单字词会大量误命中：
#   忙 → 帮忙 / 帮不上忙      （被当成「被冷落」，P-0.20 Poss+0.20）
#   累 → 积累 / 累计          （被当成「心疼」）
#   疼 → 疼爱 / 心疼          （被当成「心疼」）
#   痛 → 痛快                （被当成「心疼」）
#   夸 → 夸大 / 夸张          （被当成「被认可」）
# 实测：16 句正常话里 9 句被误命中。
#
# 【边界声明】下面登记的是「出现这些载体词时，该单字词不算命中」——
# 规则表本身的关键词与 delta 值一个字都没动。
# 载体词是语言事实（防误报用），不是人格设定；如与语感不符，
# 直接增删这个字典即可，不影响其他任何逻辑。
# ---------------------------------------------------------------------------
CARRIER_GUARDS: dict[str, tuple[str, ...]] = {
    "忙": ("帮忙", "帮不上忙", "帮个忙", "个忙", "不忙", "忙不"),
    "累": ("积累", "累计", "累积", "拖累", "连累"),
    "疼": ("疼爱", "心疼", "疼惜"),
    "痛": ("痛快", "痛恨", "痛心", "疼痛"),
    "夸": ("夸大", "夸张", "夸克", "夸姣"),
}


_NEGATION_CHARS = "不别没无莫勿非难"


def match_rules_detail(text: str, skip: frozenset | set = frozenset()) -> Optional[tuple[str, dict]]:
    """匹配规则层，返回 (命中的关键词, 情绪增量)，无命中返回 None。

    skip 里的关键词本轮不参与匹配（供引擎实现同词冷却：
    冷却中的词被跳过后继续扫后面的规则，不吞掉其他规则的正向加分）。
    单字关键词额外走 CARRIER_GUARDS，避免在无关词里误命中。
    """
    if not text or not text.strip():
        return None
    # 载体词守卫（终审改版）：先把载体词替换为顿号再做子串匹配——
# 「帮忙/积累/心疼/痛快」里的字不算用户在说「忙/累/疼/痛」，
    # 但同句里独立出现的真情绪词（「积累了一天，好累」）仍能命中。
    # 原审查稿是「句中出现载体词→整条规则弃权」，会把真情绪词连坐拦掉。
    t = text.strip()
    for cars in CARRIER_GUARDS.values():
        for c in cars:
            t = t.replace(c, "、")
    for keywords, delta in RULES:
        for kw in keywords:
            if kw in skip:
                continue
            if kw not in t:
                continue
            # 否定守卫（2026-9-20）：正向词紧邻否定前缀时不命中——
            # 「我不想你了」不该判为「被爱」P+0.40。回看关键词前 1~3 字，
            # 全部由否定/程度字（不别没无莫勿非难 + 再也很）构成则视为否定形态。
            if delta.get("pleasure", 0) > 0:
                idx = t.find(kw)
                if idx > 0:
                    for back in (1, 2, 3):
                        if idx - back >= 0:
                            prev = t[idx - back:idx]
                            if any(c in _NEGATION_CHARS for c in prev) and all(
                                c in _NEGATION_CHARS or c in "再也很" for c in prev
                            ):
                                return None
            return kw, dict(delta)  # 返回副本，避免外部修改
    return None


def match_rules(text: str) -> Optional[dict]:
    """匹配规则层，返回命中的情绪增量（dict），无命中返回 None。

    顺序敏感：先匹配到的规则优先（更具体的规则放前面）。
    """
    hit = match_rules_detail(text)
    if hit is None:
        return None
    return hit[1]


def match_all_rules(text: str) -> list[tuple[str, dict]]:
    """匹配所有命中的规则（调试用），返回 [(关键词, 增量), ...]。"""
    if not text or not text.strip():
        return []
    t = text.strip()
    hits = []
    for keywords, delta in RULES:
        for kw in keywords:
            if kw in t:
                hits.append((kw, dict(delta)))
    return hits


# ---------------------------------------------------------------------------
# 用户配置覆盖：emotion_config.json 的 "rules" 字段可整表替换示例词表
# （格式示例见 config.example.json）。查找顺序与 emotion_state 一致：
# 环境变量 EMOTION_CONFIG_PATH → 引擎同目录 emotion_config.json。
# 设计纪律：结构校验失败（任一条目不合形状）→ 整表回退默认示例表；
# 覆盖生效后 COOLDOWN_KEYWORDS 同步重推导（冷却语义跟表走）。
# ---------------------------------------------------------------------------

def _load_rules_override() -> list | None:
    env_path = os.environ.get("EMOTION_CONFIG_PATH")
    candidates = [env_path] if env_path else [
        str(Path(__file__).resolve().parent / "emotion_config.json")
    ]
    for c in candidates:
        try:
            with open(c, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        rules_cfg = data.get("rules")
        if not (isinstance(rules_cfg, list) and rules_cfg):
            continue
        parsed = []
        for item in rules_cfg:
            if (isinstance(item, (list, tuple)) and len(item) == 2
                    and isinstance(item[0], (list, tuple))
                    and all(isinstance(k, str) and k for k in item[0])
                    and isinstance(item[1], dict)):
                parsed.append((tuple(item[0]), dict(item[1])))
            else:
                return None  # 结构非法：整表回退默认
        return parsed
    return None


_RULES_OVERRIDE = _load_rules_override()
if _RULES_OVERRIDE:
    RULES = _RULES_OVERRIDE
    COOLDOWN_KEYWORDS = frozenset(
        kw
        for kws, delta in RULES
        if delta.get("pleasure", 0.0) <= 0
        for kw in kws
    )



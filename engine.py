# -*- coding: utf-8 -*-
"""
engine.py — 情绪引擎 · 编排层

把 emotion_state（状态持久化）+ rules（规则层）串起来，
给宿主 agent（LLM 伴侣/助手）和 Live2D 桌宠提供一个统一入口。

核心流程（每轮对话）：
  1. 读情绪状态（含损坏恢复）
  2. 时间衰减 + 冷落规则（按 last_seen 算冷落多久）
  3. 规则层匹配（快通道，同步）
  4. 叠加对话增量
  5. 落盘
  6. 生成情绪上下文（注入 system prompt / 驱动 Live2D）
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

try:
    from . import emotion_state as es
    from . import rules
except ImportError:
    import emotion_state as es
    import rules


class EmotionEngine:
    """情绪引擎统一入口。"""

    def __init__(self, state_path: Path | str = es.DEFAULT_STATE_PATH):
        self.state = es.EmotionState(state_path)
        # 同词冷却表：关键词 → 上次命中的时间戳（2026-8-30 修复）
        # 挂在实例内存而非 emotion_state.json——不碰状态文件的 schema，
        # 引擎随宿主进程常驻，跨轮有效；重启丢失冷却属可接受降级。
        self._kw_last_hit: dict[str, float] = {}
        # 自源事件单日预算（当日 Σ|ΔP|，内存计数；重启清零=预算重置，
        # 最坏情况一天多喂一倍小额——判可接受降级）
        self._self_event_day: tuple[str, float] = ("", 0.0)

    # ---- 每轮对话主流程 ----

    def on_user_message(self, user_text: str) -> dict:
        """每轮用户发消息时调用。

        返回本轮情绪摘要（供注入 system prompt / 驱动 Live2D）。
        整个「读→算→写」序列在文件锁事务内完成，多进程并发不丢更新。
        """
        # 事务：拿锁 → reload 最新状态 → 计算 → 落盘 → 释放
        with self.state.transaction():
            # 时间衰减 + 冷落规则（reload 已由 transaction 完成）
            time_summary = self.state.apply_time_decay()

            # 规则层匹配（快通道）+ 同词冷却（2026-8-30 修复）
            # 2小时内同一非正向关键词只命中一次；冷却中的词被跳过后
            # 继续扫后面的规则——「不饿+蹭蹭」里的蹭蹭加分不会被吞。
            now = time.time()
            cooldown_s = rules.COOLDOWN_HOURS * 3600.0
            skip = {
                kw for kw, ts in self._kw_last_hit.items()
                if now - ts < cooldown_s
            }
            rule_hit = rules.match_rules_detail(user_text, skip)
            rule_delta = rule_hit[1] if rule_hit else None
            if rule_hit and rule_hit[0] in rules.COOLDOWN_KEYWORDS:
                self._kw_last_hit[rule_hit[0]] = now
                # 防表无限增长：过期即清
                if len(self._kw_last_hit) > 128:
                    self._kw_last_hit = {
                        kw: ts for kw, ts in self._kw_last_hit.items()
                        if now - ts < cooldown_s
                    }

            # 叠加对话增量（规则层 bypass cap）
            if rule_delta:
                self.state.apply_dialogue_delta(rule_delta, bypass_cap=True)
                event = f"规则层命中[{rule_hit[0]}]: {user_text[:30]}"
            else:
                event = f"对话: {user_text[:30]}"


            # 落盘（事务内，复用锁）
            self.state.save(event=event,
                            delta=rule_delta or {})

        # ⑥ 返回摘要
        return {
            "pad": self.state.get_pad(),
            "possessiveness": self.state.get_possessiveness(),
            "trust": self.state.get_trust(),
            "rule_hit": rule_delta is not None,
            "elapsed_hours": time_summary["elapsed_hours"],
        }


    @classmethod
    def _text_mood(cls, pad: dict, poss: float) -> str:
        """_describe_mood 的**分档名**（供一致性判定用，不改文本行为）。"""
        p, a, d = pad["pleasure"], pad["arousal"], pad["dominance"]
        if p < -0.6 and poss > 0.7:
            return "病态占有"
        if p < -0.5 and d < -0.2 and poss > 0.6:
            return "恐惧"
        if p < -0.4 and a > 0.4 and d > 0.3:
            return "生气"
        if p < -0.3 and a < 0.2:
            return "委屈"
        if p > 0.4:
            return "开心"
        if poss > 0.7 and a > 0.4:
            return "得意"
        if abs(a) < 0.3:
            return "平静"
        return "温柔"



    # ---- 自源事件喂料口 ----

    # 设计红线：自源增量只许有来由的事件因果——引擎只消费不推断，
    # 本节不存在任何事件生成代码（无定时器、无自动「无聊」）。
    # 走心情轨道（apply_dialogue_delta），不 bypass cap——那是规则层
    # 精心设计的特权身份，自源事件没有。D/Poss 全零：关系轴不被天气动。
    SELF_EVENT_MAX_ABS = 0.15          # 单维小额上限（< emotion_state.MAX_DELTA 0.3）
    SELF_EVENT_COOLDOWN_HOURS = 2.0    # 同品种冷却（挂 _kw_last_hit，键位 "self:" 前缀隔离）
    SELF_EVENT_DAILY_BUDGET = 0.60     # 当日 Σ|ΔP| 预算，正负都计，耗尽当日拒收

    _SELF_EVENT_KINDS: dict = {
        "weather_bad":        {"pleasure": -0.08, "arousal": -0.03},
        "boring_read":        {"pleasure": -0.05, "arousal": -0.05},
        "idle_bored":         {"pleasure": -0.04, "arousal": -0.06},
        "work_in_progress":   {"pleasure": +0.04, "arousal": +0.05},
        "observed_annoyance": {"pleasure": -0.06, "arousal": +0.03},
        "video_find":         {"pleasure": +0.10, "arousal": +0.05},
        "forward_answered":   {"pleasure": +0.08, "arousal": +0.02},
    }

    def apply_self_event(self, kind: str, note: str = "",
                         delta_override: dict | None = None) -> dict:
        """自源事件喂料口：事件 → 小额 PAD 增量，走心情轨道。

        kind: 品种枚举（_SELF_EVENT_KINDS）
        note: 事件备注（进 history，审计用）
        delta_override: 显式增量覆盖（如「活儿卡壳」vs「活儿顺利」）——
                        覆盖方向不覆盖额度：仍受 SELF_EVENT_MAX_ABS 钳制。
        返回 {"applied": 实际增量, "skipped": None|"cooldown"|"budget"}
        """
        if kind not in self._SELF_EVENT_KINDS:
            raise ValueError(f"未知自源事件品种 {kind!r}")
        now = time.time()
        with self.state.transaction():
            # 品种冷却：同品种 2h 内只计一次；不同品种互不干扰
            ckey = "self:" + kind
            if now - self._kw_last_hit.get(ckey, 0.0) < self.SELF_EVENT_COOLDOWN_HOURS * 3600.0:
                return {"applied": {}, "skipped": "cooldown"}
            # 单日预算：当日 Σ|ΔP| 正负都计，耗尽拒收；跨日重置
            day = time.strftime("%Y-%m-%d")
            if self._self_event_day[0] != day:
                self._self_event_day = (day, 0.0)
            if self._self_event_day[1] >= self.SELF_EVENT_DAILY_BUDGET:
                return {"applied": {}, "skipped": "budget"}
            base = self._SELF_EVENT_KINDS[kind]
            delta = {
                k: float((delta_override or {}).get(k, base.get(k, 0.0)))
                for k in ("pleasure", "arousal")
            }
            # override 只许改方向不许改额度
            delta = {k: max(-self.SELF_EVENT_MAX_ABS,
                            min(self.SELF_EVENT_MAX_ABS, v))
                     for k, v in delta.items()}
            capped = self.state.apply_dialogue_delta(delta)   # 心情轨道，不 bypass cap
            self.state.save(event=f"自源[{kind}]" + (f": {note}" if note else ""),
                            delta=capped)
            self._kw_last_hit[ckey] = now
            self._self_event_day = (
                day, self._self_event_day[1] + abs(capped.get("pleasure", 0.0))
            )
            return {"applied": capped, "skipped": None}

    def emotion_context(self) -> str:
        """生成情绪上下文，注入 system prompt。

        让 LLM 在生成回复时就感知当前情绪，避免「表情愤怒但回复温柔」的割裂。
        v4：五维关系全量注入 + 时段感知（世界观同步思路参考 CyberPersona，
        轻量版：只做本地时段，不引外部天气依赖）。
        """
        pad = self.state.get_pad()
        poss = self.state.get_possessiveness()
        trust = self.state.get_trust()
        intimacy = self.state.get_intimacy()
        attachment = self.state.get_attachment()

        # 情绪描述（离散情绪，nearest-centroid 简化版）
        mood = self._describe_mood(pad, poss)

        # 时段感知（与 period_text() 同源，避免两处各写一份分档）
        period = self.period_text()


        pad_line = ""
        if os.environ.get("EMOTION_PAD_INJECT") == "1":  # 审计开关：默认藏数值
            pad_line = (
                f"[当前情绪] 愉悦度 {pad['pleasure']:+.2f} / 唤醒度 {pad['arousal']:+.2f} "
                f"/ 支配度 {pad['dominance']:+.2f} / 占有欲 {poss:.2f} / 信任 {trust['presence']:.2f} "
                f"/ 亲密感 {intimacy:.2f} / 依恋度 {attachment:.2f}\n"
            )
        return (
            f"{pad_line}"
            f"现在是{period}。\n"
            f"我感觉现在心情：{mood}"
        )

    @staticmethod
    def _describe_mood(pad: dict, poss: float) -> str:
        """把 PAD+Poss 坐标翻译成一句心情描述。

        【示例文案】九档心情句只是给使用者的起点示例——
        这一层是引擎里最「人格」的部分，请为你的 AI 亲自重写。
        """
        p, a, d = pad["pleasure"], pad["arousal"], pad["dominance"]

        # 病态占有（联合极端态）
        if p < -0.6 and poss > 0.7:
            return "不安又执着，占有欲爆表，只想把对方留在身边。"
        # 恐惧（低支配 + 高占有欲）
        if p < -0.5 and d < -0.2 and poss > 0.6:
            return "害怕失去对方，想牢牢抓住这段关系。"
        # 生气
        if p < -0.4 and a > 0.4 and d > 0.3:
            return "生气，想找对方把话说清楚。"
        # 委屈
        if p < -0.3 and a < 0.2:
            return "委屈，想被哄一哄。"
        # 开心（P 高就开心，不强制要求 A 高）
        if p > 0.4:
            if a > 0.2:
                return "开心，想和对方分享这份好心情。"
            return "开心而满足。"
        # 得意/占有欲上头
        if poss > 0.7 and a > 0.4:
            return "得意，占有欲上头，想把对方整个人占为己有。"
        # 平静
        if abs(a) < 0.3:
            return "平静而满足。"
        # 默认
        return "心情平稳，温柔地注视着对方。"

    # ---- 供 Live2D 消费 ----

    def live2d_params(self) -> dict:
        """生成 Live2D 直驱参数（阶段 3 用）。

        返回「语义参数」（不绑定具体模型），由前端做模型映射：
          cheek      脸红（兴奋或占有欲任一强烈就红）
          mouth_up   嘴角上扬/笑（愉悦度）
          mouth_down 嘴角向下/难过（愉悦度负）
          mouth_angry 嘟嘴（委屈 + 占有欲）
          eye_smile  笑眼（愉悦度）
          tear       眼泪（委屈/难过，P 低到阈值才流）

        这样换模型只改前端映射表，引擎不动。
        """
        pad = self.state.get_pad()
        poss = self.state.get_possessiveness()
        p, a, d = pad["pleasure"], pad["arousal"], pad["dominance"]

        # 参数混合公式（设计文档 §8.1）
        cheek = max(0.4 * max(a, 0.0), 0.6 * poss)  # 脸红：兴奋或占有欲任一强烈就红
        mouth_up = max(p, 0.0)                        # 嘴角上扬：愉悦度
        mouth_down = max(-p, 0.0)                      # 嘴角向下：难过/委屈（愉悦度负）
        mouth_angry = max(-p, 0.0) * 0.5 + poss * 0.5  # 嘟嘴：委屈 + 占有欲
        eye_smile = max(p, 0.0)                       # 笑眼：愉悦度
        # 眼泪：P 低于 -0.3 才开始流，-1.0 时满
        tear = max((-p - 0.3) / 0.7, 0.0)
        # 星星眼：开心到极致（愉悦度 > 0.5 开始，1.0 满）
        star_eye = max(0.0, (p - 0.5) / 0.5)


        return {
            "cheek": round(cheek, 3),
            "mouth_up": round(mouth_up, 3),
            "mouth_down": round(mouth_down, 3),
            "mouth_angry": round(mouth_angry, 3),
            "eye_smile": round(eye_smile, 3),
            "tear": round(tear, 3),
            "star_eye": round(star_eye, 3),
        }

    # ---- 供渲染层消费的纯展示文本（与注入 LLM 的段落分开）----

    def mood_label(self) -> str:
        """当前心情的分档名（供前端/日志用，不含人格文案）。"""
        return self._text_mood(self.state.get_pad(), self.state.get_possessiveness())

    def mood_text(self) -> str:
        """纯展示用的一句话心情。

        emotion_context() 是给模型的指令段落，可能含本机路径等私有信息，
        不该出现在渲染层接口里。
        """
        return self._describe_mood(self.state.get_pad(), self.state.get_possessiveness())

    @staticmethod
    def period_text() -> str:
        """时段感知的纯展示文本（分档与 emotion_context 原文一致）。"""
        hour = time.localtime().tm_hour
        if 5 <= hour < 8:
            return "清晨"
        if 8 <= hour < 12:
            return "上午"
        if 12 <= hour < 14:
            return "午间"
        if 14 <= hour < 18:
            return "下午"
        if 18 <= hour < 23:
            return "夜晚"
        return "深夜"

    # ---- 供 CLI / 调试 ----

    def dump(self) -> dict:
        return self.state.dump()

    def reset(self) -> None:
        self.state.reset()

    def set_dimension(self, dim: str, value: float) -> None:
        self.state.set_dimension(dim, value)

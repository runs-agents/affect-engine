# -*- coding: utf-8 -*-
"""
emotion_state.py — 情绪引擎 · 状态地基

情绪状态（PAD + 占有欲 + 亲密/依恋 + 信任）的持久化、时间衰减、冷落规则、护栏。


核心能力：
  1. 情绪状态落盘（原子写 + 文件锁 + 损坏恢复 + schema 校验）
  2. baseline 回归（情绪向底色回归，不是向 0）
  3. 冷落规则（定量分段线性公式，由信任度调节敏感度）
  4. 护栏（硬钳位 + delta cap + 软熔断 + 联合熔断 + 方向检测）
  5. CLI（--dump / --reset / --set）

用法：
  python emotion_state.py --dump          # 查看当前情绪
  python emotion_state.py --reset         # 重置到 baseline
  python emotion_state.py --set P=0.5     # 手动修正某维
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 落盘位置（可用环境变量 EMOTION_STATE_PATH 覆盖）
DEFAULT_STATE_PATH = Path(os.environ.get("EMOTION_STATE_PATH", "emotion_state.json"))

# 版本
# schema 版本（对外版独立计数，从 1 起步）
# v1 = PAD 三轴 + 占有欲 + 亲密/依恋 + 信任双子
STATE_VERSION = 1

# 维度合法区间
PAD_RANGE = (-1.0, 1.0)      # P/A/D
POSS_RANGE = (0.0, 1.0)      # Poss
TRUST_RANGE = (0.0, 1.0)     # Trust

# baseline（情绪底色锚点——【示例值】请按你的人格定制）
BASELINE = {
    "pleasure": 0.0,
    "arousal": 0.0,
    "dominance": 0.0,
    "possessiveness": 0.30,
    "intimacy": 0.30,
    "attachment": 0.30,
}

# pad 三轴（不含 possessiveness）
PAD_KEYS = ("pleasure", "arousal", "dominance")

# 冷启动信任度
COLD_START_TRUST = {"presence": 0.50, "safety": 0.50}

# 衰减系数 λ（单位：/小时）——示例值：每种情绪的「呼吸节奏」由它决定
LAMBDA = {
    "pleasure": 0.12,        # half-life ~6h
    "arousal": 1.4,          # half-life ~30min
    "dominance": 0.004,      # half-life ~7天
    "possessiveness": 0.03,  # half-life ~1天
    "trust": 0.0,            # 几乎不衰减
    "intimacy": 0.005,       # half-life ~5.8天：关系轴示例值——请按你们的关系定
    "attachment": 0.005,     # half-life ~5.8天：关系轴示例值——请按你们的关系定
}

# 单轮增量上限（delta cap）
MAX_DELTA = 0.3

# 软熔断阈值（原定义未接线，现已接入 apply_time_decay）
SOFT_BREAKER_THRESHOLD = 0.85

# 连续方向检测轮数
DIRECTION_GUARD_N = 5

# history 上限
MAX_HISTORY = 50

# 时钟漂移防护：elapsed 超过此值视为异常
MAX_ELAPSED_HOURS = 30 * 24  # 30 天
ASSUME_COLD_HOURS = 24.0     # 异常时假设冷落 1 天


# ---------------------------------------------------------------------------
# 冷启动状态
# ---------------------------------------------------------------------------

def cold_start_state() -> dict:
    """返回冷启动状态（第一次运行 / 文件损坏时使用）。"""
    return {
        "version": STATE_VERSION,
        "pad": {k: BASELINE[k] for k in PAD_KEYS},  # 只含 P/A/D 三轴
        "possessiveness": BASELINE["possessiveness"],
        # 关系轴（顶层轴，与 possessiveness 同级）
        "intimacy": BASELINE["intimacy"],
        "attachment": BASELINE["attachment"],
        "trust": dict(COLD_START_TRUST),
        "baseline": dict(BASELINE),
        "last_seen": _now_utc_iso(),
        "history": [],
    }


# ---------------------------------------------------------------------------
# 时间工具
# ---------------------------------------------------------------------------

def _now_utc_iso() -> str:
    """当前 UTC 时间，ISO 8601 带 Z 后缀。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(iso: str) -> float | None:
    """解析 UTC ISO 时间戳为 epoch 秒，失败返回 None。"""
    try:
        s = iso.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


def _elapsed_hours(last_seen: str) -> float:
    """计算距离上次互动的小时数，含时钟漂移防护。

    返回：
      - 正常：elapsed 小时数（>= 0）
      - 时钟倒拨（elapsed < 0）：视为 0（刚见过）
      - 异常（elapsed > 30 天）：视为 24 小时（假设冷落 1 天）
    """
    ts = _parse_utc(last_seen)
    if ts is None:
        return ASSUME_COLD_HOURS
    elapsed = (time.time() - ts) / 3600.0
    if elapsed < 0:
        return 0.0
    if elapsed > MAX_ELAPSED_HOURS:
        return ASSUME_COLD_HOURS
    return elapsed


# ---------------------------------------------------------------------------
# 文件锁（Windows msvcrt.locking）
# ---------------------------------------------------------------------------

def _acquire_lock(lock_path: Path, retries: int = 5, delay: float = 0.2):
    """获取文件锁，返回 (文件对象, 是否真的持有锁)。

    第二项是判断依据：原实现无论成败都返回文件对象，调用方无法区分
    「拿到锁」与「降级无锁」，并发时静默丢更新（审查 E1/E2 已复现）。
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+")
    try:
        import msvcrt
        f.seek(0)
        for attempt in range(retries):
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                return f, True
            except OSError:
                if attempt < retries - 1:
                    time.sleep(delay)
        # 重试耗尽：告警但继续（原子写仍保证文件不损坏，只是失去互斥）
        print("⚠ 文件锁获取失败（重试耗尽），以无锁模式继续：", lock_path, file=sys.stderr)
    except (ImportError, OSError) as e:
        print(f"⚠ 文件锁不可用（{e}），以无锁模式继续：", lock_path, file=sys.stderr)
    return f, False


def _release_lock(f):
    """释放文件锁并关闭文件。"""
    try:
        import msvcrt
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    except (ImportError, OSError):
        pass
    try:
        f.close()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 原子写 + 损坏恢复
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, data: dict, retries: int = 5) -> None:
    """原子写：先写【本次调用独占】的临时文件，再 os.replace 原子替换。

    原实现 tmp 名写死为 path.json.tmp，所有写入者共用同一路径：
    A 写完 → B 覆写同一 tmp → A replace（落进去的是 B 的内容）→
    B replace 时 tmp 已不存在。并发实测抛 PermissionError(WinError 5)。
    改为每次调用唯一名，并对 replace 的瞬时占用做有界重试。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        last = None
        for attempt in range(retries):
            try:
                os.replace(tmp, path)
                return
            except OSError as e:            # Windows：目标被瞬时占用
                last = e
                if attempt < retries - 1:
                    time.sleep(0.05 * (attempt + 1))
        raise last
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _load_state(path: Path) -> dict:
    """读取情绪状态，含损坏恢复 + schema 校验。

    任何解析失败 / 字段缺失 / 类型错误 → 回退冷启动状态。
    """
    if not path.is_file():
        return cold_start_state()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            return cold_start_state()
        return _validate_state(data)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return cold_start_state()


def _validate_state(data: dict) -> dict:
    """schema 校验：字段类型 + 范围钳位，缺失字段用 baseline 补。"""
    # 确保顶层结构
    if "pad" not in data or not isinstance(data["pad"], dict):
        data["pad"] = {k: BASELINE[k] for k in PAD_KEYS}
    if "baseline" not in data or not isinstance(data["baseline"], dict):
        data["baseline"] = dict(BASELINE)
    if "trust" not in data or not isinstance(data["trust"], dict):
        data["trust"] = dict(COLD_START_TRUST)

    # 钳位 pad 三轴（并清理多余键，如旧版本误塞的 possessiveness）
    data["pad"] = {k: _clamp_float(data["pad"].get(k), PAD_RANGE, BASELINE[k]) for k in PAD_KEYS}

    # 钳位 possessiveness
    data["possessiveness"] = _clamp_float(
        data.get("possessiveness"), POSS_RANGE, BASELINE["possessiveness"]
    )

    # v4：钳位 intimacy / attachment（v3 旧文件缺键 → baseline 补，向后兼容）
    for rel_key in ("intimacy", "attachment"):
        data[rel_key] = _clamp_float(
            data.get(rel_key), POSS_RANGE, BASELINE[rel_key]
        )

    # 钳位 baseline
    for key in ("pleasure", "arousal", "dominance"):
        data["baseline"][key] = _clamp_float(
            data["baseline"].get(key), PAD_RANGE, BASELINE[key]
        )
    data["baseline"]["possessiveness"] = _clamp_float(
        data["baseline"].get("possessiveness"), POSS_RANGE, BASELINE["possessiveness"]
    )
    # v4：baseline 也补 intimacy/attachment
    for rel_key in ("intimacy", "attachment"):
        data["baseline"][rel_key] = _clamp_float(
            data["baseline"].get(rel_key), POSS_RANGE, BASELINE[rel_key]
        )

    # 钳位 trust
    for key in ("presence", "safety"):
        data["trust"][key] = _clamp_float(
            data["trust"].get(key), TRUST_RANGE, COLD_START_TRUST[key]
        )


    # last_seen 缺失则补当前时间
    if not isinstance(data.get("last_seen"), str):
        data["last_seen"] = _now_utc_iso()

    # history 裁剪
    if not isinstance(data.get("history"), list):
        data["history"] = []
    data["history"] = data["history"][-MAX_HISTORY:]

    data["version"] = STATE_VERSION
    return data


def _clamp_float(val, rng: tuple, default: float) -> float:
    """把值钳位到 [rng[0], rng[1]]，非法值回退 default。"""
    try:
        v = float(val)
    except (TypeError, ValueError):
        return default
    lo, hi = rng
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# 用户配置覆盖（emotion_config.json —— 格式示例见 config.example.json）
# 查找顺序：环境变量 EMOTION_CONFIG_PATH → 引擎同目录 emotion_config.json。
# 设计纪律：缺省可跑（无配置文件=全默认）；非法值回退默认（类型校验+钳位）。
# ---------------------------------------------------------------------------

def _load_user_config() -> dict:
    env_path = os.environ.get("EMOTION_CONFIG_PATH")
    candidates = [env_path] if env_path else [
        str(Path(__file__).resolve().parent / "emotion_config.json")
    ]
    for c in candidates:
        try:
            with open(c, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            continue
    return {}


_CFG = _load_user_config()


def _apply_config_overrides() -> None:
    """把用户配置合并进 BASELINE / LAMBDA（非法回退默认，越界钳位）。"""
    base_cfg = _CFG.get("baseline")
    if isinstance(base_cfg, dict):
        for key in ("pleasure", "arousal", "dominance"):
            if key in base_cfg:
                BASELINE[key] = _clamp_float(base_cfg[key], PAD_RANGE, BASELINE[key])
        for key in ("possessiveness", "intimacy", "attachment"):
            if key in base_cfg:
                BASELINE[key] = _clamp_float(base_cfg[key], POSS_RANGE, BASELINE[key])
    lam_cfg = _CFG.get("lambda")
    if isinstance(lam_cfg, dict):
        for key, default in LAMBDA.items():
            if key in lam_cfg:
                try:
                    v = float(lam_cfg[key])
                except (TypeError, ValueError):
                    continue  # 非法值：保持默认
                if v >= 0:
                    LAMBDA[key] = v


_apply_config_overrides()


# ---------------------------------------------------------------------------
# 情绪状态类
# ---------------------------------------------------------------------------

class EmotionState:
    """情绪状态管理器。"""

    def __init__(self, path: Path | str = DEFAULT_STATE_PATH):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(".json.lock")
        # 事务状态必须是【每线程】的：server.py 是 ThreadingHTTPServer +
        # 模块级单例，多个请求线程共用同一个实例。原实现用实例属性
        # _in_transaction，任一线程置 True 后其他线程都误判为「嵌套调用」
        # 而直接放行，互斥形同虚设（审查 E2 复现：A 未落盘的改动被 B 覆盖）。
        self._tx_local = threading.local()
        self._tx_lock = threading.RLock()
        self.state = _load_state(self.path)

    @property
    def _in_transaction(self) -> bool:
        """当前【线程】是否处于事务中（跨线程互不可见）。"""
        return getattr(self._tx_local, "depth", 0) > 0

    # ---- 事务（读-改-写原子化）----

    @contextmanager
    def transaction(self):
        """把「读→算→写」整个序列变成原子事务。

        用法：
            with st.transaction():
                st.apply_time_decay()
                st.apply_dialogue_delta(delta)
                st.save(event=...)   # 事务内 save 不再重复拿锁

        事务期间持有文件锁，其他进程的 transaction() 会阻塞等待，
        防止 A 进程的计算结果被 B 进程的旧状态覆盖。
        """
        if self._in_transaction:
            # 同线程嵌套调用（如同一事务里两次 save）：复用外层锁
            self._tx_local.depth += 1
            try:
                yield self
            finally:
                self._tx_local.depth -= 1
            return
        # 先线程内串行化，再取进程间文件锁。原实现只有后者：同进程的两个
        # 线程对同一文件加锁必然失败（Windows 锁按句柄），重试耗尽后静默
        # 降级为无锁 —— 两道防线同时失守。
        with self._tx_lock:
            lock, locked = _acquire_lock(self.lock_path)
            self._tx_local.depth = 1
            try:
                if not locked:
                    print(f"⚠ 未取得文件锁，本次事务仅线程内互斥：{self.lock_path}",
                          file=sys.stderr)
                # 拿到锁后立刻读最新状态，保证基于最新数据计算
                self.state = _load_state(self.path)
                yield self
            finally:
                self._tx_local.depth = 0
                _release_lock(lock)

    def reload(self) -> None:
        """重新从磁盘读最新状态（多进程共享时，避免用内存里的旧状态覆盖）。

        事务内调用会被跳过：transaction() 已经在拿锁后 reload 过了，
        事务中途 reload 会读到其他进程的中间态。
        """
        if self._in_transaction:
            return
        self.state = _load_state(self.path)

    # ---- 读取 ----

    def get_pad(self) -> dict:
        return dict(self.state["pad"])

    def get_possessiveness(self) -> float:
        return self.state["possessiveness"]

    def get_intimacy(self) -> float:
        """亲密感（0-1）。"""
        return self.state["intimacy"]

    def get_attachment(self) -> float:
        """依恋度（0-1）。"""
        return self.state["attachment"]

    def get_trust(self) -> dict:
        return dict(self.state["trust"])

    def get_baseline(self) -> dict:
        return dict(self.state["baseline"])

    def get_last_seen(self) -> str:
        return self.state["last_seen"]

    def dump(self) -> dict:
        """返回完整状态（深拷贝）。"""
        return json.loads(json.dumps(self.state, ensure_ascii=False))

    # ---- 时间衰减 + 冷落规则 ----

    def apply_time_decay(self) -> dict:
        """按 elapsed_time 做 baseline 回归 + 冷落规则。

        返回本轮时间流逝产生的情绪变化摘要（供 history 记录）。
        """
        elapsed = _elapsed_hours(self.state["last_seen"])

        # ① 冷落增量（定量分段线性公式）
        cold = self._cold_shoulder_delta(elapsed)

        # ② 冷落增量修改 baseline（临时 baseline，避免双罚）
        temp_baseline = dict(self.state["baseline"])
        temp_baseline["pleasure"] = _clamp_float(
            temp_baseline["pleasure"] + cold["pleasure"], PAD_RANGE, temp_baseline["pleasure"]
        )
        temp_baseline["arousal"] = _clamp_float(
            temp_baseline["arousal"] + cold["arousal"], PAD_RANGE, temp_baseline["arousal"]
        )
        temp_baseline["possessiveness"] = _clamp_float(
            temp_baseline["possessiveness"] + cold["possessiveness"],
            POSS_RANGE,
            temp_baseline["possessiveness"],
        )
        # D 不受冷落影响（底色）

        # ③ 向临时 baseline 回归（超阈值的轴按软熔断加倍衰减）
        for key in ("pleasure", "arousal", "dominance"):
            self.state["pad"][key] = self._regress(
                self.state["pad"][key], temp_baseline[key],
                LAMBDA[key] * self._soft_breaker_factor(self.state["pad"][key]),
                elapsed,
            )
        self.state["possessiveness"] = self._regress(
            self.state["possessiveness"],
            temp_baseline["possessiveness"],
            LAMBDA["possessiveness"],
            elapsed,
        )
        # v4：亲密感/依恋度极慢回归（几乎不淡，但长期冷落仍会缓缓下降）
        for rel_key in ("intimacy", "attachment"):
            self.state[rel_key] = self._regress(
                self.state[rel_key],
                temp_baseline[rel_key],
                LAMBDA[rel_key],
                elapsed,
            )

        # ④ 更新 last_seen
        self.state["last_seen"] = _now_utc_iso()

        return {"elapsed_hours": elapsed, "cold_delta": cold}

    def _cold_shoulder_delta(self, elapsed: float) -> dict:
        """冷落增量（定量分段线性公式），由信任度调节敏感度。

        返回 {"pleasure": ΔP, "arousal": ΔA, "possessiveness": ΔPoss}
        """
        # 基础冷落增量（分段线性）
        p = self._piecewise_p(elapsed)
        a = self._piecewise_a(elapsed)
        poss = self._piecewise_poss(elapsed)

        # 信任度调节敏感度：实际 = 基础 × (1.5 - trust.presence)
        sensitivity = 1.5 - self.state["trust"]["presence"]
        sensitivity = max(0.5, min(1.5, sensitivity))  # 钳位到 [0.5, 1.5]

        return {
            "pleasure": p * sensitivity,
            "arousal": a * sensitivity,
            "possessiveness": poss * sensitivity,
        }

    @staticmethod
    def _piecewise_p(elapsed: float) -> float:
        """P 冷落增量（分段线性，负值 = 愉悦度下降）。"""
        if elapsed < 0.5:
            return 0.0
        if elapsed < 3:
            return -0.10 * (elapsed - 0.5) / 2.5
        if elapsed < 12:
            return -0.10 - 0.20 * (elapsed - 3) / 9
        return -0.30 - 0.10 * min((elapsed - 12) / 12, 1)

    @staticmethod
    def _piecewise_a(elapsed: float) -> float:
        """A 冷落增量（分段线性，正值 = 唤醒度上升）。"""
        if elapsed < 3:
            return 0.0
        if elapsed < 12:
            return 0.10 * (elapsed - 3) / 9
        return 0.10

    @staticmethod
    def _piecewise_poss(elapsed: float) -> float:
        """Poss 冷落增量（分段线性，正值 = 占有欲上升）。连续无阶跃。"""
        if elapsed < 0:
            return 0.0
        if elapsed < 0.5:
            # 第一段线性爬坡：从 0 连续升到 0.05（消除原 0.02→0.05 的边界阶跃）
            return 0.10 * elapsed
        if elapsed < 3:
            return 0.05 + 0.10 * (elapsed - 0.5) / 2.5
        if elapsed < 12:
            return 0.15 + 0.20 * (elapsed - 3) / 9
        return 0.35 + 0.15 * min((elapsed - 12) / 12, 1)

    @staticmethod
    def _soft_breaker_factor(value: float) -> float:
        """软熔断：|value| 超阈值时衰减系数加倍。

        常量与注释一直在（SOFT_BREAKER_THRESHOLD / docstring 第 4 项），
        但此前没有任何代码读取它 —— 声明了的护栏实际不存在。
        """
        return 2.0 if abs(value) > SOFT_BREAKER_THRESHOLD else 1.0

    @staticmethod
    def _regress(value: float, baseline: float, lam: float, elapsed: float) -> float:
        """向 baseline 回归：value(t) = baseline + (value0 - baseline) * e^(-λt)。"""
        if lam <= 0:
            return value
        return baseline + (value - baseline) * math.exp(-lam * elapsed)

    # ---- 对话增量 ----

    def apply_dialogue_delta(self, delta: dict, bypass_cap: bool = False) -> dict:
        """叠加本轮对话增量（含 delta cap + 硬钳位 + 软熔断 + 联合熔断）。

        delta 格式：{"pleasure": ΔP, "arousal": ΔA, "dominance": ΔD,
                     "possessiveness": ΔPoss, "trust": {"presence": Δ, "safety": Δ}}
        bypass_cap=True 时跳过 delta cap（规则层精心设计的增量直接应用，只受硬钳位约束）。
        返回实际应用的增量（含 cap 截断）。
        """
        # ① delta cap：每维 |Δ| ≤ 0.3（规则层可 bypass）
        cap_range = (-1.0, 1.0) if bypass_cap else (-MAX_DELTA, MAX_DELTA)
        capped = {}
        for key in ("pleasure", "arousal", "dominance"):
            capped[key] = _clamp_float(delta.get(key, 0.0), cap_range, 0.0)
        capped["possessiveness"] = _clamp_float(
            delta.get("possessiveness", 0.0), cap_range, 0.0
        )
        # v4：亲密感/依恋度增量（cap 同其他轴）
        for rel_key in ("intimacy", "attachment"):
            capped[rel_key] = _clamp_float(delta.get(rel_key, 0.0), cap_range, 0.0)

        # ② 叠加到当前值
        for key in ("pleasure", "arousal", "dominance"):
            self.state["pad"][key] = _clamp_float(
                self.state["pad"][key] + capped[key], PAD_RANGE, self.state["pad"][key]
            )
        self.state["possessiveness"] = _clamp_float(
            self.state["possessiveness"] + capped["possessiveness"],
            POSS_RANGE,
            self.state["possessiveness"],
        )
        # v4：亲密感/依恋度叠加
        for rel_key in ("intimacy", "attachment"):
            self.state[rel_key] = _clamp_float(
                self.state[rel_key] + capped[rel_key],
                POSS_RANGE,
                self.state[rel_key],
            )

        # ③ trust 更新（慢速累积）
        if isinstance(delta.get("trust"), dict):
            for key in ("presence", "safety"):
                t_delta = _clamp_float(delta["trust"].get(key, 0.0), (-0.1, 0.1), 0.0)
                self.state["trust"][key] = _clamp_float(
                    self.state["trust"][key] + t_delta, TRUST_RANGE, self.state["trust"][key]
                )

        # ④ 方向检测：连续 DIRECTION_GUARD_N 轮同向 → 追加回拉（常量此前零引用）
        self._apply_direction_guard(capped)

        # ⑤ 联合熔断：危险联合状态触发
        self._apply_joint_breaker()

        return capped

    @staticmethod
    def _direction_of(delta: dict) -> int:
        """本轮增量在愉悦轴的方向：+1 / -1 / 0。"""
        v = delta.get("pleasure", 0.0)
        if v > 1e-9:
            return 1
        if v < -1e-9:
            return -1
        return 0

    def _apply_direction_guard(self, capped: dict) -> None:
        """连续 DIRECTION_GUARD_N 轮同向推愉悦轴时，往 baseline 拉一点。

        原实现只定义了 DIRECTION_GUARD_N=5 与 docstring 里的「方向检测」，
        无任何代码读取。此处按注释字面接通：靠 history 里的 delta 判连续轮次，
        不改 schema（旧条目无 delta 时按 0 处理，不计入连续）。
        """
        d = self._direction_of(capped)
        if not d:
            return
        streak = 0
        for entry in reversed(self.state.get("history") or []):
            if self._direction_of(entry.get("delta") or {}) == d:
                streak += 1
            else:
                break
        if streak + 1 >= DIRECTION_GUARD_N:
            self.state["pad"]["pleasure"] = _clamp_float(
                self.state["pad"]["pleasure"] - d * 0.05, PAD_RANGE,
                self.state["pad"]["pleasure"],
            )

    def _apply_joint_breaker(self) -> list[str]:
        """联合熔断：检测危险联合状态，返回触发的熔断类型列表。"""
        p = self.state["pad"]["pleasure"]
        a = self.state["pad"]["arousal"]
        poss = self.state["possessiveness"]
        triggered = []

        # 病态占有：P < -0.6 且 Poss > 0.7
        if p < -0.6 and poss > 0.7:
            triggered.append("病态占有")
            # Poss 衰减加倍（临时降低 Poss，向 baseline 回拉）
            # default 必须是 0 而非 poss：回退值若取 poss，poss=0.05 时
            # 熔断会把 Poss 卡死在 0.05 永远拉不回 0（审查 S1，2026-08-29）
            self.state["possessiveness"] = _clamp_float(
                poss - 0.05, POSS_RANGE, 0.0
            )
        # 崩溃边缘：P < -0.7 且 A > 0.7
        if p < -0.7 and a > 0.7:
            triggered.append("崩溃边缘")
            self.state["pad"]["pleasure"] = _clamp_float(p + 0.1, PAD_RANGE, p)
        # 麻木：P < -0.5 且 A < -0.5
        if p < -0.5 and a < -0.5:
            triggered.append("麻木")
            self.state["pad"]["arousal"] = _clamp_float(a + 0.1, PAD_RANGE, a)

        return triggered

    # ---- 落盘 ----

    def save(self, event: str = "", delta: dict | None = None) -> None:
        """原子写落盘，含 history 记录。"""
        # 记录 history（含 pad_after 快照）
        if event:
            entry = {
                "t": _now_utc_iso(),
                "event": event,
                "delta": delta or {},
                "pad_after": {
                    "pleasure": self.state["pad"]["pleasure"],
                    "arousal": self.state["pad"]["arousal"],
                    "dominance": self.state["pad"]["dominance"],
                    "possessiveness": self.state["possessiveness"],
                },
            }
            self.state["history"].append(entry)
            self.state["history"] = self.state["history"][-MAX_HISTORY:]

        # 原子写（事务内已持有锁则复用，否则单独拿锁）
        if self._in_transaction:
            _atomic_write(self.path, self.state)
        else:
            lock, _locked = _acquire_lock(self.lock_path)
            try:
                _atomic_write(self.path, self.state)
            finally:
                _release_lock(lock)

    # ---- 重置 / 手动修正 ----

    def reset(self) -> None:
        """重置情绪到 baseline。"""
        self.state = cold_start_state()
        self.save(event="reset")

    def set_dimension(self, dim: str, value: float) -> None:
        """手动修正某维（调试/异常恢复）。"""
        if dim in ("pleasure", "arousal", "dominance"):
            self.state["pad"][dim] = _clamp_float(value, PAD_RANGE, self.state["pad"][dim])
        elif dim == "possessiveness":
            self.state["possessiveness"] = _clamp_float(value, POSS_RANGE, self.state["possessiveness"])
        elif dim in ("intimacy", "attachment"):
            # v4：亲密感/依恋度手动修正
            self.state[dim] = _clamp_float(value, POSS_RANGE, self.state[dim])
        elif dim in ("presence", "safety"):
            self.state["trust"][dim] = _clamp_float(value, TRUST_RANGE, self.state["trust"][dim])
        else:
            raise ValueError(f"未知维度: {dim}（可选 pleasure/arousal/dominance/possessiveness/intimacy/attachment/presence/safety）")
        self.save(event=f"set {dim}={value}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    """命令行入口。"""
    args = sys.argv[1:]

    if not args or args[0] in ("--dump", "-d"):
        st = EmotionState()
        print(json.dumps(st.dump(), ensure_ascii=False, indent=2))
        return

    if args[0] in ("--reset", "-r"):
        st = EmotionState()
        st.reset()
        print("已重置到 baseline。")
        return

    if args[0] in ("--set", "-s") and len(args) >= 2:
        st = EmotionState()
        for kv in args[1:]:
            if "=" not in kv:
                print(f"格式错误: {kv}（应为 维度=值，如 P=0.5）")
                continue
            dim, val = kv.split("=", 1)
            # 支持别名 P/A/D/Poss/Intim/Attach
            alias = {"P": "pleasure", "A": "arousal", "D": "dominance", "Poss": "possessiveness",
                     "Intim": "intimacy", "Attach": "attachment"}
            dim = alias.get(dim, dim)
            try:
                st.set_dimension(dim, float(val))
                print(f"已设置 {dim} = {val}")
            except (ValueError, TypeError) as e:
                print(f"设置失败 {kv}: {e}")
        return

    print(__doc__)


if __name__ == "__main__":
    _cli()

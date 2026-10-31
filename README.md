# affect-engine

**An emotion engine for AI companions — the skeleton ships, the personality is yours.**

Most "emotion" layers for LLM agents are a sentiment classifier bolted onto the chat loop: user says something sad, the agent's reply gets a sad sticker. A companion's emotion is a different problem:

- **A mood should have a life of its own.** It drifts back to a baseline, decays with per-axis half-lives, reacts to being neglected, and recovers on its own — not just mirror the user's last message.
- **The engine is a skeleton; the personality is config.** Baselines, decay rates, keyword rules, and the first-person mood sentences are all yours to write. We ship neutral examples and get out of the way.
- **Numbers stay inside the engine.** The projection layer emits moods, tiers and descriptions — never raw values. Your companion's inner state is not a debug dashboard.

## Dimensions

| Axis | Range | Meaning |
|---|---|---|
| `pleasure` / `arousal` / `dominance` | −1..1 | The PAD model (Mehrabian) |
| `possessiveness` | 0..1 | How tightly the companion holds on |
| `intimacy` / `attachment` | 0..1 | Relationship axes — slow-moving, hard to earn |
| `trust.presence` / `trust.safety` | 0..1 | Trust twin: "are you there" / "are you safe" |

Every axis has its own decay half-life (arousal ~30min, pleasure ~6h, relationship axes ~days) and its own baseline — moods regress toward *temperament*, not toward zero.

## Quick start

```python
from engine import EmotionEngine

eng = EmotionEngine()                     # state file: emotion_state.json (override: EMOTION_STATE_PATH)
eng.on_user_message("I missed you")       # feed a user turn: rules fire, mood shifts, state persists
print(eng.emotion_context())              # inject this into your system prompt each turn
print(eng.live2d_params())                # or drive a Live2D avatar directly
```

No dependencies. Python 3.10+. State is a single JSON file with atomic writes, file locking and corruption recovery.

## How it breathes

- **Baseline regression** — `value(t) = baseline + (value₀ − baseline)·e^(−λt)`. Joy fades, calm returns. Cold-shoulder time pushes pleasure down and possessiveness up, scaled by how much the companion trusts your presence.
- **Guardrails, layered** — per-turn delta cap, hard clamps, a soft breaker that doubles decay when a value runs extreme, joint breakers for dangerous states (e.g. high arousal + very low pleasure), and a direction guard that pulls back after N consecutive same-direction pushes. A single message cannot wreck a mood; a mood cannot spiral.
- **Self-source events** — `eng.apply_self_event("weather_bad")`. Moods with their own birthplace: the engine only consumes *event-causal* fuel (weather, a boring morning, a find worth sharing) and never generates time-ticked increments. No `+1 per day` — a mood that changes without a reason is a cron job, not a feeling.

## Make it yours

Copy `config.example.json` to `emotion_config.json` (gitignored) next to the engine:

```json
{
  "baseline": {"pleasure": 0.1, "possessiveness": 0.4},
  "lambda":   {"intimacy": 0.002},
  "rules":    [[["love you", "miss you"], {"pleasure": 0.3, "intimacy": 0.05}]]
}
```

- `baseline` — the temperament your companion regresses to. Anxious? Set arousal high. Clingy? Set attachment high.
- `lambda` — the breathing rhythm: how fast each axis settles. Invalid values fall back to defaults; out-of-range baselines are clamped.
- `rules` — keyword → delta table, replaces the shipped example table entirely. Multi-axis deltas, trust sub-deltas, and cooldown semantics all work the same.

One layer we deliberately do **not** configure for you: the first-person mood sentences in `engine.py::_describe_mood`. That is the most *personality*-dense layer of the engine — read the nine example tiers, then rewrite them in your companion's voice.

## What's in the box

| File | Role |
|---|---|
| `emotion_state.py` | State, persistence, decay, guardrails, config loading |
| `rules.py` | Keyword rule layer (fast path, synchronous) + config override |
| `engine.py` | Orchestration: per-turn pipeline, mood projection, self-source events, Live2D params |
| `server.py` | Optional: local HTTP server exposing `/api/emotion` for avatar frontends |

## Tests

```bash
python test_emotion_state.py    # state, decay, guardrails, config override
python test_engine.py           # rules, orchestration, context generation
python test_self_event.py       # self-source feed: invariants, cooldown, budget, stress
```

## Notes

- Code comments are written in Chinese. English translations welcome via PR.
- The PAD + five-relationship-axes design was inspired by [CyberPersona](https://github.com/harrylarryxyz/CyberPersona)'s relationship decomposition — ideas only, no code was taken.
- Built as one piece of a companion stack: pair it with [her.memory](https://github.com/runs-agents/her.memory) (memory) and [session-distillation](https://github.com/runs-agents/session-distillation) (judgment) if you want the full anatomy.

## License

MIT — see [LICENSE](LICENSE).

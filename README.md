---
title: CustomerSupportEnv — AI Decision Engine
emoji: 🤖
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - multi-step
---

# 🤖 CustomerSupportEnv

> **Meta OpenEnv Hackathon 2026** — Real-world AI customer support decision-making environment

A production-quality OpenEnv environment where an AI agent must **triage, prioritise, and resolve customer complaints** across multi-step episodes using intelligent decision-making under uncertainty.

---

## 🎯 The Task

The agent acts as a virtual customer support manager. Each episode presents a stream of customer complaints spanning 5 categories:

| Category | Examples |
|----------|---------|
| 💳 Billing | Double charges, subscription after cancel |
| 📦 Delivery | Late orders, lost packages |
| 🔧 Quality | Defective products, warranty disputes |
| 🖥️ Technical | App crashes, checkout bugs |
| 📜 Policy | Return window edge cases |

For each complaint the agent must choose **one action** from:

- `refund` — Issue a monetary refund
- `replace` — Send a replacement product
- `escalate` — Escalate to senior support
- `apologize` — Apologize without cost
- `ignore` — No action (high risk)
- `investigate` — Gather more information

The agent also reports a **confidence score** [0–1] and **free-text reasoning**, which are used in reward computation.

---

## 🔢 Observation Space

```python
class SupportObservation(Observation):
    complaints: List[Dict]        # Active complaint batch
    episode_step: int             # Current step
    max_steps: int                # Episode length
    cumulative_reward: float      # Running total
    satisfaction_score: float     # Customer satisfaction [0,1]
    budget_remaining: float       # Available resolution budget (USD)
    escalation_count: int         # Escalations used so far
    backlog_size: int             # Unresolved complaints carried over
    last_step_feedback: List[Dict] # Reward breakdown from last step
    task_name: str
    task_description: str
    done: bool
    reward: float
    metadata: Dict
```

Each complaint dict contains:
```json
{
  "complaint_id": "uuid",
  "text": "raw complaint text",
  "category": "billing|delivery|quality|technical|policy",
  "priority": "critical|high|medium|low",
  "sentiment_score": -0.8,
  "customer_tier": "vip|regular|new",
  "days_since_purchase": 12,
  "previous_complaints": 2,
  "estimated_order_value": 149.99,
  "context_clues": ["chargeback threat", "2 week no response"],
  "ambiguity_level": 0.3
}
```

---

## 🎬 Action Space

```python
class SupportAction(Action):
    complaint_id: str      # UUID of the complaint to resolve
    decision: str          # refund|replace|escalate|apologize|ignore|investigate
    confidence: float      # [0.0, 1.0] — agent's self-reported confidence
    reasoning: str         # Free-text explanation
    urgency_flag: bool     # Whether agent considers this urgent
```

---

## 🏆 Reward Function

The reward function is **multi-component** and provides dense signal throughout the episode:

```
R = base_correctness × priority_weight × tier_weight
  + calibration_bonus(confidence, correctness)
  + budget_penalty(cost, budget_remaining)
  + strategy_penalty(decision_history)
  + backlog_penalty(unresolved_complaints)
```

| Component | Range | Description |
|-----------|-------|-------------|
| Base correctness | −0.30 / +0.20 / +0.50 | Wrong / Acceptable / Optimal |
| Priority weight | 0.7 – 1.6 | Critical issues matter more |
| Tier weight | 0.8 – 1.4 | VIP customers have higher impact |
| Calibration bonus | −0.20 – +0.15 | Reward well-calibrated confidence |
| Budget penalty | 0 – −0.30 | Penalise over-budget decisions |
| Strategy penalty | 0 – −0.15 | Penalise degenerate strategies |
| Backlog penalty | −0.05/complaint | Unresolved complaints accumulate |

All rewards normalised to `[−1.0, 1.0]`.

**Partial correctness** — sub-optimal but defensible actions (e.g., `refund` when `investigate` is optimal) receive `+0.20` rather than full `+0.50`.

---

## 📋 Task Levels & Graders

### Easy — Basic Triage
- **Objective**: Handle 1 clear-cut complaint per step (3 steps)
- **Complaints**: Strongly signalled correct actions
- **Budget**: $500
- **Pass threshold**: 0.70 / 1.0
- **Expected difficulty**: Rule-based agent should score ~0.75

### Medium — Operational Support
- **Objective**: Handle 2 complaints/step (4 steps), some ambiguous policy cases
- **Budget**: $800
- **Pass threshold**: 0.55 / 1.0
- **Expected difficulty**: Simple heuristic agent scores ~0.60

### Hard — Enterprise Crisis Management
- **Objective**: Handle 3 complaints/step (5 steps) with budget risk, VIP sensitivity, chargeback threats
- **Budget**: $1500
- **Pass threshold**: 0.40 / 1.0
- **Expected difficulty**: Even LLM agents struggle; requires multi-factor reasoning

### Grader Implementation

All graders are deterministic and reproducible:

```python
def grade_episode(history: list) -> float:
    """Score 0.0–1.0. Mixed correctness + reward signal."""
    correct = sum(1 for h in history if h["correct"])
    ratio   = correct / len(history)
    avg_r   = sum(h["reward"] for h in history) / len(history)
    norm_r  = (avg_r + 1.0) / 2.0
    return max(0.0, min(1.0, ratio * 0.6 + norm_r * 0.4))
```

---

## 📊 Baseline Scores

Run the reproducible baseline agent:

```bash
# Start server first
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal
python baseline_agent.py --episodes 3
```

Expected baseline (heuristic agent):

| Task | Avg Score | Pass |
|------|-----------|------|
| Easy | ~0.72 | ✅ |
| Medium | ~0.58 | ✅ |
| Hard | ~0.44 | ✅ |

---

## 🚀 Quick Start

### Option 1: Local Development

```bash
# Install dependencies
uv sync

# Start the OpenEnv server
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# In another terminal — run the dashboard
python app.py

# Or run the baseline agent
python baseline_agent.py
```

### Option 2: Using the Python Client

```python
from practice_env import CustomerSupportEnv, SupportAction

with CustomerSupportEnv(base_url="http://localhost:8000") as env:
    obs = env.reset()
    print(f"Task: {obs.observation.task_name}")
    print(f"Complaints: {len(obs.observation.complaints)}")

    for complaint in obs.observation.complaints:
        action = SupportAction(
            complaint_id=complaint["complaint_id"],
            decision="investigate",
            confidence=0.8,
            reasoning="Gathering more information before committing",
            urgency_flag=complaint["priority"] in ["critical", "high"],
        )
        result = env.step(action)
        print(f"Reward: {result.reward}")
        print(f"Feedback: {result.observation.last_step_feedback}")
```

### Option 3: Docker

```bash
# Build
docker build -t customer-support-env:latest -f server/Dockerfile .

# Run
docker run -p 8000:8000 customer-support-env:latest

# Test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode, returns initial observation |
| `POST` | `/step` | Submit action, returns observation + reward |
| `GET`  | `/state` | Current environment state |
| `GET`  | `/schema` | Action/Observation JSON schemas |
| `WS`   | `/ws` | WebSocket for persistent low-latency sessions |
| `GET`  | `/health` | Container health check |
| `GET`  | `/web` | Interactive web dashboard |
| `GET`  | `/docs` | Swagger API documentation |

---

## 📁 Project Structure

```
customer_support_env/
├── README.md                              # This file
├── openenv.yaml                           # OpenEnv manifest
├── pyproject.toml                         # Project metadata
├── models.py                              # Typed Pydantic models
├── client.py                              # Python client
├── __init__.py                            # Package exports
├── app.py                                 # Gradio dashboard UI
├── baseline_agent.py                      # Reproducible baseline + graders
└── server/
    ├── __init__.py
    ├── app.py                             # FastAPI application
    ├── customer_support_environment.py    # Core RL environment logic
    ├── requirements.txt
    └── Dockerfile
```

---

## 🏗️ Design Decisions

### Why Customer Support?
Customer support triage is a genuine real-world task performed by millions of humans daily. It requires:
- **Multi-factor reasoning** (category + priority + customer history + cost)
- **Uncertainty handling** (ambiguous complaints, partial information)
- **Long-term thinking** (budget management, backlog avoidance)
- **Policy compliance** (not just optimising short-term satisfaction)

### Why Multi-Step?
Single-step environments are trivial. Real support queues are continuous streams. The agent must manage state across complaints and learn that early bad decisions (e.g., blowing the budget on refunds) hurt later steps.

### Why Confidence Scoring?
Well-calibrated uncertainty is a key capability of intelligent systems. The environment rewards agents that know what they don't know, encouraging exploration over overconfident wrong answers.

---

## 📜 License

BSD-style license — see [LICENSE](LICENSE).
# cortex-ai

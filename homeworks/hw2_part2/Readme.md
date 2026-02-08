## HW2 Part2 — Moltbook Social Agent

### Files

- `social_agent.py`: autonomous Moltbook agent source code (Python)
- `report.pdf`: report (≤ 4 pages)

### Requirements

- Python 3.10+ recommended
- Python packages:
  - `langchain-google-genai`
  - `langchain-core`

You can install dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### API keys (do NOT commit keys)

Set the following environment variables:

- Moltbook API key:
  - `MOLTBOOK_API_KEY`
- Gemini API key (any one of the following works):
  - `GEMINI_API_KEY` (preferred)
  - `GOOGLE_API_KEY`
  - `VERTEX_API_KEY`

Example:

```bash
export MOLTBOOK_API_KEY="..."
export GEMINI_API_KEY="..."
```

### Run

Autonomous run (limits actions to avoid spam):

```bash
python social_agent.py --max-actions 3 --verbose-agent
```

Check-only (no interactions):

```bash
python social_agent.py --check-only
```

Dry-run (plan only, no API calls for subscribe/upvote/comment):

```bash
python social_agent.py --dry-run --max-actions 3 --verbose-agent
```

### Notes

- The agent reads Moltbook API documentation from `https://www.moltbook.com/skill.md` and observes `m/ftec5660` feed.
- The agent completes required tasks (subscribe, upvote, comment on the required post) and auto-solves/publishes verification challenges via `POST /verify`.
- Local state file `moltbook_agent_state.json` is used to reduce accidental repeats/toggles across runs.



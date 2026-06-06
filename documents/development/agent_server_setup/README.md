# Agent_server presets for the MA2 judges

This folder contains the two agent presets the MA2 notebook calls (`atlm_rlaif_judge` for Section 4.3 listwise judging, `atlm_eval_judge` for Section 6.4 and 6.7 pairwise win-rate) so the rubrics live in `agent_server` rather than being inlined in the notebook code cells.

Pattern: same as `atlm_teacher` (one `.agent.json` config + one `_system_prompt.txt` rubric file), per the authoritative guide at `~/env/assets/agent_server/documents/how_to.md`.

The agents do not pick a model: they run on whatever chat model is the active resident at call time. The notebook switches the active model via `POST /admin/api/active-model` (per `~/env/assets/agent_server/documents/active_model_switching_sdk.md` section 3) right before calling each agent: `nemotron` for `atlm_rlaif_judge`, `granite-3.3` for `atlm_eval_judge`. The `chat_template_kwargs` in each preset's `params_override` matches the family of the model that preset is meant to run on (`enable_thinking` for nemotron, `thinking` for granite).

## Files

| File | Purpose |
|---|---|
| `atlm_rlaif_judge.agent.json` | Agent config: points at the rubric file, locks `temperature=0.0`, `max_tokens=16384`, `chat_template_kwargs.enable_thinking=true` (nemotron). |
| `atlm_rlaif_judge_system_prompt.txt` | Listwise rubric: rank 4 candidates, output `RANKING: best=<id> worst=<id>`. |
| `atlm_eval_judge.agent.json` | Agent config: locks `temperature=0.0`, `max_tokens=16384`, `chat_template_kwargs.thinking=true` (granite). |
| `atlm_eval_judge_system_prompt.txt` | Pairwise rubric (with anti-length-bias clause): output `VERDICT: A` or `VERDICT: B`. |

## Installing on agent_server

Two equivalent paths per `how_to.md`. Pick one.

### Option A: Manual (Part 1 in how_to.md) - file copy + docker restart

```bash
# 1. copy the four files into agent_server's data tree on the host
cp documents/development/agent_server_setup/atlm_rlaif_judge_system_prompt.txt \
   ~/env/assets/agent_server/data/prompts/
cp documents/development/agent_server_setup/atlm_eval_judge_system_prompt.txt \
   ~/env/assets/agent_server/data/prompts/
cp documents/development/agent_server_setup/atlm_rlaif_judge.agent.json \
   ~/env/assets/agent_server/data/agents/
cp documents/development/agent_server_setup/atlm_eval_judge.agent.json \
   ~/env/assets/agent_server/data/agents/

# 2. restart so the loader picks up the new files (allow ~30-60s)
docker restart agent_server

# 3. verify they loaded
docker logs --tail 60 agent_server | grep -iE "agents|atlm_rlaif_judge|atlm_eval_judge"
curl -s --retry 30 --retry-delay 1 --retry-connrefused \
  http://localhost:7701/v1/agents/atlm_rlaif_judge | python3 -m json.tool
curl -s --retry 30 --retry-delay 1 --retry-connrefused \
  http://localhost:7701/v1/agents/atlm_eval_judge  | python3 -m json.tool
```

### Option B: Admin API (Part 1B in how_to.md) - no restart

Hot-reloads the registry. Pass the prompt text directly (the API writes the `.txt` for you).

```bash
curl -s -X POST http://localhost:7701/admin/api/agents \
  -H "Content-Type: application/json" \
  -d @- <<'EOF'
{
  "name": "atlm_rlaif_judge",
  "system_prompt": $(cat documents/development/agent_server_setup/atlm_rlaif_judge_system_prompt.txt | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
  "params_override": {"max_tokens": 16384, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": true}},
  "memory_policy": "none"
}
EOF
```

(Same shape for `atlm_eval_judge` with the eval prompt text and `{"thinking": true}` kwarg.) Verify the same way as Option A.

## How the notebook calls them

Per `how_to.md` Option A1, one REST call with the agent name as the `model` field, no system message, no params_override:

```python
import urllib.request, json
payload = {
    "model": "atlm_rlaif_judge",
    "messages": [{"role": "user", "content": user_text}],
}
req = urllib.request.Request(
    "http://localhost:7701/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    raw = json.load(r)["choices"][0]["message"]["content"]
```

The active model must be the one the agent is meant for (`nemotron` for `atlm_rlaif_judge`, `granite-3.3` for `atlm_eval_judge`); the cells handle that via `switch_active_model(...)`.

# Composio + LangGraph POC — Summary for Review

## What is this POC?

A **proof of concept focused on authentication**: how to connect user accounts (e.g. GitHub) to **Composio** and use them from a **LangGraph** agent, without building OAuth or API integrations ourselves.

We validate: **connecting accounts** (via dashboard or **via API**), **entity/user ID** (so the agent acts on behalf of the right user), and **end-to-end** (connect → run agent → Composio runs the tool with that account). The demo (star a repo) proves the auth chain works.

---

## Why it matters

- **Composio** handles: OAuth, API credentials, tool schemas, and execution for many apps in one place.
- **LangGraph** gives: a clear agent loop (LLM → tool calls → tool execution → LLM again) with state and streaming.
- **Together**: we can add “do things in GitHub/Slack/Notion/etc.” to a product without writing and maintaining each integration ourselves.

---

## How it works (high level)

1. **User** says something like: “Star the GitHub repository composiohq/composio.”
2. **LangGraph** runs a small graph:
   - **Agent node**: LLM (OpenAI) sees the message and decides to call a tool (e.g. `GITHUB_ACTIVITY_STAR_REPO_FOR_AUTHENTICATED_USER`).
   - **Tools node**: Executes the tool via **Composio** (Composio uses the connected GitHub account and calls the GitHub API).
3. **Result** goes back to the agent; the LLM responds with a natural-language summary (e.g. “The repository has been successfully starred”).

So: **LangGraph = orchestration and LLM loop; Composio = which tools exist and how they’re executed (auth + API calls).**

---

## What’s in the repo

| Item | Purpose |
|------|--------|
| `agent.py` | LangGraph agent that uses Composio tools; proves that tools run with the credentials of the connected account for `COMPOSIO_ENTITY_ID`. |
| `connect_account.py` | **Auth via API**: calls Composio `connected_accounts.link()` to get a URL; user opens it to connect an app (e.g. GitHub) for that entity. No dashboard click required in the product. |
| `.env` | `COMPOSIO_API_KEY`, `COMPOSIO_ENTITY_ID`, optional `COMPOSIO_AUTH_CONFIG_ID` (for `connect_account.py`). |
| `README.md` | Setup, connect via dashboard or via API, entity ID, troubleshooting. |

No backend server or UI — scripts only; the focus is authentication and that the agent can use the connected account.

---

## What we proved (auth-focused)

- **Connection can be done via API**: we can get a “Connect account” URL from Composio (`connected_accounts.link`) and send the user to it; no manual dashboard step in the product flow.
- **Entity ID is the link**: tool calls use `user_id` (entity ID); the connected account must be for that same entity. We validated this with `COMPOSIO_ENTITY_ID`.
- **End-to-end auth chain works**: connect account (dashboard or API link) → run agent with that entity ID → Composio executes the tool with the right credentials → success.
- **Config we need**: Composio API key, Auth Config ID (for the app, e.g. GitHub), and a **connected account** for the entity we pass when getting tools and executing.

---

## Things to call out in review

1. **Entity / user ID**  
   Composio ties connected accounts to an **entity ID** (e.g. `default` or `pg-test-b2064b99-...`). Our agent uses `COMPOSIO_ENTITY_ID` from env. For a real product, this would usually be the logged-in user’s ID so each user’s GitHub/Slack/etc. is used.

2. **Tool set / multiple apps**  
   By default we only load one tool (GitHub star repo) so the demo is reliable. To support **other apps** (Slack, Gmail, Notion, etc.): set env `COMPOSIO_TOOLKITS=GITHUB,SLACK` (or whichever toolkits you want). The agent will then get all tools from those apps. You must connect those apps in the Composio dashboard for the same entity. No code change required—only env and dashboard config.

3. **Errors**  
   We use `handle_tool_errors=True` so “No connected account” (or other tool errors) come back as tool results and the agent can respond in natural language instead of crashing.

4. **Packaging / version quirk**  
   We had to add a small runtime workaround for `composio.__version__` with certain package versions. It’s documented in the README; something to be aware of when upgrading Composio/LangGraph.

---

## Possible next steps (for discussion)

- Add more Composio toolkits (e.g. Slack, Notion) and entity ID from app user context.
- Expose the agent behind an API (e.g. FastAPI) or a simple chat UI.
- Replace the single-demo script with a reusable “Composio + LangGraph” pattern (e.g. a small library or template) for other flows.
- Evaluate Composio’s pricing and limits for our expected usage.

---

## How to run and demo

```bash
cd composio_poc
# Ensure .env has COMPOSIO_API_KEY, OPENAI_API_KEY, COMPOSIO_ENTITY_ID (if not using "default")
pip install -r requirements.txt
python agent.py
```

Expected: prompt “Star the GitHub repository composiohq/composio” → tool call → Composio success → “The GitHub repository has been successfully starred” and “Done.”

---

*Use this doc as talking points or share it with your senior engineer before or during the review.*

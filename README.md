# Composio + LangGraph POC (Authentication)

Proof of concept **focused on authentication**: connect user accounts to Composio (dashboard or API) and use them from a LangGraph agent so it can act on behalf of the right user.

## What this POC does

- **Auth**: Connect apps (e.g. GitHub) for an entity — via dashboard or API (`connect_account.py`). Agent runs tools using that connected account.
- **Demo**: "Star the repo composiohq/composio" proves the chain: connected account → entity ID → Composio tool → success.
- (Optional) The agent receives a natural-language request (e.g. “Star the repo composiohq/composio”), decides to call the right Composio tool, and the workflow runs the tool and returns.

## Step-by-step setup

### 1. Create a virtual environment (recommended)

```bash
cd composio_poc
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get API keys

- **Composio:** Sign up at [app.composio.dev](https://app.composio.dev), create an app, and copy your **API key**.
- **OpenAI:** Get an API key from [platform.openai.com](https://platform.openai.com) (used by the LangGraph agent’s LLM).

### 4. Configure environment

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Edit `.env` and set:

- `COMPOSIO_API_KEY=your_composio_api_key`
- `OPENAI_API_KEY=your_openai_api_key`
- `COMPOSIO_ENTITY_ID=your_entity_id` (optional; default is `"default"`. Set this to the **USER ID** of a connected account in the Composio dashboard if you connected under a different entity.)
- `COMPOSIO_TOOLKITS=GITHUB,SLACK` (optional; comma-separated list of Composio toolkits. If set, the agent gets **all tools** from these apps. If unset, only the GitHub star-repo tool is loaded for the demo. Connect the corresponding apps in the Composio dashboard for your entity.)

### 5. Connect GitHub (or another app) to Composio

You need at least one **connected account** (e.g. GitHub) so the agent can run tools. Do this in the **Composio Dashboard** (the old CLI is deprecated and broken).

#### How to connect GitHub (dashboard, step by step)

1. **Sign in**  
   Go to **[app.composio.dev](https://app.composio.dev)** (or [platform.composio.dev](https://platform.composio.dev)) and log in.

2. **Create an Auth Config for GitHub**  
   - Open the **Auth Configs** page: **[platform.composio.dev → Auth Configs](https://platform.composio.dev?next_page=/auth-configs)**  
   - Click **Create Auth Config**  
   - Select the **GitHub** toolkit  
   - Choose **OAuth2** as the authentication method  
   - For development, use **Composio’s managed authentication** (no need to create your own OAuth app)  
   - Leave default scopes unless you need more permissions  
   - Click **Create Auth Configuration**  
   - Copy or note the **Auth Config ID** (you may need it if you use a custom user ID later)

3. **Connect your GitHub account**  
   - Either use the **Tool Router** quick link for GitHub: **[Try GitHub (connect)](https://platform.composio.dev/auth?next_page=/tool-router?toolkits=github)**  
   - Or in the dashboard: go to **Connected accounts** (or the section where you manage connections), choose **Connect account**, select the GitHub auth config you created, and complete the OAuth flow in the browser (authorize Composio to access your GitHub).

4. **Use the default user**  
   The POC uses `user_id="default"`. When connecting GitHub, ensure the connection is for entity ID **default** (or you’ll see “No connected account found for entity ID default”). If your dashboard has a “default” or similar user, that connection will be used. If you only have one connected account, it’s usually used as default.

**Option B — Connect via API (no dashboard click)**  
Use the script that calls Composio’s API to get a connect link:

1. In the dashboard, create **one** Auth Config for GitHub (or the app you want) and copy its **Auth Config ID** (e.g. `ac_v7suRfc0P_Ve`).
2. In `.env` set:
   - `COMPOSIO_AUTH_CONFIG_ID=ac_xxxx` (the ID you copied)
   - `COMPOSIO_ENTITY_ID=default` (or the user/entity you want this connection under).
3. Run:
   ```bash
   python connect_account.py
   ```
4. Open the printed URL in a browser and complete OAuth. The account is then connected for that entity; `agent.py` will use it when run with the same `COMPOSIO_ENTITY_ID`.

Optional: `COMPOSIO_CALLBACK_URL=https://yourapp.com/callback` to redirect the user after OAuth.

### 6. Run the agent

```bash
python agent.py
```

Default run: the agent is asked to “Star the GitHub repository composiohq/composio”. It should call the Composio GitHub tool and you’ll see the streamed response.

## Project layout

- `agent.py` — LangGraph workflow + Composio toolset; one node calls the LLM, another runs tools.
- `connect_account.py` — Get a "Connect account" URL via Composio API; open the URL to complete OAuth and link the app for `COMPOSIO_ENTITY_ID`.
- `requirements.txt` — `composio-langgraph`, `langgraph`, `langchain-openai`, etc.
- `.env.example` — Template for `COMPOSIO_API_KEY`, `OPENAI_API_KEY`, `COMPOSIO_AUTH_CONFIG_ID`, etc.
- `.env` — Your secrets (create from `.env.example`; do not commit).

## Trying other actions or apps

- **Other GitHub actions:** Change the user message in `agent.py` (e.g. “Create an issue in composiohq/composio titled Hello”).
- **Other Composio apps:** In `agent.py`, change `get_tools(apps=[App.GITHUB])` to e.g. `apps=[App.SLACK]` or `[App.GITHUB, App.SLACK]`, then run `composio add slack` (or the relevant app) and use a prompt that uses that app.

## Troubleshooting

- **`ImportError: cannot import name 'Composio' from 'composio.client'`**  
  This comes from the deprecated **composio-core** package. Uninstall it and use the Dashboard to connect apps instead:
  ```bash
  pip uninstall composio-core
  ```
  Then connect GitHub (or other apps) via [app.composio.dev](https://app.composio.dev) as in step 5 above.

- **`ModuleNotFoundError: No module named 'composio.__version__'`**  
  Caused by a mismatch between the main **composio** package and **composio-langgraph**. The agent includes a small workaround that injects `composio.__version__` at runtime. Do not install both `composio` and `composio-core`; let **composio-langgraph** pull in its required **composio** dependency.

- **`No connected account found for entity ID default for toolkit github`**  
  Connect GitHub in the Composio dashboard and ensure the connection is for the **entity/user ID `default`** (see step 5).

## References

- [Composio LangGraph docs (v1)](https://v1.docs.composio.dev/frameworks/langgraph)
- [Composio quickstart](https://docs.composio.dev/docs/quickstart)
- [LangGraph](https://langchain-ai.github.io/langgraph/)

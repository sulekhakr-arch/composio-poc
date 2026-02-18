"""
Composio + LangGraph Agent with AI Tool Abstraction Layer.

Run: python agent.py

Flow:
  1. Login
  2. Select an app (GitHub, Slack, Notion, etc.)
  3. Auth Config created in code
  4. Logged-in user ID passed to Composio
  5. OAuth connection opens in browser
  6. User describes what they want (e.g. "Book an appointment")
  7. AI classifies tool fields into Primary / Secondary / Auto
  8. Chatbot collects only Primary fields from user, auto-fills the rest
  9. Tool executes with merged fields
"""
# Workaround: composio.__version__ module may be missing
import sys
import types
if "composio.__version__" not in sys.modules:
    _v = types.ModuleType("composio.__version__")
    _v.__version__ = "0.11.0"
    sys.modules["composio.__version__"] = _v

import json
import os
import re
import time
import webbrowser
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from composio import Composio
from composio_langgraph import LanggraphProvider


# =============================================================================
# Configuration
# =============================================================================

AVAILABLE_TOOLKITS = ["GITHUB", "SLACK", "NOTION", "GMAIL", "GOOGLECALENDAR"]
# Note: GMAIL and GOOGLECALENDAR require a verified Google OAuth app.
# They will work in production with the product team's OAuth credentials.

# Composio clients
composio_client = Composio()
composio_langgraph = Composio(provider=LanggraphProvider())


# =============================================================================
# Step 1 — Auth Config (created in code, no dashboard)
# =============================================================================

# Cache auth config IDs so we don't recreate every run
AUTH_CONFIG_CACHE_FILE = os.path.join(os.path.dirname(__file__), "auth_configs.json")


def load_auth_config_cache() -> dict:
    if os.path.exists(AUTH_CONFIG_CACHE_FILE):
        with open(AUTH_CONFIG_CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_auth_config_cache(cache: dict):
    with open(AUTH_CONFIG_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def get_or_create_auth_config(toolkit: str) -> str:
    """
    Create auth config with required scopes — all in Python, no dashboard.
    Caches the ID so it's only created once.
    """
    cache = load_auth_config_cache()

    # 1. Use cached/manual ID if available
    if toolkit in cache:
        return cache[toolkit]

    # 2. Try Composio's default config for this toolkit
    try:
        auth_config_id = composio_client.toolkits._get_auth_config_id(
            toolkit=toolkit.lower()
        )
        cache[toolkit] = auth_config_id
        save_auth_config_cache(cache)
        return auth_config_id
    except Exception:
        pass

    # 3. Create a new one programmatically
    try:
        response = composio_client.auth_configs.create(
            toolkit=toolkit.lower(),
            options={
                "name": f"{toolkit}_full_access",
                "type": "use_composio_managed_auth",
            },
        )
        auth_config_id = response.id
    except Exception as exc:
        print(f"    [!] Could not create auth config: {exc}")
        print(f"    Create it in Composio dashboard and add to auth_configs.json:")
        print(f'    {{ "{toolkit}": "ac_xxxxx" }}')
        return None

    cache[toolkit] = auth_config_id
    save_auth_config_cache(cache)
    return auth_config_id


# =============================================================================
# Step 2 — User ID (from your app's login system)
# =============================================================================

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")


def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def login() -> tuple:
    """
    Login flow. In production, replaced by your actual auth (JWT, session, etc.).
    Returns (username, user_id).
    """
    users = load_users()

    if users:
        print(f"  Existing users: {', '.join(users.keys())}")
    print("  Enter your username (new or existing):")

    try:
        username = input("  Username > ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(0)

    if not username:
        print("  No username entered. Bye!")
        sys.exit(0)

    if username in users:
        user_id = users[username]
        print(f"  Welcome back, {username}! (user_id: {user_id})")
    else:
        user_id = f"{username}@user"
        users[username] = user_id
        save_users(users)
        print(f"  New user created: {username} (user_id: {user_id})")

    return username, user_id


# =============================================================================
# Step 3 — Connect Account (OAuth)
# =============================================================================

def check_connection(toolkit: str, user_id: str) -> bool:
    """Check if user has an active connection for a toolkit."""
    try:
        connections = composio_client.connected_accounts.list(
            user_ids=[user_id],
            statuses=["ACTIVE"],
        )
        for conn in connections.items:
            tk = getattr(conn, "toolkit", None)
            slug = getattr(tk, "slug", "") if tk else ""
            if slug.upper() == toolkit.upper():
                return True
        return False
    except Exception:
        return False


def get_connected_toolkits(user_id: str) -> list:
    """Get list of connected toolkit names for a user."""
    connected = []
    try:
        connections = composio_client.connected_accounts.list(
            user_ids=[user_id],
            statuses=["ACTIVE"],
        )
        for conn in connections.items:
            tk = getattr(conn, "toolkit", None)
            slug = getattr(tk, "slug", "").upper() if tk else ""
            if slug and slug not in connected:
                connected.append(slug)
    except Exception:
        pass
    return connected


def connect_account(toolkit: str, user_id: str, auth_config_id: str) -> bool:
    """Connect an account via OAuth. Opens browser, polls until connected."""
    try:
        print(f"    Creating connection for user '{user_id}'...")
        connection_request = composio_client.connected_accounts.initiate(
            user_id=user_id,
            auth_config_id=auth_config_id,
        )

        redirect_url = getattr(connection_request, "redirect_url", None)
        if not redirect_url:
            print(f"    [!] No redirect URL for {toolkit}.")
            return False

        print(f"\n    Open this URL to connect your {toolkit} account:")
        print(f"    {redirect_url}")
        print()

        try:
            webbrowser.open(redirect_url)
            print("    (Browser opened. Complete the OAuth flow.)")
        except Exception:
            print("    (Could not open browser. Copy the URL above.)")

        print(f"    Waiting for {toolkit} connection...", end="", flush=True)
        for _ in range(60):
            time.sleep(2)
            print(".", end="", flush=True)
            if check_connection(toolkit, user_id):
                print(" Connected!")
                return True

        print(" Timed out.")
        return False

    except Exception as exc:
        if "Multiple connected accounts" in str(exc):
            return True
        print(f"    [!] Error: {exc}")
        return False


# =============================================================================
# Interactive Setup
# =============================================================================

def interactive_setup(user_id: str) -> list:
    """Interactive app connection for the logged-in user."""
    while True:
        connected = get_connected_toolkits(user_id)

        print(f"\n  {'=' * 50}")
        print(f"  Logged-in User ID: {user_id}")
        print(f"  {'=' * 50}")

        print(f"\n  Available Apps:")
        for i, tk in enumerate(AVAILABLE_TOOLKITS, 1):
            if tk in connected:
                print(f"    {i}. {tk}  [connected]")
            else:
                print(f"    {i}. {tk}")

        print()
        if connected:
            print("  Type a number to connect another app, or 'done' to start chatbot.")
        else:
            print("  Type a number to connect your first app.")
        print("  Type 'quit' to exit.")

        try:
            choice = input("\n  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if choice in ("done", "d"):
            if not connected:
                print("  Connect at least one app first!")
                continue
            break
        if choice == "":
            if connected:
                break
            continue
        if choice in ("quit", "exit", "q"):
            print("Bye!")
            sys.exit(0)

        toolkit = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(AVAILABLE_TOOLKITS):
                toolkit = AVAILABLE_TOOLKITS[idx]
            else:
                print("  Invalid number.")
                continue
        else:
            toolkit = choice.upper()

        if toolkit not in AVAILABLE_TOOLKITS:
            print(f"  '{toolkit}' is not available.")
            continue

        if toolkit in connected:
            print(f"  {toolkit} is already connected!")
            continue

        # Step 1: Get Auth Config
        print(f"\n  Step 1: Getting auth config for {toolkit}...")
        try:
            auth_config_id = get_or_create_auth_config(toolkit)
            if not auth_config_id:
                continue
            print(f"    Auth Config ID: {auth_config_id}")
        except Exception as exc:
            print(f"    [!] Failed: {exc}")
            continue

        # Step 2: Pass logged-in user ID
        print(f"\n  Step 2: Using logged-in user ID: {user_id}")

        # Step 3: Connect Account (OAuth)
        print(f"\n  Step 3: Connecting {toolkit} account...")
        success = connect_account(toolkit, user_id, auth_config_id)
        if success:
            print(f"\n  [OK] {toolkit} connected for user {user_id}")
        else:
            print(f"\n  [!!] {toolkit} connection failed.")

    return get_connected_toolkits(user_id)


# =============================================================================
# LangGraph Agent + Chatbot
# =============================================================================

TOOLKIT_TOOLS = {
    "GITHUB": [
        "GITHUB_GET_A_REPOSITORY",
        "GITHUB_LIST_REPOSITORIES_FOR_AUTHENTICATED_USER",
        "GITHUB_LIST_ISSUES_FOR_A_REPOSITORY",
        "GITHUB_CREATE_AN_ISSUE",
        "GITHUB_CREATE_A_PULL_REQUEST",
        "GITHUB_ACTIVITY_STAR_REPO_FOR_AUTHENTICATED_USER",
        "GITHUB_GET_A_USER",
        "GITHUB_GET_THE_AUTHENTICATED_USER",
    ],
    "GMAIL": [
        "GMAIL_SEND_EMAIL",
        "GMAIL_FETCH_EMAILS",
    ],
    "SLACK": [
        "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL",
    ],
    "NOTION": [
        "NOTION_ADD_PAGE_CONTENT",
        "NOTION_ADD_MULTIPLE_PAGE_CONTENT",
    ],
    "GOOGLECALENDAR": [
        "GOOGLECALENDAR_CREATE_EVENT",
        "GOOGLECALENDAR_LIST_CALENDARS",
        "GOOGLECALENDAR_FIND_FREE_SLOTS",
        "GOOGLECALENDAR_DELETE_EVENT",
    ],
}


def build_agent(toolkits: list, user_id: str):
    """Build the LangGraph agent with tools from connected toolkits."""
    tool_slugs = []
    fallback_toolkits = []
    for tk in toolkits:
        if tk in TOOLKIT_TOOLS and TOOLKIT_TOOLS[tk] is not None:
            tool_slugs.extend(TOOLKIT_TOOLS[tk])
        else:
            fallback_toolkits.append(tk)

    if tool_slugs:
        tools = composio_langgraph.tools.get(user_id=user_id, tools=tool_slugs)
    else:
        tools = []

    if fallback_toolkits:
        extra = composio_langgraph.tools.get(user_id=user_id, toolkits=fallback_toolkits)
        tools = list(tools) + list(extra)

    tool_node = ToolNode(tools, handle_tool_errors=True)
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState):
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist)
        system_msg = SystemMessage(content=(
            f"You are a helpful assistant with access to external tools. "
            f"Today is {now.strftime('%A, %B %d, %Y')}. "
            f"The current time is {now.strftime('%I:%M %p')} IST (Asia/Kolkata, UTC+5:30). "
            f"The user is in the Indian timezone (IST). "
            f"When the user refers to relative dates like 'tomorrow', 'next week', "
            f"'today', always compute the correct calendar date from the above. "
            f"Use ISO format (YYYY-MM-DDTHH:MM) for all datetime tool parameters. "
            f"Default timezone for calendar events is Asia/Kolkata."
        ))
        messages = [system_msg] + state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "tools"
        return "__end__"

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_edge("__start__", "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile(), len(tools)


# =============================================================================
# AI Tool Abstraction Layer
# Classifies Composio tool fields → Primary / Secondary / Auto
# =============================================================================

FIELD_CLASSIFICATION_PROMPT = """You are an AI tool abstraction layer for MyOperator.

Given a user's objective and a Composio tool's parameters, classify each field into:
- PRIMARY: Essential input needed for the user's goal
- SECONDARY: Optional config with sensible defaults (e.g. timezone, duration)
- AUTO: System-controlled, user never sees (e.g. calendar_id="primary", send_updates=true)

PRIMARY fields have two sub-types — this is critical:
- STATIC (is_dynamic=false): The value CAN be inferred from the user's objective.
  Example: user says "book dentist appointment" → title = "Dentist Appointment" (inferred).
  For static fields, you MUST provide "generated_value" with the inferred value and
  "generated_description" explaining how it was derived from the objective.
- DYNAMIC (is_dynamic=true): The value is unique/personal and CANNOT be inferred.
  Example: attendee email, specific date/time — these must be asked.
  For dynamic fields, provide "description" with a plain English prompt for the user.

Rules:
1. Minimize PRIMARY fields — only what's absolutely needed
2. Among primary, maximize STATIC — infer as much as possible from the objective
3. Only mark a field DYNAMIC if its value is truly unknowable from the objective
4. Assign safe defaults for SECONDARY and AUTO fields
5. Return valid JSON only, no markdown

User Objective: {objective}
Tool: {tool_name}
Tool Description: {tool_description}
Tool Parameters: {tool_params}

Return JSON in this exact format:
{{
  "tool_slug": "{tool_slug}",
  "objective": "{objective}",
  "primary_fields": [
    {{
      "field_key": "parameter_name",
      "label": "Human-friendly label",
      "is_dynamic": false,
      "generated_value": "value inferred from the objective",
      "generated_description": "How this was inferred from the objective"
    }},
    {{
      "field_key": "parameter_name",
      "label": "Human-friendly label",
      "is_dynamic": true,
      "description": "Plain English prompt asking the user for this value"
    }}
  ],
  "secondary_fields": [
    {{
      "field_key": "parameter_name",
      "label": "Human-friendly label",
      "default_value": "sensible default"
    }}
  ],
  "auto_fields": [
    {{
      "field_key": "parameter_name",
      "value": "auto value"
    }}
  ]
}}
"""


def get_tool_schema(tool_slug: str, user_id: str) -> dict:
    """Fetch the full schema/parameters for a Composio tool."""
    try:
        tools = composio_langgraph.tools.get(user_id=user_id, tools=[tool_slug])
        if tools:
            tool = tools[0]
            schema = {}
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.schema() if hasattr(tool.args_schema, 'schema') else {}
            return {
                "name": tool.name,
                "description": getattr(tool, 'description', ''),
                "parameters": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
    except Exception as exc:
        print(f"    [!] Could not fetch schema for {tool_slug}: {exc}")
    return {}


def classify_tool_fields(objective: str, tool_slug: str, user_id: str) -> dict:
    """
    Use LLM to classify a Composio tool's fields into Primary/Secondary/Auto.
    Returns the JSON contract for UI rendering and execution.
    """
    schema = get_tool_schema(tool_slug, user_id)
    if not schema:
        return None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = FIELD_CLASSIFICATION_PROMPT.format(
        objective=objective,
        tool_name=schema.get("name", tool_slug),
        tool_description=schema.get("description", ""),
        tool_params=json.dumps(schema.get("parameters", {}), indent=2),
        tool_slug=tool_slug,
    )

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        print("    [!] Could not parse LLM response")
        return None


NL_CONVERT_PROMPT = """Convert the user's natural language input into the exact API format required.

Field: {field_key}
Field description: {field_desc}
User input: "{user_input}"
Today's date: {today}
User timezone: IST (Asia/Kolkata, UTC+5:30)

Rules:
- For datetime fields: convert to ISO format YYYY-MM-DDTHH:MM in IST (e.g. "tomorrow 3pm" → "2026-02-19T15:00")
- For email fields: extract the email address, validate format
- For arrays: wrap single values in a list (e.g. "john@x.com" → ["john@x.com"])
- For repo fields: ensure format is "owner/repo"
- If the input is already in the correct format, return it as-is
- Return ONLY the converted value, nothing else. No quotes, no explanation.
"""


def convert_natural_language(field_key: str, field_desc: str, user_input: str) -> str:
    """Convert user's natural language input to API-compatible format."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%Y-%m-%d (%A)")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = NL_CONVERT_PROMPT.format(
        field_key=field_key,
        field_desc=field_desc,
        user_input=user_input,
        today=today,
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().strip('"').strip("'")


def validate_input(field_key: str, value: str) -> tuple:
    """
    Basic validation. Returns (is_valid, error_message).
    """
    lower_key = field_key.lower()

    if "email" in lower_key or "attendee" in lower_key:
        # Check if it looks like an email or list of emails
        emails = re.findall(r'[\w.+-]+@[\w-]+\.[\w.]+', value)
        if not emails:
            return False, "Please enter a valid email address (e.g. name@example.com)"
        return True, None

    if "datetime" in lower_key or "date" in lower_key or "time" in lower_key:
        # Accept ISO format or natural language (will be converted)
        if not value:
            return False, "Please enter a date/time"
        return True, None

    if "repo" in lower_key or "repository" in lower_key:
        if "/" not in value and "owner" not in lower_key:
            return False, "Please use format owner/repo (e.g. composiohq/composio)"
        return True, None

    if not value:
        return False, "This field is required"

    return True, None


def collect_primary_fields(contract: dict) -> dict:
    """
    Collect primary fields conversationally.
    - Static fields (is_dynamic=false): auto-filled from the objective, shown to user
    - Dynamic fields (is_dynamic=true): asked one at a time with validation + NL conversion
    """
    values = {}
    primary = contract.get("primary_fields", [])

    if not primary:
        print("\n  Agent > No input needed — everything is auto-filled!")
        return values

    static_fields = [f for f in primary if not f.get("is_dynamic", True)]
    dynamic_fields = [f for f in primary if f.get("is_dynamic", True)]

    # Auto-fill static fields (inferred from objective)
    if static_fields:
        print("\n  Agent > I inferred these from your request:")
        for field in static_fields:
            label = field.get("label", field["field_key"])
            gen_value = field.get("generated_value", "")
            gen_desc = field.get("generated_description", "")
            if gen_value:
                values[field["field_key"]] = gen_value
                reason = f" ({gen_desc})" if gen_desc else ""
                print(f"    • {label}: {gen_value}{reason}")

        # Let user override if they want
        try:
            override = input("\n  Agent > Look good? Press Enter to confirm, or type 'edit' to change > ").strip()
        except (EOFError, KeyboardInterrupt):
            return values

        if override.lower() == "edit":
            for field in static_fields:
                label = field.get("label", field["field_key"])
                current = values.get(field["field_key"], "")
                try:
                    new_val = input(f"    {label} [{current}] > ").strip()
                except (EOFError, KeyboardInterrupt):
                    return values
                if new_val:
                    values[field["field_key"]] = new_val

    # Collect dynamic fields from user
    if dynamic_fields:
        total = len(dynamic_fields)
        if static_fields:
            print(f"\n  Agent > Now I just need {total} more thing{'s' if total > 1 else ''} from you.")
        else:
            print(f"\n  Agent > I need {total} thing{'s' if total > 1 else ''} from you.")

        for i, field in enumerate(dynamic_fields, 1):
            label = field.get("label", field["field_key"])
            desc = field.get("description", "")
            field_key = field["field_key"]

            if desc:
                print(f"\n  Agent > ({i}/{total}) {desc}")
            else:
                print(f"\n  Agent > ({i}/{total}) What's the {label}?")

            while True:
                try:
                    val = input(f"  You > ").strip()
                except (EOFError, KeyboardInterrupt):
                    return values

                if not val:
                    print(f"  Agent > I need this to proceed. Please enter the {label}.")
                    continue

                is_valid, error_msg = validate_input(field_key, val)
                if not is_valid:
                    print(f"  Agent > {error_msg}")
                    continue

                converted = convert_natural_language(field_key, desc or label, val)
                if converted != val:
                    print(f"  Agent > Got it! I'll use: {converted}")

                values[field_key] = converted
                break

    return values


def merge_and_execute(contract: dict, user_values: dict) -> str:
    """Merge primary + secondary + auto fields and build execution query."""
    tool_slug = contract["tool_slug"]

    # Start with auto fields
    params = {}
    for field in contract.get("auto_fields", []):
        params[field["field_key"]] = field["value"]

    # Add secondary defaults
    for field in contract.get("secondary_fields", []):
        params[field["field_key"]] = field.get("default_value", "")

    # Add user-provided primary values
    params.update(user_values)

    # Show summary
    print(f"\n  Agent > Here's what I'll do:")
    for field in contract.get("primary_fields", []):
        key = field["field_key"]
        label = field.get("label", key)
        is_dynamic = field.get("is_dynamic", True)
        tag = "" if is_dynamic else " [inferred]"
        if key in user_values:
            print(f"    • {label}: {user_values[key]}{tag}")

    secondary = contract.get("secondary_fields", [])
    if secondary:
        defaults_str = ", ".join(
            f"{f.get('label', f['field_key'])}={f.get('default_value', '')}"
            for f in secondary if f.get('default_value')
        )
        if defaults_str:
            print(f"    • Defaults: {defaults_str}")

    print(f"\n  Agent > Executing...")

    # Build the execution query for the LangGraph agent
    param_str = ", ".join(f"{k}={v}" for k, v in params.items() if v)
    query = f"Use the tool {tool_slug} with these exact parameters: {param_str}"

    return query


# =============================================================================
# Smart Chatbot (with abstraction layer)
# =============================================================================

FUNCTION_TOOL_MAP = [
    {
        "keywords": ["book", "appointment"],
        "slug": "GOOGLECALENDAR_CREATE_EVENT",
    },
    {
        "keywords": ["schedule", "meeting"],
        "slug": "GOOGLECALENDAR_CREATE_EVENT",
    },
    {
        "keywords": ["create", "event"],
        "slug": "GOOGLECALENDAR_CREATE_EVENT",
    },
    {
        "keywords": ["send", "email"],
        "slug": "GMAIL_SEND_EMAIL",
    },
    {
        "keywords": ["fetch", "email"],
        "slug": "GMAIL_FETCH_EMAILS",
    },
    {
        "keywords": ["read", "email"],
        "slug": "GMAIL_FETCH_EMAILS",
    },
    {
        "keywords": ["star", "repo"],
        "slug": "GITHUB_ACTIVITY_STAR_REPO_FOR_AUTHENTICATED_USER",
    },
    {
        "keywords": ["create", "issue"],
        "slug": "GITHUB_CREATE_AN_ISSUE",
    },
    {
        "keywords": ["list", "repo"],
        "slug": "GITHUB_LIST_REPOSITORIES_FOR_AUTHENTICATED_USER",
    },
    {
        "keywords": ["send", "slack"],
        "slug": "SLACK_SENDS_A_MESSAGE_TO_A_SLACK_CHANNEL",
    },
]


def detect_function(user_input: str) -> str:
    """Match user input to a tool by checking if ALL keywords for a function appear anywhere in the input."""
    words = set(user_input.lower().split())
    for entry in FUNCTION_TOOL_MAP:
        if all(kw in words for kw in entry["keywords"]):
            return entry["slug"]
    return None


def run_chatbot(agent_app, connected_apps: list, user_id: str):
    """Interactive chatbot with AI tool abstraction."""
    print("\n" + "=" * 60)
    print("  CHATBOT READY")
    print("=" * 60)
    print(f"\n  Connected apps: {', '.join(connected_apps)}")
    print(f"\n  You can:")
    print("    1. Ask anything in natural language (agent mode)")
    print("    2. Set up a function like 'book appointment' (smart mode)")
    print("       → I'll simplify the tool and only ask what's needed")

    print("\n  Examples:")
    if "GITHUB" in connected_apps:
        print("    'Get info about composiohq/composio'")
        print("    'Star the repo composiohq/composio'")
        print("    'Create issue in my-repo titled Bug Report'")
    if "GMAIL" in connected_apps:
        print("    'Send email to user@example.com'")
    if "GOOGLECALENDAR" in connected_apps:
        print("    'Book appointment'  → smart mode, minimal questions")
    if "SLACK" in connected_apps:
        print("    'Send slack message to #general'")

    print("\n  Special commands:")
    print("    'simplify <tool_slug>'  → see field classification for any tool")
    print("    'quit'                  → exit")
    print("-" * 60)

    while True:
        try:
            query = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        # --- Command: simplify a tool ---
        if query.lower().startswith("simplify "):
            tool_slug = query.split(" ", 1)[1].strip().upper()
            objective = input("  What's the user's goal? > ").strip()
            if not objective:
                objective = f"Use {tool_slug}"

            print(f"\n  Classifying fields for {tool_slug}...")
            contract = classify_tool_fields(objective, tool_slug, user_id)
            if contract:
                print(f"\n  JSON Contract:")
                print(json.dumps(contract, indent=2))
            continue

        # --- Smart mode: detect known function ---
        matched_tool = detect_function(query)
        if matched_tool:
            print(f"\n  Detected function → {matched_tool}")
            print(f"  Analyzing tool fields...")

            contract = classify_tool_fields(query, matched_tool, user_id)
            if contract:
                primary = contract.get("primary_fields", [])
                static_count = sum(1 for f in primary if not f.get("is_dynamic", True))
                dynamic_count = sum(1 for f in primary if f.get("is_dynamic", True))
                secondary_count = len(contract.get("secondary_fields", []))
                auto_count = len(contract.get("auto_fields", []))
                print(f"  Fields: {static_count} static + {dynamic_count} dynamic primary, "
                      f"{secondary_count} secondary, {auto_count} auto")

                # Collect only primary fields from user
                user_values = collect_primary_fields(contract)

                # Merge and execute
                exec_query = merge_and_execute(contract, user_values)

                # Run through agent
                for chunk in agent_app.stream(
                    {"messages": [HumanMessage(content=exec_query)]},
                    stream_mode="values",
                ):
                    last = chunk["messages"][-1]
                    if hasattr(last, "content") and last.content and not getattr(last, "tool_calls", None) and last.type == "ai":
                        print(f"\nAgent > {last.content}")
            else:
                print("  Could not analyze tool. Falling back to agent mode...")
                for chunk in agent_app.stream(
                    {"messages": [HumanMessage(content=query)]},
                    stream_mode="values",
                ):
                    last = chunk["messages"][-1]
                    if hasattr(last, "content") and last.content and not getattr(last, "tool_calls", None) and last.type == "ai":
                        print(f"\nAgent > {last.content}")
            continue

        # --- Normal agent mode ---
        for chunk in agent_app.stream(
            {"messages": [HumanMessage(content=query)]},
            stream_mode="values",
        ):
            last = chunk["messages"][-1]
            if hasattr(last, "content") and last.content and not getattr(last, "tool_calls", None) and last.type == "ai":
                print(f"\nAgent > {last.content}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Composio + LangGraph Agent")
    print("  (Fully programmatic — no dashboard needed)")
    print("=" * 60)
    print()

    # Step 0: Login
    username, user_id = login()
    print()

    # Step 1: Connect apps
    connected_toolkits = interactive_setup(user_id)

    if not connected_toolkits:
        print("\n  No apps connected. Bye!")
        sys.exit(0)

    print(f"\n  User:       {username}")
    print(f"  User ID:    {user_id}")
    print(f"  Connected:  {', '.join(connected_toolkits)}")

    # Step 2: Build agent
    print("  Loading tools...")
    agent_app, tool_count = build_agent(connected_toolkits, user_id)
    print(f"  {tool_count} tools loaded")

    # Step 3: Start chatbot with AI abstraction layer
    run_chatbot(agent_app, connected_toolkits, user_id)

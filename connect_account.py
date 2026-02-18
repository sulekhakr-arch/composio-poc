"""
Connect an app (e.g. GitHub) to Composio via API — no dashboard, no auth config ID needed.

Usage:
  python connect_account.py                    # connects GitHub for entity "default"
  python connect_account.py SLACK              # connects Slack
  python connect_account.py GMAIL my-user-123  # connects Gmail for entity "my-user-123"

What it does:
  1. Calls composio.toolkits.authorize() — finds or creates the auth config automatically.
  2. Prints a URL. Open it → complete OAuth → account appears in Composio dashboard.

Env (.env):
  COMPOSIO_API_KEY   — required (from https://app.composio.dev)
"""
import sys
import types
if "composio.__version__" not in sys.modules:
    _v = types.ModuleType("composio.__version__")
    _v.__version__ = "0.11.0"
    sys.modules["composio.__version__"] = _v

import os
from dotenv import load_dotenv
load_dotenv()

from composio import Composio


def main():
    # Args: toolkit (default GITHUB), entity_id (default from env or "default")
    toolkit = sys.argv[1].upper() if len(sys.argv) > 1 else "GITHUB"
    entity_id = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("COMPOSIO_ENTITY_ID", "default")

    composio = Composio()

    print(f"Toolkit:   {toolkit}")
    print(f"Entity ID: {entity_id}")
    print()
    print("Requesting connect link from Composio...")

    # Step 1: Find or create an auth config for this toolkit (no manual dashboard step)
    auth_config_id = composio.toolkits._get_auth_config_id(toolkit=toolkit.lower())
    print(f"Auth Config: {auth_config_id} (auto-resolved)")

    # Step 2: Check if there's already an active connection for this entity + auth config
    existing = composio.connected_accounts.list(
        user_ids=[entity_id],
        auth_config_ids=[auth_config_id],
        statuses=["ACTIVE"],
    )
    if existing.items:
        print(f"\nAlready connected! Entity '{entity_id}' has an active {toolkit} connection.")
        print(f"Account ID: {existing.items[0].id}")
        print("No action needed. Run: python agent.py")
        sys.exit(0)

    # Step 3: Create a connect link
    connection_request = composio.connected_accounts.initiate(
        user_id=entity_id,
        auth_config_id=auth_config_id,
    )

    redirect_url = getattr(connection_request, "redirect_url", None)

    if not redirect_url:
        print("No redirect URL returned. The account may already be connected, or the toolkit doesn't need OAuth.")
        print(f"Connection request ID: {connection_request.id}")
        sys.exit(0)

    print(f"Open this URL in a browser to connect {toolkit}:")
    print(redirect_url)
    print()

    try:
        import webbrowser
        webbrowser.open(redirect_url)
        print("(Browser should have opened.)")
    except Exception:
        pass

    print()
    print("After you complete the OAuth flow:")
    print(f"  - The connection will appear in Composio dashboard under entity '{entity_id}'")
    print(f"  - agent.py will be able to use {toolkit} tools for that entity")


if __name__ == "__main__":
    main()

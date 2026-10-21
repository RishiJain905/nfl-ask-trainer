"""PreToolUse guard: block any tool call that could read or reveal credentials.

Runs before Bash, PowerShell, Read, Grep and Glob calls. Exit code 2 blocks the
call and sends the stderr message back to the agent. See CLAUDE.md -> Security.
"""

import json
import re
import sys

ALLOWED_ENV_FILES = {".env.example"}

# Any ".env"-style filename: .env, .env.local, .env.production, ...
ENV_FILE = re.compile(r"\.env[\w.\-]*", re.IGNORECASE)

# Other credential stores.
CREDENTIAL_FILES = re.compile(
    r"_netrc|\.netrc|kaggle\.json|\.docker[\\/]+config\.json", re.IGNORECASE
)

# Commands that dump environment variables, load the env file, or unwrap secrets.
ENV_DUMP = re.compile(
    r"\bprintenv\b"
    r"|(?:^|[;&|]\s*)env\s*(?:$|[;&|>])"
    r"|(?:^|[;&|]\s*)set\s*(?:$|[;&|>])"
    r"|\bexport\s+-p\b|\bdeclare\s+-[a-z]*x"
    r"|\bcmd(?:\.exe)?\s+/c\s+set\b"
    r"|\b(?:Get-ChildItem|gci|dir|ls|Get-Item|gi)\s+env:"
    r"|\[(?:System\.)?Environment\]::"
    r"|docker(?:-|\s+)compose\s+config"
    r"|docker\s+inspect"
    r"|docker\s+exec\b.*\b(?:env|printenv)\b"
    r"|\benviron\b|getenv\("
    r"|dotenv_values|load_dotenv|find_dotenv|get_secret_value",
    re.IGNORECASE,
)

# Direct references to secret variables in shell syntax ($X, ${X}, $env:X).
SECRET_VARS = re.compile(
    r"(?:\$env:|\$\{|\$)(?:WANDB_API_KEY|NEO4J_PASSWORD|NEO4J_AUTH|LLM_API_KEY"
    r"|OPENROUTER_API_KEY|ODDS_API_KEY\d*|KAGGLE_KEY|SMTP_PASSWORD)\b",
    re.IGNORECASE,
)


def text_to_check(tool: str, tool_input: dict) -> str:
    if tool in ("Bash", "PowerShell"):
        return str(tool_input.get("command", ""))
    keys = ["file_path", "path", "glob"]
    if tool == "Glob":
        keys.append("pattern")
    return " ".join(str(tool_input.get(k, "")) for k in keys)


def violation(tool: str, text: str) -> str | None:
    for match in ENV_FILE.finditer(text):
        if match.group(0).lower() not in ALLOWED_ENV_FILES:
            return f"references a secret env file ({match.group(0)})"
    if CREDENTIAL_FILES.search(text):
        return "references a credential store"
    if tool in ("Bash", "PowerShell"):
        if ENV_DUMP.search(text):
            return "could print environment variables or resolved secrets"
        if SECRET_VARS.search(text):
            return "references a secret environment variable"
    return None


def main() -> int:
    try:
        data = json.load(sys.stdin)
        tool = data.get("tool_name", "")
        reason = violation(tool, text_to_check(tool, data.get("tool_input") or {}))
    except Exception as exc:  # fail closed: an unparseable call is blocked, not allowed
        print(
            f"BLOCKED by block_secrets.py: could not inspect tool call ({exc!r}).", file=sys.stderr
        )
        return 2
    if reason:
        print(
            f"BLOCKED by .claude/hooks/block_secrets.py: this {tool} call {reason}. "
            "Agents must never read or reveal credentials (see CLAUDE.md -> Security). "
            "Use `ask-trainer doctor` (or `uv run nfl doctor` in the NFL repo) to check "
            "whether variables are set, or ask Rishi.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

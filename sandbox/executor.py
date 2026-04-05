"""
sandbox/executor.py
────────────────────────────────────────────────────────────────────────────
E2B Sandbox Executor for viz_mcp_agent.

Optional isolation layer — NOT used by the MCP server in normal operation.
The MCP server renders charts directly (in-process matplotlib).
This module exists as a drop-in upgrade for production environments where
data isolation is required (e.g. untrusted datasets, multi-tenant hosting).

Architecture Fit
─────────────────
The current multi-agent system works as:

  Orchestrator → mcp_server.py → TOOL_REGISTRY[tool](data) → base64 PNG
                                 ↑ direct function call (local)

With the sandbox, the flow becomes:

  Orchestrator → mcp_server.py → SandboxExecutor.run_tool(tool, data)
                                 ↑ serialises data as JSON → ships to E2B
                                 ↑ executes render script remotely
                                 ↑ downloads PNG → base64

Usage (optional — plug into mcp_server.py when needed)
──────────────────────────────────────────────────────────
    from sandbox.executor import SandboxExecutor

    with SandboxExecutor() as executor:
        png_b64 = executor.run_tool(
            tool_name="bar_chart",
            data={"x_col": [...], "y_col": [...], "x_col_name": "Month"},
            title="Monthly Revenue",
            options={"color": "#4C72B0", "show_values": True},
        )

Environment variables
──────────────────────
    E2B_API_KEY       — your E2B API key (https://e2b.dev)  [REQUIRED]
    SANDBOX_TIMEOUT   — idle timeout in seconds (default 300)
    SANDBOX_TEMPLATE  — E2B template id (default "base")
"""

from __future__ import annotations

import base64
import json
import os
import textwrap
import time
import uuid
from typing import Any, Optional

# ---------------------------------------------------------------------------
# E2B import — lazy so the rest of the codebase works without e2b installed.
# The ImportError is only raised when SandboxExecutor is actually instantiated.
# ---------------------------------------------------------------------------
_e2b_available = True
try:
    from e2b import Sandbox
except ImportError:
    _e2b_available = False
    Sandbox = None  # type: ignore


# ── constants ───────────────────────────────────────────────────────────────

_SANDBOX_DEPS = [
    "matplotlib",
    "seaborn",
    "scipy",
    "numpy",
    "pandas",
]

_REMOTE_WORKDIR = "/home/user/viz_agent"
_REMOTE_OUT_DIR = f"{_REMOTE_WORKDIR}/out"

# ---------------------------------------------------------------------------


class SandboxExecutionError(RuntimeError):
    """Raised when the render script exits with a non-zero code."""


class SandboxExecutor:
    """
    Manages one long-lived E2B sandbox for isolated chart rendering.

    Instead of running matplotlib in the MCP server process, this class:
    1. Serialises the data dict + tool params as JSON.
    2. Generates a self-contained Python render script.
    3. Uploads and executes it inside an E2B cloud sandbox.
    4. Downloads the output PNG and returns it as base64.

    The sandbox is created lazily on first use and reused across calls.
    Call `close()` (or use as a context manager) to shut it down.

    Parameters
    ----------
    timeout :
        Sandbox idle timeout in seconds (default: SANDBOX_TIMEOUT env or 300).
    template :
        E2B sandbox template (default: SANDBOX_TEMPLATE env or "base").
    api_key :
        E2B API key (default: E2B_API_KEY env).
    verbose :
        Print debug output to the terminal.
    """

    def __init__(
        self,
        timeout: Optional[int] = None,
        template: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        if not _e2b_available:
            raise ImportError(
                "e2b package not found. Install it with:  pip install e2b"
            )

        self.timeout  = timeout or int(os.getenv("SANDBOX_TIMEOUT", 300))
        self.template = template or os.getenv("SANDBOX_TEMPLATE", "base")
        self.api_key  = api_key or os.getenv("E2B_API_KEY", "")
        self.verbose  = verbose

        if not self.api_key:
            raise EnvironmentError(
                "E2B_API_KEY is not set. "
                "Export it as an environment variable or pass api_key=... explicitly."
            )

        self._sandbox: Optional[Sandbox] = None
        self._deps_installed: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _ensure_sandbox(self) -> Sandbox:
        """Create the sandbox if it doesn't exist yet."""
        if self._sandbox is None:
            if self.verbose:
                print("[executor] Booting E2B sandbox ...")
            self._sandbox = Sandbox(
                template=self.template,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self._run_cmd(f"mkdir -p {_REMOTE_OUT_DIR}")
            if self.verbose:
                print(f"[executor] Sandbox id: {self._sandbox.id}")
        return self._sandbox

    def _ensure_deps(self) -> None:
        """pip-install required packages exactly once per sandbox lifetime."""
        if self._deps_installed:
            return
        deps_str = " ".join(_SANDBOX_DEPS)
        if self.verbose:
            print(f"[executor] Installing deps: {deps_str}")
        result = self._run_cmd(
            f"pip install --quiet {deps_str}",
            timeout=120,
        )
        if result.exit_code != 0:
            raise SandboxExecutionError(
                f"Dependency installation failed:\n{result.stderr}"
            )
        self._deps_installed = True

    def close(self) -> None:
        """Explicitly shut down the sandbox and free resources."""
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception:
                pass
            finally:
                self._sandbox = None
                self._deps_installed = False

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "SandboxExecutor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _run_cmd(self, cmd: str, timeout: int = 60):
        """Run a shell command in the sandbox and return the result."""
        sandbox = self._ensure_sandbox()
        return sandbox.commands.run(cmd, timeout=timeout)

    def _upload_script(self, source: str) -> str:
        """Write source to a uniquely named .py file inside the sandbox."""
        script_name = f"render_{uuid.uuid4().hex[:8]}.py"
        remote_path = f"{_REMOTE_WORKDIR}/{script_name}"
        sandbox = self._ensure_sandbox()
        sandbox.files.write(remote_path, source.encode())
        return remote_path

    def _download_png(self, remote_png_path: str) -> bytes:
        """Read the rendered PNG back from the sandbox."""
        sandbox = self._ensure_sandbox()
        return sandbox.files.read(remote_png_path)

    # ── public API ───────────────────────────────────────────────────────────

    def run_tool(
        self,
        tool_name: str,
        data: dict[str, Any],
        title: str = "",
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Execute a visualization tool inside the sandbox and return base64 PNG.

        This method:
        1. Serialises `data` and `options` as JSON.
        2. Generates a self-contained Python script that imports the tool
           from `tools.visualizations`, reconstructs the numpy arrays,
           calls the tool function, and saves the figure.
        3. Uploads and runs the script inside E2B.
        4. Downloads the output PNG and returns base64.

        Parameters
        ----------
        tool_name :
            Registry key (e.g. "bar_chart", "scatter_plot").
        data :
            Dict of column arrays — values must be JSON-serialisable lists.
            Example: {"x_col": [1, 2, 3], "y_col": [10, 20, 30]}
        title :
            Chart title string.
        options :
            Extra kwargs passed to the tool function
            (e.g. {"color": "#4C72B0", "show_values": True}).

        Returns
        -------
        str
            Base64-encoded PNG (no data-URI prefix).
        """
        options = options or {}

        # 1. Ensure sandbox + deps
        self._ensure_sandbox()
        self._ensure_deps()

        # 2. Generate the render script
        output_name = f"chart_{uuid.uuid4().hex[:8]}.png"
        remote_out = f"{_REMOTE_OUT_DIR}/{output_name}"
        render_script = _build_render_script(
            tool_name=tool_name,
            data=data,
            title=title,
            options=options,
            output_path=remote_out,
        )

        # 3. Upload script
        remote_script = self._upload_script(render_script)

        # 4. Execute
        if self.verbose:
            print(f"[executor] Running {tool_name} in sandbox ...")

        t0 = time.monotonic()
        result = self._run_cmd(f"python {remote_script}", timeout=60)
        elapsed = time.monotonic() - t0

        if self.verbose:
            if result.stdout:
                print("[sandbox stdout]\n", textwrap.indent(result.stdout, "  "))
            if result.stderr:
                print("[sandbox stderr]\n", textwrap.indent(result.stderr, "  "))
            print(f"[executor] Finished in {elapsed:.2f}s  exit={result.exit_code}")

        if result.exit_code != 0:
            raise SandboxExecutionError(
                f"Render script failed (exit {result.exit_code}):\n"
                f"{result.stderr or result.stdout}"
            )

        # 5. Download PNG and base64-encode
        png_bytes = self._download_png(remote_out)
        return base64.b64encode(png_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Render script generator
# ---------------------------------------------------------------------------

def _serialise_for_json(data: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays to lists for JSON serialisation."""
    import numpy as np

    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            # nested dict (e.g. correlation_matrix "columns" key)
            out[k] = _serialise_for_json(v)
        else:
            out[k] = v
    return out


def _build_render_script(
    tool_name: str,
    data: dict[str, Any],
    title: str,
    options: dict[str, Any],
    output_path: str,
) -> str:
    """
    Generate a self-contained Python script that the sandbox can execute.

    The script:
    1. Imports matplotlib + numpy
    2. Reconstructs the data dict from embedded JSON
    3. Calls the tool function from tools.visualizations
    4. Saves the figure to OUTPUT_PATH
    """
    serialised_data = _serialise_for_json(data)
    data_json = json.dumps(serialised_data, default=str)
    options_json = json.dumps(options, default=str)

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ── Auto-generated sandbox render script ──────────────────────────────
        # Tool: {tool_name}
        # ───────────────────────────────────────────────────────────────────────

        import json
        import sys
        import os

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # ── Reconstruct data from embedded JSON ──────────────────────────────
        _raw_data = json.loads('''{data_json}''')

        # Convert lists back to numpy arrays
        data = {{}}
        for key, val in _raw_data.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
                data[key] = np.array(val, dtype=float)
            elif isinstance(val, list):
                data[key] = np.array(val)
            elif isinstance(val, dict):
                # nested dict (correlation matrix columns)
                data[key] = {{k: np.array(v, dtype=float) for k, v in val.items()}}
            else:
                data[key] = val

        title = {title!r}
        options = json.loads('''{options_json}''')
        OUTPUT_PATH = {output_path!r}

        # ── Import and call the tool ─────────────────────────────────────────
        # Add project root so imports work
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from core.base_tool import TOOL_REGISTRY
        import tools.visualizations  # triggers @viz_tool registration

        tool_fn = TOOL_REGISTRY[{tool_name!r}]
        result = tool_fn(data, title, **options)

        if result.success:
            # Save the image bytes directly
            with open(OUTPUT_PATH, "wb") as f:
                import base64
                f.write(base64.b64decode(result.image_b64))
            print(f"Chart saved to {{OUTPUT_PATH}}")
        else:
            print(f"Tool error: {{result.error}}", file=sys.stderr)
            sys.exit(1)
    """)

    return script

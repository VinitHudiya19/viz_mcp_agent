"""
sandbox/executor.py
────────────────────────────────────────────────────────────────────────────
E2B Sandbox Executor for viz_mcp_agent.

Isolation layer for chart rendering.  When enabled, instead of running
matplotlib in the MCP server process, the executor:

  1. Boots a long-lived E2B cloud sandbox.
  2. Uploads project source files (core/, tools/) once per session.
  3. Installs Python deps (matplotlib, seaborn, etc.) once per session.
  4. For each tool call: serialises data → generates render script →
     executes in sandbox → downloads PNG → returns base64.

Usage
──────
    from sandbox.executor import SandboxExecutor

    with SandboxExecutor() as executor:
        png_b64 = executor.run_tool(
            tool_name="bar_chart",
            data={"x_col": [...], "y_col": [...]},
            title="Monthly Revenue",
            options={"color": "#4C72B0"},
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
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# E2B import — lazy so the rest of the codebase works without e2b installed.
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

# Project source files to upload to the sandbox
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_FILES = [
    "core/__init__.py",
    "core/base_tool.py",
    "tools/__init__.py",
    "tools/visualizations.py",
]

# ---------------------------------------------------------------------------


class SandboxExecutionError(RuntimeError):
    """Raised when the render script exits with a non-zero code."""


class SandboxExecutor:
    """
    Manages one long-lived E2B sandbox for isolated chart rendering.

    The sandbox is created lazily on first use and reused across calls
    (avoiding ~2s cold start per request). Project source files and
    dependencies are uploaded/installed once per session.

    Parameters
    ----------
    timeout :
        Sandbox idle timeout in seconds.
    template :
        E2B sandbox template.
    api_key :
        E2B API key.
    verbose :
        Print debug output to terminal.
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
                "e2b package not found. Install it with:  pip install 'e2b>=0.17'"
            )

        self.timeout  = timeout or int(os.getenv("SANDBOX_TIMEOUT", 300))
        self.template = template or os.getenv("SANDBOX_TEMPLATE", "base")
        self.api_key  = api_key or os.getenv("E2B_API_KEY", "")
        self.verbose  = verbose

        if not self.api_key:
            raise EnvironmentError(
                "E2B_API_KEY is not set. "
                "Export it or pass api_key=... explicitly."
            )

        self._sandbox: Optional[Sandbox] = None
        self._deps_installed: bool = False
        self._sources_uploaded: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _ensure_sandbox(self) -> Sandbox:
        """Create the sandbox if it doesn't exist yet."""
        if self._sandbox is None:
            if self.verbose:
                print("[sandbox] Booting E2B sandbox ...")
            self._sandbox = Sandbox.create(
                template=self.template,
                api_key=self.api_key,
                timeout=self.timeout,
            )
            self._run_cmd(f"mkdir -p {_REMOTE_WORKDIR}/core {_REMOTE_WORKDIR}/tools {_REMOTE_OUT_DIR}")
            if self.verbose:
                print(f"[sandbox] Sandbox id: {self._sandbox.sandbox_id}")
        return self._sandbox

    def _ensure_deps(self) -> None:
        """pip-install required packages once per sandbox lifetime."""
        if self._deps_installed:
            return
        deps_str = " ".join(_SANDBOX_DEPS)
        if self.verbose:
            print(f"[sandbox] Installing deps: {deps_str}")
        result = self._run_cmd(
            f"pip install --quiet {deps_str}",
            timeout=120,
        )
        if result.exit_code != 0:
            raise SandboxExecutionError(
                f"Dependency installation failed:\n{result.stderr}"
            )
        self._deps_installed = True

    def _ensure_sources(self) -> None:
        """Upload project source files (core/, tools/) once per sandbox."""
        if self._sources_uploaded:
            return
        sandbox = self._ensure_sandbox()
        if self.verbose:
            print("[sandbox] Uploading project source files ...")
        for rel_path in _SOURCE_FILES:
            local_path = _PROJECT_ROOT / rel_path
            remote_path = f"{_REMOTE_WORKDIR}/{rel_path}"
            if local_path.exists():
                content = local_path.read_bytes()
                sandbox.files.write(remote_path, content)
                if self.verbose:
                    print(f"  -> {rel_path} ({len(content)} bytes)")
            else:
                print(f"  [WARN] Missing: {local_path}")
        self._sources_uploaded = True

    def close(self) -> None:
        """Shut down the sandbox and free resources."""
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception:
                pass
            finally:
                self._sandbox = None
                self._deps_installed = False
                self._sources_uploaded = False

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> "SandboxExecutor":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _run_cmd(self, cmd: str, timeout: int = 60):
        """Run a shell command in the sandbox."""
        sandbox = self._ensure_sandbox()
        return sandbox.commands.run(cmd, timeout=timeout)

    def _upload_script(self, source: str) -> str:
        """Write source to a .py file inside the sandbox."""
        script_name = f"render_{uuid.uuid4().hex[:8]}.py"
        remote_path = f"{_REMOTE_WORKDIR}/{script_name}"
        sandbox = self._ensure_sandbox()
        sandbox.files.write(remote_path, source.encode())
        return remote_path

    def _download_png(self, remote_path: str) -> bytes:
        """Read the rendered PNG from the sandbox."""
        sandbox = self._ensure_sandbox()
        return sandbox.files.read(remote_path, format="bytes")

    # ── public API ───────────────────────────────────────────────────────────

    def run_tool(
        self,
        tool_name: str,
        data: dict[str, Any],
        title: str = "",
        options: dict[str, Any] | None = None,
    ) -> str:
        """
        Execute a visualization tool inside the E2B sandbox.

        Parameters
        ----------
        tool_name :
            Registry key (e.g. "bar_chart", "scatter_plot").
        data :
            Dict of column arrays. Values can be lists or numpy arrays.
        title :
            Chart title.
        options :
            Extra kwargs for the tool function.

        Returns
        -------
        str
            Base64-encoded PNG string.
        """
        options = options or {}

        # 1. Boot sandbox + install deps + upload source files
        self._ensure_sandbox()
        self._ensure_deps()
        self._ensure_sources()

        # 2. Generate render script
        output_name = f"chart_{uuid.uuid4().hex[:8]}.png"
        remote_out = f"{_REMOTE_OUT_DIR}/{output_name}"
        render_script = _build_render_script(
            tool_name=tool_name,
            data=data,
            title=title,
            options=options,
            output_path=remote_out,
        )

        # 3. Upload + execute
        remote_script = self._upload_script(render_script)

        if self.verbose:
            print(f"[sandbox] Running {tool_name} ...")

        t0 = time.monotonic()
        result = self._run_cmd(f"cd {_REMOTE_WORKDIR} && python {remote_script}", timeout=60)
        elapsed = time.monotonic() - t0

        if self.verbose:
            if result.stdout:
                print("[sandbox stdout]\n", textwrap.indent(result.stdout, "  "))
            if result.stderr:
                print("[sandbox stderr]\n", textwrap.indent(result.stderr, "  "))
            print(f"[sandbox] Finished in {elapsed:.2f}s  exit={result.exit_code}")

        if result.exit_code != 0:
            raise SandboxExecutionError(
                f"Render script failed (exit {result.exit_code}):\n"
                f"{result.stderr or result.stdout}"
            )

        # 4. Download PNG → base64
        png_bytes = self._download_png(remote_out)
        return base64.b64encode(png_bytes).decode("utf-8")


# ---------------------------------------------------------------------------
# Render script generator
# ---------------------------------------------------------------------------

def _serialise_for_json(data: dict[str, Any]) -> dict[str, Any]:
    """Convert numpy arrays to JSON-safe lists."""
    import numpy as np

    out = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
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
    Generate a self-contained Python script for the sandbox.

    The script imports the @viz_tool functions from the uploaded project
    files, reconstructs numpy arrays from embedded JSON, runs the tool,
    and saves the output PNG.
    """
    serialised = _serialise_for_json(data)
    data_json = json.dumps(serialised, default=str)
    options_json = json.dumps(options, default=str)

    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        # ── Auto-generated sandbox render script ──
        # Tool: {tool_name}
        # ───────────────────────────────────────────

        import json
        import sys
        import os
        import base64

        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        # ── Reconstruct data from JSON ────────────
        _raw = json.loads('''{data_json}''')

        data = {{}}
        for key, val in _raw.items():
            if isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
                data[key] = np.array(val, dtype=float)
            elif isinstance(val, list):
                data[key] = np.array(val)
            elif isinstance(val, dict):
                data[key] = {{k: np.array(v, dtype=float) for k, v in val.items()}}
            else:
                data[key] = val

        title = {title!r}
        options = json.loads('''{options_json}''')
        OUTPUT_PATH = {output_path!r}

        # ── Import tool from uploaded project files ──
        from core.base_tool import TOOL_REGISTRY
        import tools.visualizations

        tool_fn = TOOL_REGISTRY[{tool_name!r}]
        result = tool_fn(data, title, **options)

        if result.success:
            with open(OUTPUT_PATH, "wb") as f:
                f.write(base64.b64decode(result.image_b64))
            print(f"OK: {{OUTPUT_PATH}}")
        else:
            print(f"FAIL: {{result.error}}", file=sys.stderr)
            sys.exit(1)
    """)

    return script

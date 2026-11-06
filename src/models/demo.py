"""Offline heuristic model — no API key, no network, no cost.

Produces structured, genuinely useful scaffolding (plans, test skeletons,
docstrings, code templates) and is explicit that it is *not* a substitute
for a full cloud model. Deterministic by design so the app is fully
exercisable offline and in tests.
"""
from __future__ import annotations

import re

_DEMO_NOTE = (
    "> **Offline demo brain** — no API key is configured, so this reply was generated "
    "locally by heuristic templates. Add a `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / "
    "`GOOGLE_API_KEY` (or point Ollama at a pulled model) in **Settings** for full intelligence.\n\n"
)


def _extract_function_name(prompt: str) -> str:
    match = re.search(r"(?:function|def|method)\s+`?([A-Za-z_][A-Za-z0-9_]*)`?", prompt, re.I)
    if match:
        return match.group(1)
    match = re.search(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
    if match:
        return match.group(1)
    return "target_function"


def _extract_def_block(prompt: str) -> re.Match | None:
    return re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", prompt)


def _detect_language(prompt: str) -> str:
    p = prompt.lower()
    for lang, markers in (("python", ("python", ".py", "pytest", "def ")),
                           ("typescript", ("typescript", ".ts", "interface ")),
                           ("javascript", ("javascript", ".js", "node", "react")),
                           ("go", ("golang", ".go", "func ")),
                           ("rust", ("rust", ".rs", "fn ")),
                           ("java", ("java", ".java"))):
        if any(m in p for m in markers):
            return lang
    return "python"


def _intent(prompt: str) -> str:
    p = prompt.lower()
    first_line = p.split("\n", 1)[0]  # instructions first; source code must not leak into intent
    if "output only the completion" in first_line or "complete the code at the cursor" in first_line:
        return "complete"
    if any(w in p for w in ("pytest", "unit test", "test case", "write tests", "generate test")):
        return "tests"
    if "docstring" in p or "doc string" in p or "documentation string" in p:
        return "docstring"
    if any(w in p for w in ("explain", "what does", "how does", "walk me through")):
        return "explain"
    if any(w in p for w in ("fix", "bug", "error", "traceback", "exception", "not working")):
        return "debug"
    if any(w in p for w in ("refactor", "clean", "simplify", "improve")):
        return "refactor"
    if any(w in p for w in ("review", "audit", "code review")):
        return "review"
    if any(w in p for w in ("optimize", "performance", "faster", "slow")):
        return "optimize"
    if any(w in p for w in ("implement", "create", "build", "write", "add", "generate")):
        return "codegen"
    return "general"


def _tests_reply(prompt: str, quiet: bool = False) -> str:
    note = "" if quiet else _DEMO_NOTE
    func = _extract_function_name(prompt)
    if quiet:
        # Quiet mode: emit ONLY runnable test code (used by the Generate-tests button)
        return f"""import pytest

# TODO: point this import at the module under test, e.g.
# from src.mymodule import {func}


class Test_{func}:
    def test_{func}_happy_path(self):
        pytest.skip("Fill in arguments and assertions for {func}()")

    @pytest.mark.parametrize("value,expected", [(1, 1), (0, 0)])
    def test_{func}_parametrized(self, value, expected):
        assert value == expected  # TODO: real expectations
"""

    return f"""{note}Here is a solid pytest starting point for `{func}`:

```python
import pytest

# TODO: point this import at the module under test, e.g.
# from src.mymodule import {func}


class Test_{func}:
    def test_{func}_happy_path(self):
        result = {func}()  # TODO: pass realistic arguments
        assert result is not None

    def test_{func}_edge_cases(self):
        with pytest.raises(Exception):
            {func}()  # TODO: invalid input should raise

    @pytest.mark.parametrize("value,expected", [(1, 1), (0, 0)])
    def test_{func}_parametrized(self, value, expected):
        assert {func}(value) == expected  # TODO: real expectations
```

**Next steps**
1. Fix the import to the module under test.
2. Replace the placeholder assertions with the real contract of `{func}`.
3. Add fixtures in `conftest.py` for shared setup.
4. Run with `pytest -q` and iterate."""


def _docstring_reply(prompt: str, quiet: bool = False) -> str:
    match = _extract_def_block(prompt)
    if match:
        name, params_raw = match.group(1), match.group(2)
        params = [p.split(":")[0].split("=")[0].strip()
                  for p in params_raw.split(",")
                  if p.strip() and p.strip() not in ("self", "cls", "*", "**")]
        args_block = "\n".join(f"    {p}: Description of {p}." for p in params) or "    None"
        if quiet:
            # Quiet mode: emit ONLY the docstring body (used by Add-docstring button)
            return (f"One-line summary of what {name} does.\n\n"
                    f"Longer description: the why, inline invariants, side effects.\n\n"
                    f"Args:\n{args_block}\n\n"
                    f"Returns:\n    Description of the return value.\n\n"
                    f"Raises:\n    ValueError: When preconditions are violated (if applicable).")
        indented_args = "\n".join(f"        {p}: Description of {p}." for p in params) or "        None"
        return f"""{_DEMO_NOTE}Suggested Google-style docstring for `{name}`:

```python
\"\"\"One-line summary of what {name} does.

Longer description: the why, the important invariants, and any
side effects / exceptions worth documenting.

    Args:
{indented_args}

    Returns:
        Description of the return value.

    Raises:
        ValueError: When preconditions are violated (if applicable).
\"\"\"
```

Paste it directly under the `def {name}(...)` line (mind indentation)."""
    return f"""{_DEMO_NOTE}Paste the function below / select it in the file view and I'll be more precise. A solid docstring template:

```python
\"\"\"One-line summary.

    Args:
        param: Description.

    Returns:
        Description of the return value.
\"\"\"
```"""


def _explain_reply(prompt: str) -> str:
    return f"""{_DEMO_NOTE}**How to read this code**

Since I'm running offline I can't reason about the full codebase semantics, but here is a reliable review workflow:

1. **Entry points** — start from `main`/`__main__`/route handlers and trace inward.
2. **Data flow** — follow the primary argument/return path; note every mutation and I/O boundary.
3. **State & side effects** — globals, caches, DB writes, file I/O: each is a risk hotspot.
4. **Error paths** — every `except`/`try` is a place behavior diverges; check for swallowed errors.
5. **Tests** — map each test file to the module it covers; gaps = untested behavior.

For a line-level explanation of a specific function, open it in **Context**, click the exact line to set the cursor, and ask again with the open file included."""


def _debug_reply(prompt: str) -> str:
    return f"""{_DEMO_NOTE}**Debugging playbook (offline heuristics)**

1. **Read the traceback bottom-up** — the last frame is where it blew up; the frames above are how you got there.
2. **Reproduce minimally** — a 10-line snippet that triggers the bug beats an hour of guessing.
3. **Bisect** — comment out half the suspect code; the bug is in the half that still fails.
4. **Check the usual suspects** — `None` where you expected a value, mutable default args, wrong indent/await, stale state, off-by-one.
5. **Print types, not just values** — `type(x)` catches 80% of "impossible" bugs.
6. **Verify the fix** — run the failing case, then the whole suite: `python -m pytest -q`.

Capture the actual traceback with full context — ask me again with the file open and the cursor on the failing line for a targeted answer."""


def _refactor_reply(prompt: str) -> str:
    return f"""{_DEMO_NOTE}**Refactor checklist (offline heuristics)**

1. **Name things first** — renaming to intention-revealing names is the cheapest, safest structural change.
2. **Extract until each function does one thing** — target ≤ 40 lines, single responsibility.
3. **Kill duplication** — third copy of a pattern is when you extract it, not the second.
4. **Make illegal states unrepresentable** — types/enums/frozen dataclasses over loose dicts.
5. **Keep the public API stable** — refactor internals, run the suite after every step.
6. **Small commits** — one refactoring = one commit = one green test run."""


def _review_reply(prompt: str) -> str:
    return f"""{_DEMO_NOTE}**Review rubric (offline heuristics)**

- **Correctness** — does it handle empty/None/huge/malformed inputs? Boundary conditions?
- **Security** — injection (SQL/shell/path traversal), secrets in code, unsafe deserialization.
- **Error handling** — no bare `except:`, no swallowed exceptions, meaningful messages.
- **Performance** — accidental O(n²), N+1 queries, unbounded memory, sync I/O in hot paths.
- **Tests** — is the new behavior covered? Are tests deterministic (no real clocks/networks)?
- **Style** — consistent with the codebase; names say why, not how."""


def _optimize_reply(prompt: str) -> str:
    return f"""{_DEMO_NOTE}**Performance checklist (offline heuristics)**

1. **Measure first** — `python -m cProfile -s cumtime your_entry.py` or `timeit`; never optimize blind.
2. **Algorithmic wins dominate** — O(n log n) beats micro-tuning of O(n²).
3. **Cache expensive pure computations** — `functools.lru_cache` is one line.
4. **I/O is usually the bottleneck** — batch queries, avoid N+1, stream instead of slurping files.
5. **Data structures** — `set`/`dict` for membership, generators for pipelines, `__slots__` for hot classes.
6. **Re-measure** — keep the benchmark in `benchmarks/` so regressions are visible."""


def _codegen_reply(prompt: str, language: str) -> str:
    func = _extract_function_name(prompt)
    if language == "python":
        code = f"""def {func}(items: list) -> list:
    \"\"\"TODO: describe the contract of {func}.

    Args:
        items: Input collection.

    Returns:
        Processed list.
    \"\"\"
    result = []
    for item in items:
        # TODO: implement the actual transformation
        result.append(item)
    return result
"""
    elif language in ("typescript", "javascript"):
        code = f"""export function {func}(items: any[]): any[] {{
  // TODO: implement the actual transformation
  return items.map((item) => {{
    return item;
  }});
}}
"""
    else:
        code = f"// TODO: implement {func}\n"
    return f"""{_DEMO_NOTE}Here's a scaffold in **{language}** to start from:

```{language}
{code}```

**To finish it you need:** the exact input/output contract, edge cases to reject, and whether it must be pure or may do I/O. Fill those in — or connect a real model in Settings and I'll implement it completely."""


def _general_reply(prompt: str) -> str:
    topics = re.findall(r"[A-Za-z][A-Za-z0-9_+.-]{3,}", prompt)[:6]
    topic_line = ", ".join(dict.fromkeys(topics)) or "your question"
    return f"""{_DEMO_NOTE}Here's how I'd approach **{topic_line}**:

1. **Pin down the outcome** — one sentence: what does "done" look like, and how will you verify it?
2. **Map the terrain** — use **Context → semantic search** to find related code; open the top hits as tabs.
3. **Work in a loop** — plan → smallest working change → run tests → refine.
4. **Delegate the grunt work** — tests (Button: *Generate tests*), docs (*Add docstring*), and multi-file edits (**Agent** tab) are one click each.
5. **Checkpoint** — commit after each green step.

For full-fidelity answers, wire up an API key or a local Ollama model in **Settings → Models**."""


def _complete_reply(prompt: str) -> str:
    language = _detect_language(prompt)
    if language == "python":
        return "    ...\n"
    return "  // ...\n"


def generate(prompt: str, system_prompt: str | None = None) -> str:
    _only_re = re.compile(r"\bonly\b", re.I)
    quiet = bool((system_prompt and _only_re.search(system_prompt)) or _only_re.search(prompt or ""))
    intent = _intent(prompt)
    language = _detect_language(prompt)
    if intent == "complete":
        return _complete_reply(prompt)
    if intent == "tests":
        return _tests_reply(prompt, quiet)
    if intent == "docstring":
        return _docstring_reply(prompt, quiet)
    if intent == "explain":
        return _explain_reply(prompt)
    if intent == "debug":
        return _debug_reply(prompt)
    if intent == "refactor":
        return _refactor_reply(prompt)
    if intent == "review":
        return _review_reply(prompt)
    if intent == "optimize":
        return _optimize_reply(prompt)
    if intent == "codegen":
        return _codegen_reply(prompt, language)
    return _general_reply(prompt)

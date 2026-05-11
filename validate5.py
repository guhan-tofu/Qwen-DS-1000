"""
validate_samples.py
--------------------
Validates generated data science coding samples stored in a .jsonl file
where each line has the structure:
    {"question": "...", "answer": "..."}

Handles all supported frameworks:
  pandas, numpy         — standard result variable validation
  matplotlib            — no result variable; execution success is sufficient
  pytorch, tensorflow   — modules may not be installed; skipped gracefully
  sklearn, scipy        — load_data() / load_*() stubs with realistic DataFrames

Automatic repairs applied before execution:
  1. Optional modules (matplotlib, torch, tensorflow, seaborn) → SKIP if absent
  2. Missing </code> before BEGIN SOLUTION → closing tag inserted
  3. Missing <code> after BEGIN SOLUTION → opening tag inserted
  4. Leading whitespace stripped line-by-line from setup code blocks
  5. 'ANSWER:' label stripped from the start of the answer field
  6. load_data() and variant load_*() stubs returning DataFrames shaped to
     match what the setup code expects (columns, dtypes, n_rows)

Modes:
  (default)       Print results to console only, write no output files.
  --mode review   Write passing samples to <stem>_valid.jsonl and failing
                  samples (with an added 'failure_reason' field) to
                  <stem>_failed.jsonl for manual inspection and correction.

Usage:
    python validate_samples.py <path_to_jsonl> [--mode review]
"""

import argparse
import json
import os
import re
import textwrap
import traceback


# ── ANSI colours ───────────────────────────────────────────────────────────────
PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
WARN = "\033[93m WARN \033[0m"
SKIP = "\033[94m SKIP \033[0m"

# ── Optional modules ───────────────────────────────────────────────────────────
OPTIONAL_MODULES = {
    "torch", "tensorflow", "tf",
    "matplotlib", "seaborn", "mpl_toolkits"
}


# ── Sanitisation ───────────────────────────────────────────────────────────────

def _fix_indentation(code: str) -> str:
    """
    Fix leading-whitespace issues in setup code without breaking
    legitimate nested indentation (e.g. PyTorch class/method bodies).

    Strategy:
      1. Try textwrap.dedent() — strips a uniform common prefix safely.
      2. If the result still won't compile, fall back to per-line lstrip()
         for the rare case of inconsistently indented top-level lines.
      3. If neither compiles, return the dedented version and let exec()
         surface the real error rather than silently mangling the code.
    """
    stripped = textwrap.dedent(code).strip()
    try:
        compile(stripped, "<setup>", "exec")
        return stripped
    except (IndentationError, SyntaxError):
        pass

    lstripped = "\n".join(line.lstrip() for line in code.splitlines()).strip()
    try:
        compile(lstripped, "<setup>", "exec")
        return lstripped
    except (IndentationError, SyntaxError):
        pass

    # Neither worked — return dedented and let the error be reported naturally
    return stripped


def _inject_pass_into_empty_bodies(code: str) -> str:
    """
    Insert 'pass' after any def/class line whose body contains only
    comments or blank lines (no executable statement).

    This is needed for PyTorch samples where the setup defines a class
    with placeholder methods like:

        def __getitem__(self, idx):
            # IMPLEMENTATION HERE

    Python requires at least one real statement in a function body.
    Inserting 'pass' lets the setup compile so the answer can be
    appended and exec'd as the intended implementation.
    """
    lines = code.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        result.append(line)
        stripped = line.rstrip()
        # Detect a def/class opener
        if re.match(r"(\s*)(def |class )\S", stripped) and stripped.endswith(":"):
            indent = len(line) - len(line.lstrip())
            # Scan ahead for the first non-empty, non-comment line
            j = i + 1
            has_real_body = False
            while j < len(lines):
                next_stripped = lines[j].strip()
                if not next_stripped or next_stripped.startswith("#"):
                    j += 1
                    continue
                # Real line found — check if it's indented inside this block
                next_indent = len(lines[j]) - len(lines[j].lstrip())
                if next_indent > indent:
                    has_real_body = True
                break
            if not has_real_body:
                result.append(" " * (indent + 4) + "pass")
        i += 1
    return "\n".join(result)


def sanitise_question(question: str) -> str:
    """
    Fix common model-generation quirks in the question field:
    1. Missing </code> before BEGIN SOLUTION — insert it.
    2. Missing <code> after BEGIN SOLUTION — insert it.
    3. Indentation issues in setup <code> blocks — smart dedent with fallback.
    4. Empty def/class bodies in setup — insert 'pass' so code compiles.
    """
    # Fix 1: missing </code> before BEGIN SOLUTION
    if "BEGIN SOLUTION" in question:
        pre, post = question.split("BEGIN SOLUTION", 1)
        if "<code>" in pre and "</code>" not in pre:
            pre = pre.rstrip() + "\n</code>\n"
        question = pre + "BEGIN SOLUTION" + post

    # Fix 2: missing <code> after BEGIN SOLUTION
    question = re.sub(
        r"(BEGIN SOLUTION\s*)(?!<code>)",
        r"\1<code>",
        question
    )

    # Fix 3 + 4: smart indentation fix + pass injection on setup block
    if "BEGIN SOLUTION" in question:
        pre, post = question.split("BEGIN SOLUTION", 1)

        def _fix_setup_block(match):
            tag_open  = match.group(1)
            body      = match.group(2)
            tag_close = match.group(3)
            fixed = _fix_indentation(body)
            fixed = _inject_pass_into_empty_bodies(fixed)
            return f"{tag_open}\n{fixed}\n{tag_close}"

        pre = re.sub(r"(<code>)(.*?)(</code>)", _fix_setup_block, pre, flags=re.DOTALL)
        question = pre + "BEGIN SOLUTION" + post

    return question


def sanitise_answer(answer: str) -> str:
    """Strip a leading 'ANSWER:' label the model sometimes emits."""
    return re.sub(r'^\s*ANSWER:\s*', '', answer.strip())


# ── Parsing ────────────────────────────────────────────────────────────────────

def extract_setup_code(question: str) -> str | None:
    """Pull the setup <code> block from before BEGIN SOLUTION."""
    pre_solution = question.split("BEGIN SOLUTION")[0]
    match = re.search(r"<code>(.*?)</code>", pre_solution, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def extract_result_variable(question: str) -> str | None:
    """
    Detect the result variable LHS from the placeholder line.
    Returns None if no placeholder found (side-effect-only problems).
    """
    match = re.search(r"^([\w\s,\[\]]+?)\s*=\s*\.{3}", question, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def extract_problem_title(question: str) -> str:
    """Pull the first non-empty line after 'Problem:' as a short display title."""
    match = re.search(r"Problem:\s*\n(.+)", question)
    if match:
        return match.group(1).strip()[:80]
    return "(no title found)"


def parse_result_variable_names(lhs: str) -> list[str]:
    """Split 'x_train, x_test, ...' into individual variable names."""
    return [v.strip() for v in lhs.split(",") if v.strip()]


# ── load_data() stub ───────────────────────────────────────────────────────────

# ── Shared stub infrastructure ─────────────────────────────────────────────────
# N=30 gives balanced classes with enough members for n_splits=5 CV.
_STUB_BASE = textwrap.dedent("""\
    import numpy as _np_stub
    import pandas as _pd_stub
    _N = 30

    # Mixed DataFrame (numerical + categorical + text + target)
    _STUB_MIXED = _pd_stub.DataFrame({
        'age':      _np_stub.random.randint(20, 60, _N).astype(float),
        'income':   _np_stub.random.randint(30000, 90000, _N).astype(float),
        'expenses': _np_stub.random.randint(10000, 50000, _N).astype(float),
        'num1':     _np_stub.random.rand(_N),
        'num2':     _np_stub.random.rand(_N),
        'gender':   _np_stub.random.choice(['M', 'F'], _N),
        'city':     _np_stub.random.choice(['NY', 'LA', 'SF'], _N),
        'cat1':     _np_stub.random.choice(['a', 'b', 'c'], _N),
        'cat2':     _np_stub.random.choice(['x', 'y'], _N),
        'text':     ['sample document ' + str(i) for i in range(_N)],
        'label':    _np_stub.random.randint(0, 2, _N),
        'target':   _np_stub.tile([0, 1], _N // 2),
    })
    # Numeric-only DataFrame (no categoricals — safe for StandardScaler etc.)
    _STUB_NUM = _pd_stub.DataFrame({
        'age':      _np_stub.random.randint(20, 60, _N).astype(float),
        'income':   _np_stub.random.randint(30000, 90000, _N).astype(float),
        'expenses': _np_stub.random.randint(10000, 50000, _N).astype(float),
        'num1':     _np_stub.random.rand(_N),
        'num2':     _np_stub.random.rand(_N),
        'target':   _np_stub.tile([0, 1], _N // 2),
    })
    _STUB_X_NUM = _STUB_NUM.drop(columns=['target'])
    _STUB_X_MIX = _STUB_MIXED.drop(columns=['target', 'text', 'label'])
    _STUB_y     = _STUB_NUM['target'].values

    def load_multilabel_data(*args, **kwargs):
        _ym = _np_stub.random.randint(0, 2, (_N, 3))
        return _STUB_X_NUM.values.copy(), _ym, _STUB_X_NUM.values.copy()

    def load_high_dim_data(*args, **kwargs):
        return _np_stub.random.rand(_N, 50), _STUB_y.copy()
""")


def _classify_stub(setup: str) -> str:
    """
    Inspect the setup code and return one of:
      'text_df'       — needs a DataFrame with 'text' and 'label' columns
      'dict_df'       — data is accessed as data['X_train'] etc.
      'mixed_df'      — needs both numeric and categorical columns
      'numeric_df'    — needs numeric-only DataFrame (drop target, no OHE)
      'xy_numeric'    — X, y = load_data() where X should be numeric
      'numeric_array' — X = load_data() where X is a 2-D numpy array
    """
    if re.search(r"TfidfVectorizer|CountVectorizer|data\[.text.\]", setup):
        return "text_df"
    if re.search(r"data\[.X_train.\]", setup):
        return "dict_df"
    if re.search(r"select_dtypes", setup):
        return "mixed_df"
    if re.search(r'(?:X|X_train)\s*,\s*(?:y|y_train)\s*=\s*load_data\(\)', setup):
        return "xy_numeric"
    if re.search(r'\bX\s*=\s*load_data\(\)', setup):
        return "numeric_array"
    if re.search(r'drop.*target|target.*drop', setup) and \
       not re.search(r'OneHotEncoder|categorical', setup):
        return "numeric_df"
    # Default: mixed so ColumnTransformer with both types works
    return "mixed_df"


def build_preamble(setup: str) -> str:
    """Return the type-aware preamble to prepend before the setup code."""
    if not re.search(r'\bload_\w+\s*\(', setup):
        return ""

    stub_type = _classify_stub(setup)

    overrides = {
        "text_df": textwrap.dedent("""\
            def load_data(*a, **k):
                return _STUB_MIXED[['text', 'label']].copy()
        """),
        "dict_df": textwrap.dedent("""\
            def load_data(*a, **k):
                return _pd_stub.DataFrame({
                    'X_train': _STUB_X_NUM['age'].values,
                    'y_train': _STUB_y,
                })
        """),
        "mixed_df": textwrap.dedent("""\
            def load_data(*a, **k):
                return _STUB_MIXED.copy()
        """),
        "numeric_df": textwrap.dedent("""\
            def load_data(*a, **k):
                return _STUB_NUM.copy()
        """),
        "xy_numeric": textwrap.dedent("""\
            def load_data(*a, **k):
                return _STUB_X_NUM.copy(), _STUB_y.copy()
        """),
        "numeric_array": textwrap.dedent("""\
            def load_data(*a, **k):
                return _STUB_X_NUM.values.copy()
        """),
    }

    return _STUB_BASE + "\n" + overrides[stub_type]


# ── Optional-module detection ──────────────────────────────────────────────────

def detect_optional_imports(setup: str, answer: str) -> set[str]:
    """Return optional module names imported anywhere in setup or answer."""
    combined = setup + "\n" + answer
    found = set()
    for mod in OPTIONAL_MODULES:
        if re.search(rf"\bimport\s+{mod}\b|\bfrom\s+{mod}\b", combined):
            found.add(mod)
    return found


def is_module_available(module_name: str) -> bool:
    import importlib.util
    check_name = "tensorflow" if module_name == "tf" else module_name
    return importlib.util.find_spec(check_name) is not None


# ── Structure checks ───────────────────────────────────────────────────────────

def check_question_structure(question: str) -> list[str]:
    """Lightweight structural checks — run AFTER sanitisation."""
    warnings = []
    if "BEGIN SOLUTION" not in question:
        warnings.append("question does not contain 'BEGIN SOLUTION'")
    if not question.rstrip().endswith("<code>"):
        warnings.append("question does not end with '<code>' after sanitisation "
                        "(solution may have leaked in)")
    if "<code>" not in question.split("BEGIN SOLUTION")[0]:
        warnings.append("no setup <code> block found before BEGIN SOLUTION")
    return warnings


# ── Execution ──────────────────────────────────────────────────────────────────

def run_sample(setup: str, answer: str) -> tuple[bool, str, dict]:
    """Execute preamble + setup + answer in a shared namespace."""
    namespace = {}
    preamble  = build_preamble(setup)
    combined  = preamble + "\n" + setup + "\n" + answer
    try:
        exec(combined, namespace)
    except Exception:
        return False, traceback.format_exc(), namespace
    return True, "", namespace


# ── Reporting ──────────────────────────────────────────────────────────────────

def summarise_value(value) -> str:
    try:
        import pandas as pd
        if isinstance(value, pd.DataFrame):
            return f"DataFrame  shape={value.shape}  columns={list(value.columns)}"
        if isinstance(value, pd.Series):
            return f"Series  len={len(value)}  dtype={value.dtype}  name={value.name}"
        if isinstance(value, pd.Index):
            return f"Index  len={len(value)}  dtype={value.dtype}"
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return f"ndarray  shape={value.shape}  dtype={value.dtype}"
    except ImportError:
        pass
    return f"{type(value).__name__}  value={repr(value)[:120]}"


def check_result_variables(var_names: list[str], namespace: dict) -> tuple[bool, str]:
    lines  = []
    all_ok = True
    for name in var_names:
        sentinel = object()
        val = namespace.get(name, sentinel)
        if val is sentinel:
            lines.append(f"  '{name}' was NOT assigned")
            all_ok = False
        else:
            lines.append(f"  '{name}': {summarise_value(val)}")
    return all_ok, "\n".join(lines)


# ── Output paths ───────────────────────────────────────────────────────────────

def derive_output_paths(input_path: str) -> tuple[str, str]:
    stem, ext = os.path.splitext(input_path)
    ext = ext or ".jsonl"
    return f"{stem}_valid{ext}", f"{stem}_failed{ext}"


# ── Main ───────────────────────────────────────────────────────────────────────

def validate_file(path: str, mode: str) -> None:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    passed = failed = skipped = struct_warned = 0

    valid_fh = failed_fh = None
    if mode == "review":
        valid_path, failed_path = derive_output_paths(path)
        valid_fh  = open(valid_path,  "w", encoding="utf-8")
        failed_fh = open(failed_path, "w", encoding="utf-8")

    print(f"\nValidating {len(lines)} sample(s) from '{path}'\n{'─' * 65}")

    for i, line in enumerate(lines, start=1):
        d        = json.loads(line)
        question = sanitise_question(d.get("question", ""))
        answer   = sanitise_answer(d.get("answer", ""))
        title    = extract_problem_title(question)

        print(f"\nSample {i}: {title}")

        failure_reason = None
        was_skipped    = False

        # 1. Field presence
        if not question or not answer:
            missing = []
            if not question: missing.append("'question'")
            if not answer:   missing.append("'answer'")
            msg = f"Missing fields: {', '.join(missing)}"
            print(f"[{FAIL}] {msg}")
            failure_reason = msg
            failed += 1

        # 2. Structure checks (after sanitisation)
        if failure_reason is None:
            struct_warnings = check_question_structure(question)
            if struct_warnings:
                struct_warned += 1
                for w in struct_warnings:
                    print(f"[{WARN}] Structure: {w}")

        # 3. Parse setup code
        if failure_reason is None:
            setup = extract_setup_code(question)
            if setup is None:
                msg = "Could not parse setup <code> block from question."
                print(f"[{FAIL}] {msg}")
                failure_reason = msg
                failed += 1

        # 4. Check for optional modules — skip if not installed
        if failure_reason is None:
            missing_mods = {
                m for m in detect_optional_imports(setup, answer)
                if not is_module_available(m)
            }
            if missing_mods:
                print(f"[{SKIP}] Optional module(s) not installed: "
                      f"{', '.join(sorted(missing_mods))} — skipping execution.")
                skipped    += 1
                was_skipped = True

        # 5. Execute setup + answer
        if failure_reason is None and not was_skipped:
            success, error_msg, namespace = run_sample(setup, answer)
            if not success:
                print(f"[{FAIL}] Execution error:")
                for err_line in error_msg.strip().splitlines():
                    print(f"         {err_line}")
                failure_reason = error_msg.strip()
                failed += 1

        # 6. Result variable check
        if failure_reason is None and not was_skipped:
            lhs = extract_result_variable(question)
            if lhs is None:
                print(f"[{PASS}] (side-effect only — no result variable expected)")
                passed += 1
            else:
                var_names = parse_result_variable_names(lhs)
                all_ok, summary = check_result_variables(var_names, namespace)
                if all_ok:
                    print(f"[{PASS}]")
                    print(summary)
                    passed += 1
                else:
                    msg = f"Result variable(s) not assigned:\n{summary}"
                    print(f"[{FAIL}] {msg}")
                    failure_reason = msg
                    failed += 1

        # 7. Write to output files (review mode)
        if mode == "review":
            if failure_reason is None:
                valid_fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            elif not was_skipped:
                d_out = {**d, "failure_reason": failure_reason}
                failed_fh.write(json.dumps(d_out, ensure_ascii=False) + "\n")

    if mode == "review":
        valid_fh.close()
        failed_fh.close()

    print(f"\n{'─' * 65}")
    print(f"Results : {passed} passed, {failed} failed, {skipped} skipped  "
          f"({struct_warned} structural warning(s))  "
          f"out of {len(lines)} total.")
    if mode == "review":
        valid_path, failed_path = derive_output_paths(path)
        print(f"\nValid samples  → {valid_path}  ({passed} sample(s))")
        print(f"Failed samples → {failed_path}  ({failed} sample(s), "
              f"includes 'failure_reason' field)")
    print("─" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate generated data science coding samples."
    )
    parser.add_argument("path", help="Path to the .jsonl file to validate.")
    parser.add_argument(
        "--mode",
        choices=["review"],
        default=None,
        help=(
            "review: write passing samples to <stem>_valid.jsonl and "
            "failing samples (with 'failure_reason') to <stem>_failed.jsonl."
        )
    )
    args = parser.parse_args()
    validate_file(args.path, mode=args.mode or "")
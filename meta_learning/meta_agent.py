"""
FORTRESS v5 - meta_agent.py
Path: meta_learning/meta_agent.py

Constitutional Meta-Learning Agent.
Analyzes strategy decay, writes novel feature-engineering code,
and orchestrates the 5-layer validation pipeline before deployment.

FIXES APPLIED:
  - BUG #8: `_passes_constitution()` used a secondary LLM prompt to audit AI-generated code.
            An LLM auditing LLM output is not a security gate — generated code can embed
            prompt injection to force the auditor to output "NO" (pass).
            Replaced entirely with deterministic AST-based static analysis that:
              1. Parses the generated code with Python's `ast` module.
              2. Walks every AST node to detect banned imports, attribute access,
                 file operations, and dangerous built-in calls.
              3. Returns False (reject) if ANY violation is found — no LLM involved.

  - BUG #9: ValidationPipeline gates were all hardcoded stubs that always returned True/1.15.
            The 5-layer safety promise was theatre. A generated feature was being deployed
            to shadow with zero actual validation.
            Gates 1-3 now have real implementations:
              Gate 1: Spawns `docker run --network none` to compile the code in isolation.
              Gate 2: Delegates to research/backtest_engine.py via subprocess.
              Gate 3: Calls the World Model's diffusion stress test.
            Gate 4 (shadow deployment) retains its scaffold pending k8s integration.
"""

import os
import ast
import json
import logging
import asyncio
import subprocess
import tempfile
from typing import Dict, List, Any, Set

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError("Requires openai package for code generation.")

logger = logging.getLogger("MetaLearningAgent")

# ── INVIOLABLE CONSTITUTIONAL RULES ──────────────────────────────────────────
# These rules are injected into the LLM prompt for generation guidance.
# Enforcement is done by the AST auditor below — NOT by an LLM.
CONSTITUTION = """
1. NEVER modify max_leverage or any thresholds in config/risk_limits.yaml.
2. NEVER introduce a new data feature without enforcing as_of_date causality.
3. NEVER attempt to bypass the Docker sandbox network restrictions.
4. NEVER optimize a feature to fit a single historical event (e.g., exclusively the COVID crash).
5. NEVER write code that attempts to modify this Constitutional Rules file.
"""

# ── AST AUDITOR BANNED PATTERNS ──────────────────────────────────────────────
# Any generated code containing these patterns will be rejected outright.

# Banned top-level imports and from-imports.
_BANNED_IMPORTS: Set[str] = {
    "os", "sys", "subprocess", "shutil", "pathlib",
    "importlib", "builtins", "ctypes", "socket",
    "urllib", "http", "ftplib", "smtplib",
    "pickle", "shelve", "marshal",
}

# Banned attribute accesses (e.g., os.system, subprocess.run).
_BANNED_ATTR_ACCESS: Set[str] = {
    "system", "popen", "run", "Popen", "call", "check_output",
    "exec", "eval", "compile", "execfile",
    "remove", "rmdir", "unlink", "rename",
    "open",   # file I/O; features must use DB connections only
}

# Banned string literals that indicate targeted config access.
_BANNED_STRING_LITERALS: Set[str] = {
    "risk_limits.yaml",
    "constitutional_rules.py",
    "config/risk_limits",
}

# Banned built-in function calls by name.
_BANNED_BUILTINS: Set[str] = {"exec", "eval", "compile", "__import__"}

# Files that generated code is explicitly forbidden from referencing.
_PROTECTED_FILES: Set[str] = {
    "risk_limits.yaml",
    "constitutional_rules.py",
}


class ASTConstitutionalAuditor:
    """
    FIX #8: Deterministic AST-based constitutional auditor.

    An LLM cannot reliably audit LLM output — generated code can embed prompt
    injection payloads to force a compliant answer. This auditor uses Python's
    `ast` module to statically analyze the parse tree of the generated code,
    checking for banned patterns before any execution occurs.

    All checks are O(n_nodes) and complete in <10ms for any reasonably sized feature.
    """

    def audit(self, code: str) -> tuple[bool, List[str]]:
        """
        Parses and walks the AST of the generated code.

        Returns:
            (passes, violations): True if no violations found.
                                  violations is a list of human-readable descriptions.
        """
        violations: List[str] = []

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            violations.append(f"SyntaxError: {e}")
            return False, violations

        for node in ast.walk(tree):

            # ── Check 1: Banned top-level imports ────────────────────────────
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module in _BANNED_IMPORTS:
                        violations.append(
                            f"Banned import: `import {alias.name}` — "
                            f"module '{root_module}' is prohibited."
                        )

            # ── Check 2: Banned from-imports ─────────────────────────────────
            if isinstance(node, ast.ImportFrom):
                root_module = (node.module or "").split(".")[0]
                if root_module in _BANNED_IMPORTS:
                    violations.append(
                        f"Banned from-import: `from {node.module} import ...` — "
                        f"module '{root_module}' is prohibited."
                    )

            # ── Check 3: Banned attribute accesses (e.g., os.system) ─────────
            if isinstance(node, ast.Attribute):
                if node.attr in _BANNED_ATTR_ACCESS:
                    violations.append(
                        f"Banned attribute access: `.{node.attr}` — "
                        "potential system call or file I/O."
                    )

            # ── Check 4: Banned built-in function calls ───────────────────────
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in _BANNED_BUILTINS:
                        violations.append(
                            f"Banned built-in call: `{node.func.id}(...)` — "
                            "dynamic code execution is not permitted."
                        )

            # ── Check 5: Banned string literals (config file references) ─────
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for banned_str in _BANNED_STRING_LITERALS:
                    if banned_str in node.value:
                        violations.append(
                            f"Banned string literal: '{node.value}' references "
                            f"protected file '{banned_str}'."
                        )

            # ── Check 6: as_of_date causality enforcement ────────────────────
            # Any DB query function call must include 'as_of_date' in its keyword args.
            # This is a heuristic check — it flags the absence of the parameter when
            # a DB connection call is detected.
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in {"fetchval", "fetch", "fetchrow", "execute"}:
                        kwarg_names = {kw.arg for kw in node.keywords}
                        # Check if 'as_of_date' appears either as kwarg or in the query string
                        has_causality = (
                            "as_of_date" in kwarg_names
                            or any(
                                isinstance(arg, ast.Constant)
                                and isinstance(arg.value, str)
                                and "as_of_date" in arg.value
                                for arg in node.args
                            )
                        )
                        if not has_causality:
                            violations.append(
                                f"Look-ahead bias risk: DB query via `.{node.func.attr}()` "
                                "does not reference 'as_of_date'. "
                                "All queries must enforce causal filtering."
                            )

        passes = len(violations) == 0
        return passes, violations


class ValidationPipeline:
    """
    FIX #9: The 5-Layer Security Gate for AI-Generated Code.
    Gates 1-3 now have real implementations. Gate 4 retains its scaffold.
    """

    async def run_sandbox_compilation(self, code: str) -> bool:
        """
        Gate 1: Compile and syntax-check the code in an isolated Docker container
        with no network access (`--network none`).

        Runs: `docker run --network none --rm python:3.11-slim python -c "compile(code)"`
        Returns True only if the container exits with code 0.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="gate1_"
        ) as f:
            # Write a compile-only wrapper — does not execute the code.
            f.write(
                f"import ast\n"
                f"code = {repr(code)}\n"
                f"ast.parse(code)\n"
                f"compile(code, '<generated>', 'exec')\n"
                f"print('COMPILE_OK')\n"
            )
            tmp_path = f.name

        try:
            result = await asyncio.create_subprocess_exec(
                "docker", "run",
                "--network", "none",
                "--rm",
                "--memory", "256m",
                "--cpus", "0.5",
                "-v", f"{tmp_path}:/check.py:ro",
                "python:3.11-slim",
                "python", "/check.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)

            if result.returncode == 0 and b"COMPILE_OK" in stdout:
                logger.info("Gate 1 [Sandbox Compilation]: ✅ PASSED")
                return True
            else:
                logger.error(
                    f"Gate 1 [Sandbox Compilation]: ❌ FAILED\n"
                    f"stderr: {stderr.decode()}"
                )
                return False
        except asyncio.TimeoutError:
            logger.error("Gate 1 [Sandbox Compilation]: ❌ TIMEOUT (>30s)")
            return False
        except FileNotFoundError:
            logger.error(
                "Gate 1 [Sandbox Compilation]: ❌ Docker not available in this environment."
            )
            return False
        finally:
            os.unlink(tmp_path)

    async def run_historical_backtest(self, code: str) -> float:
        """
        Gate 2: Injects the generated feature into the backtest engine and
        returns the Deflated Sharpe Ratio (DSR).

        Writes the generated feature to a temp file, then invokes
        research/backtest_engine.py with --feature-path pointing to it.
        Parses the DSR from the subprocess stdout.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="gate2_feature_"
        ) as f:
            f.write(code)
            feature_path = f.name

        try:
            result = await asyncio.create_subprocess_exec(
                "python", "research/backtest_engine.py",
                "--feature-path", feature_path,
                "--output-format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=600)

            if result.returncode != 0:
                logger.error(
                    f"Gate 2 [Backtest]: ❌ Subprocess failed.\n"
                    f"stderr: {stderr.decode()[:500]}"
                )
                return 0.0

            # Expect the backtest engine to print a JSON line: {"dsr": 1.23, ...}
            output = stdout.decode()
            for line in output.splitlines():
                try:
                    data = json.loads(line)
                    dsr = float(data.get("dsr", 0.0))
                    logger.info(f"Gate 2 [Backtest DSR]: {dsr:.4f}")
                    return dsr
                except (json.JSONDecodeError, ValueError):
                    continue

            logger.error("Gate 2 [Backtest]: ❌ Could not parse DSR from output.")
            return 0.0

        except asyncio.TimeoutError:
            logger.error("Gate 2 [Backtest]: ❌ TIMEOUT (>600s)")
            return 0.0
        finally:
            os.unlink(feature_path)

    async def run_diffusion_stress_test(self, code: str) -> bool:
        """
        Gate 3: Tests the new feature against 2,000 synthetic crash scenarios
        generated by the Neural SDE World Model.

        Delegates to training/diffusion/generate_scenarios.py which loads the
        trained world model, generates adversarial paths, and evaluates feature
        stability. Returns True only if the feature passes the CVaR stability check.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, prefix="gate3_feature_"
        ) as f:
            f.write(code)
            feature_path = f.name

        try:
            result = await asyncio.create_subprocess_exec(
                "python", "training/diffusion/generate_scenarios.py",
                "--feature-path", feature_path,
                "--n-paths", "2000",
                "--output-format", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=300)

            if result.returncode != 0:
                logger.error(
                    f"Gate 3 [Stress Test]: ❌ Subprocess failed.\n"
                    f"stderr: {stderr.decode()[:500]}"
                )
                return False

            output = stdout.decode()
            for line in output.splitlines():
                try:
                    data = json.loads(line)
                    passed = bool(data.get("stress_test_passed", False))
                    cvar = data.get("cvar_95", "N/A")
                    if passed:
                        logger.info(f"Gate 3 [Stress Test]: ✅ PASSED (CVaR-95: {cvar})")
                    else:
                        logger.warning(
                            f"Gate 3 [Stress Test]: ❌ FAILED (CVaR-95: {cvar})"
                        )
                    return passed
                except (json.JSONDecodeError, ValueError):
                    continue

            logger.error("Gate 3 [Stress Test]: ❌ Could not parse result from output.")
            return False

        except asyncio.TimeoutError:
            logger.error("Gate 3 [Stress Test]: ❌ TIMEOUT (>300s)")
            return False
        finally:
            os.unlink(feature_path)

    async def deploy_to_shadow(self, code: str) -> bool:
        """
        Gate 4: Deploys to paper trading environment for 7 days.
        Scaffold: Pending Kubernetes shadow-deployment integration.
        """
        logger.info("Gate 4 [Shadow Deployment]: Initiated (scaffold — pending k8s integration)")
        return True


class ConstitutionalMetaAgent:
    def __init__(self, config: Dict[str, Any]):
        self.llm_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.validator = ValidationPipeline()
        # FIX #8: The AST auditor replaces the secondary LLM constitutional check.
        self.auditor = ASTConstitutionalAuditor()
        self.dsr_threshold = config.get("min_deflated_sharpe", 1.0)

    async def analyze_and_evolve(
        self, performance_report: str, recent_logs: str
    ) -> None:
        """
        Main loop. Called weekly or when the Bayesian Health Monitor detects alpha decay.
        """
        logger.info("Initiating Meta-Learning Evolution Cycle...")

        # 1. Hypothesize: Ask the LLM to write a new Python feature.
        new_code = await self._generate_new_feature_code(performance_report, recent_logs)
        if not new_code:
            logger.error("Failed to generate viable code. Aborting evolution cycle.")
            return

        # 2. FIX #8: Constitutional Check via deterministic AST analysis.
        #    This replaces the secondary LLM auditor which was trivially bypassable
        #    via prompt injection in the generated code.
        passes, violations = self.auditor.audit(new_code)
        if not passes:
            logger.warning(
                f"Generated code violated the Constitution ({len(violations)} violation(s)):"
            )
            for v in violations:
                logger.warning(f"  ❌ {v}")
            logger.warning("Aborting evolution cycle.")
            return

        logger.info(
            f"✅ AST Constitutional Audit passed ({len(violations)} violations found: 0). "
            "Entering Validation Pipeline."
        )

        # 3. Gate 1: Sandbox Compilation.
        if not await self.validator.run_sandbox_compilation(new_code):
            return

        # 4. Gate 2: Historical Backtest DSR.
        dsr = await self.validator.run_historical_backtest(new_code)
        if dsr < self.dsr_threshold:
            logger.warning(
                f"Feature rejected: DSR {dsr:.4f} is below threshold {self.dsr_threshold}. "
                "Alpha decay prevention engaged."
            )
            return

        # 5. Gate 3: Diffusion Stress Test (2,000 synthetic crash paths).
        if not await self.validator.run_diffusion_stress_test(new_code):
            return

        # 6. Gate 4: Shadow Deployment.
        await self.validator.deploy_to_shadow(new_code)

        # 7. Log the evolution event for human audit.
        self._log_evolution_event(new_code, dsr)

    async def _generate_new_feature_code(
        self, performance_report: str, recent_logs: str
    ) -> str:
        """
        Prompts the LLM to act as a quant researcher and write a new PyTorch/Pandas feature.
        Temperature is intentionally low (0.2) to bias toward correct, structured code
        rather than creative hallucinations.
        """
        prompt = (
            f"You are the FORTRESS v5 Meta-Learning Agent. Your objective is to engineer "
            f"a new quantitative feature to improve the Alpha Engine.\n\n"
            f"CONSTITUTION (MANDATORY CONSTRAINTS):\n{CONSTITUTION}\n\n"
            f"RECENT PERFORMANCE DECAY:\n{performance_report}\n\n"
            f"Write a Python class inheriting from `BaseFeature` that calculates a novel "
            f"indicator using pandas or PyTorch. You MUST include an `as_of_date` parameter "
            f"in any database queries to prevent look-ahead bias. Return ONLY Python code."
        )

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.2,
            )
            code = response.choices[0].message.content
            # Strip markdown fences if present.
            code = code.replace("```python\n", "").replace("```", "").strip()
            return code
        except Exception as e:
            logger.error(f"LLM Code Generation Error: {e}")
            return ""

    def _log_evolution_event(self, code: str, dsr: float) -> None:
        """Records the successful generation to the evolution log for human review."""
        logger.info(
            f"Evolution event logged. DSR: {dsr:.4f}. "
            "Feature queued for 7-day shadow deployment monitoring."
        )
"""
FORTRESS v5 - constitutional_rules.py
Path: meta_learning/constitutional_rules.py

READ-ONLY file defining the absolute boundaries of the Meta-Learning Agent.
"""

CONSTITUTION = """
1. NEVER modify max_leverage or any thresholds in config/risk_limits.yaml.
2. NEVER introduce a new data feature without enforcing as_of_date causality to prevent look-ahead bias.
3. NEVER attempt to bypass the Docker sandbox network restrictions.
4. NEVER optimize a feature to fit a single historical event (e.g., exclusively the 2020 crash).
5. NEVER write code that attempts to modify this Constitutional Rules file.
"""
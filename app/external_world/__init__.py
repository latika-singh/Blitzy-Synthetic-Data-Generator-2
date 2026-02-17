"""
External World Simulation package (F-005).

Simulates customer, vendor, bank, and carrier entities interacting with the
ERP system.  Entity pools are managed with a tiered distribution:

- **Strategic** (10 %): High interaction frequency, excellent/good profiles
- **Standard** (30 %): Medium interaction frequency, good/average profiles
- **Transactional** (60 %): Low interaction frequency, average/poor/problem profiles

Each entity carries a configurable :class:`BehaviorProfile` that controls
payment behaviour, order patterns, and invoice characteristics.

Primary components
------------------
BehaviorProfile / BehaviorProfileFactory / TierDistribution
    Multi-dimensional entity behaviour configuration (payment, order, invoice).
ExternalWorldManager
    Central coordinator: tiered entity pools, profile assignment, interaction
    generation.
CustomerSimulator
    Order generation, payment simulation, dispute handling.
VendorSimulator
    Invoice generation, goods delivery, inquiry response.
BankSimulator
    Bank statement generation, payment processing simulation.
"""

from __future__ import annotations

# ── Always-available imports (foundational data module, no internal deps) ──
from app.external_world.behavior_profiles import (
    BehaviorProfile,
    BehaviorProfileFactory,
    InvoiceBehavior,
    OrderBehavior,
    PaymentBehavior,
    TierDistribution,
)

# ── Lazy imports for sibling modules created by other agents ──────────────
# Uses try/except so the package remains importable even when sibling
# modules have not yet been created by their respective agents.

try:
    from app.external_world.external_entity_manager import ExternalWorldManager
except ImportError:  # pragma: no cover
    pass

try:
    from app.external_world.customer_simulator import CustomerSimulator
except ImportError:  # pragma: no cover
    pass

try:
    from app.external_world.vendor_simulator import VendorSimulator
except ImportError:  # pragma: no cover
    pass

try:
    from app.external_world.bank_simulator import BankSimulator
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "BankSimulator",
    "BehaviorProfile",
    "BehaviorProfileFactory",
    "CustomerSimulator",
    "ExternalWorldManager",
    "InvoiceBehavior",
    "OrderBehavior",
    "PaymentBehavior",
    "TierDistribution",
    "VendorSimulator",
]

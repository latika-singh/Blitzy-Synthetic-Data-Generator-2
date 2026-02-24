"""
External World Simulation package (F-005) — Project 2: Agent & Orchestration Engine.

This package simulates external entities — customers, vendors, banks, and carriers —
that interact with the synthetic ERP system.  It is one of seven interconnected
subsystems in the Agent & Orchestration Engine and provides the external stimulus
layer that drives agent work-item generation.

Entity Pool Management
----------------------
Entities are organized into **tiered pools** with the following default distribution
(configurable via :class:`TierDistribution`):

- **Strategic** (10 %): High-value entities with excellent/good behavior profiles
  and frequent interactions.
- **Standard** (30 %): Mid-tier entities with good/average profiles and moderate
  interaction frequency.
- **Transactional** (60 %): High-volume, lower-value entities with average/poor/
  problem profiles and less predictable behavior.

Behavior Profiles
-----------------
Every entity is assigned a :class:`BehaviorProfile` that controls three behavioral
dimensions:

- :class:`PaymentBehavior` — payment timing segment, day-variance, short-pay rate,
  and dispute rate.
- :class:`OrderBehavior` — order frequency, size-pattern consistency, and monthly
  seasonality multipliers.
- :class:`InvoiceBehavior` — invoice timing speed, accuracy percentage, and inquiry
  response turnaround.

Five named quality tiers are supported: *excellent*, *good*, *average*, *poor*, and
*problem*.  The :class:`BehaviorProfileFactory` provides deterministic template lookup
and per-entity random variation for realistic diversity.

Primary Components
------------------
:class:`ExternalWorldManager`
    Central coordinator managing tiered entity pools and delegating simulation work
    to the three specialist simulators below.
:class:`CustomerSimulator`
    Generates customer orders, simulates payments, and handles dispute scenarios.
:class:`VendorSimulator`
    Generates vendor invoices, simulates goods delivery, and responds to inquiries.
:class:`BankSimulator`
    Generates bank statements and processes payment transactions.

Integration Points
------------------
- Consumes ``app.statistical`` distribution models for amount/timing/frequency sampling.
- Publishes events through ``app.events.EventBus`` for cross-subsystem coordination.
- Optionally queries Project 1 REST API (read-only via ``aiohttp``) for master data.

Usage Example::

    from app.external_world import ExternalWorldManager, BehaviorProfile

    manager = ExternalWorldManager(config=config, event_bus=event_bus, ...)
    await manager.initialize_entity_pools(customers_data, vendors_data)
    interactions = await manager.generate_daily_interactions(simulation_date)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Behavior profile models (foundational data layer — no internal app deps)
# ---------------------------------------------------------------------------
from app.external_world.behavior_profiles import (
    BehaviorProfile,
    BehaviorProfileFactory,
    InvoiceBehavior,
    OrderBehavior,
    PaymentBehavior,
    TierDistribution,
)

# ---------------------------------------------------------------------------
# Entity manager (central coordinator)
# ---------------------------------------------------------------------------
from app.external_world.external_entity_manager import ExternalWorldManager

# ---------------------------------------------------------------------------
# Specialist simulators
# ---------------------------------------------------------------------------
from app.external_world.customer_simulator import CustomerSimulator
from app.external_world.vendor_simulator import VendorSimulator
from app.external_world.bank_simulator import BankSimulator

# ---------------------------------------------------------------------------
# Public API — all symbols importable via ``from app.external_world import X``
# ---------------------------------------------------------------------------
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

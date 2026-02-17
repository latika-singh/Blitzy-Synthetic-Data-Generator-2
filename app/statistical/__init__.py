"""
Statistical Models Package (F-006) — Synthetic ERP Data Generation Platform, Project 2.

This package provides pure-function statistical distribution models for generating
realistic ERP transaction patterns. Each model class encapsulates parameterized
distributions backed by scipy.stats and numpy, supporting reproducible seeding
through numpy.random.RandomState.

Exported Models
---------------
AmountDistribution
    Log-normal purchase order amount sampling (s=1.2, scale=5000), vendor invoice
    matching with configurable match rates (95% within ±5%), customer order sizing
    by tier (small/medium/large), business-appropriate rounding, and early-payment
    discount application (e.g., 2/10 Net 30).

PaymentTimingModel
    Five-segment mixture model for customer payment timing, combining Normal and
    LogNormal distributions with profile-weighted segment selection and automatic
    weekend/holiday adjustment.

ApprovalTimingModel
    Approval processing time sampling using Normal(μ=4hrs, σ=2hrs) with escalation
    support (+1 day per escalation level) and business-day conversion utilities.

ProcessingTimingModel
    Generic ERP workflow processing durations for invoice processing, purchase order
    creation, goods receipt, and other operational tasks with configurable per-process
    parameters.

OrderFrequencyModel
    Poisson process for order generation frequency with day-of-week multipliers
    (Monday 0.85× through Friday 1.2×), supporting per-customer rate adjustment
    and intra-day order time distribution.

SelectionModel
    Pareto 80/20 vendor selection weighted by spend history, revenue-weighted
    customer selection, and 70/30 repeat-versus-new product selection with
    configurable concentration parameters.

Performance Targets
-------------------
All models are designed to achieve ≥ 10,000 samples/second (README.md line 48).
Package initialization is intentionally lightweight — no heavy computation, model
fitting, or resource loading occurs at import time. All expensive operations are
deferred to class instantiation or method invocation.

Usage
-----
Import individual models from this package for convenient access::

    from app.statistical import AmountDistribution, PaymentTimingModel
    from app.statistical import ApprovalTimingModel, ProcessingTimingModel
    from app.statistical import OrderFrequencyModel, SelectionModel

Dependencies
------------
- numpy (>=1.26.0): Array operations and reproducible random state seeding.
- scipy (>=1.12.0): Statistical distributions — lognorm, norm, poisson, pareto.
- pydantic (>=2.5.0): Configuration validation at subsystem boundaries (V2 models).
- structlog (>=24.1.0): Structured JSON logging to stdout.
"""

# ---------------------------------------------------------------------------
# Public API re-exports from submodules
# ---------------------------------------------------------------------------
# Each import below re-exports a distribution model class that is fully
# implemented in its own submodule.  No computation occurs at import time —
# classes are lightweight references until explicitly instantiated.

from app.statistical.amount_distributions import AmountDistribution
from app.statistical.payment_timing_model import PaymentTimingModel
from app.statistical.timing_models import ApprovalTimingModel, ProcessingTimingModel
from app.statistical.order_frequency_model import OrderFrequencyModel
from app.statistical.selection_models import SelectionModel

# ---------------------------------------------------------------------------
# Explicit public API declaration
# ---------------------------------------------------------------------------
__all__ = [
    "AmountDistribution",
    "PaymentTimingModel",
    "ApprovalTimingModel",
    "ProcessingTimingModel",
    "OrderFrequencyModel",
    "SelectionModel",
]

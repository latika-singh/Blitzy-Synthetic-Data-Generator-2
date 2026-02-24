"""
Vendor Simulator — Supply-side entity simulation for the ERP system.

This module implements the ``VendorSimulator`` class responsible for simulating
vendor interactions within the Synthetic ERP Data Generation Platform (Project 2,
F-005).  Vendors are the supply-side counterparts to customers and generate the
following interactions:

* **Invoice Generation**: Creates vendor invoices tied to purchase orders with
  realistic amount matching (95 % exact match, 5 % variance ±5 %), timing
  driven by ``BehaviorProfile.invoice_timing`` (immediate / prompt / slow), and
  accuracy governed by ``BehaviorProfile.invoice_accuracy``.
* **Goods Delivery**: Simulates goods receipt with tier-dependent lead times,
  partial-delivery probability (5 %), and back-order probability (2 %).
* **Inquiry Response**: Models vendor response latency to purchasing inquiries,
  parameterised by ``BehaviorProfile.response_time``.
* **Vendor Selection**: Pareto 80/20 weighted vendor selection via the
  ``SelectionModel`` for purchase-order assignment.

**Vendor Tiers** (README.md lines 469-474):
  - Strategic (10 %): short lead times, immediate invoicing, high accuracy.
  - Standard  (30 %): normal lead times, prompt invoicing, good accuracy.
  - Transactional (60 %): longer lead times, slow invoicing, variable accuracy.

All public methods are async coroutines (AAP §0.7.3 — no blocking I/O).  Data
contracts crossing subsystem boundaries use Pydantic V2 ``BaseModel`` (AAP §0.7.1).
Reproducibility is ensured via ``numpy.random.RandomState`` seeding.  Logging
uses ``structlog`` to stdout in structured JSON format (AAP §0.7.6).

Dependencies are wired through constructor injection (AAP §0.7.1), with the
``SimulationEngine`` serving as the composition root.

References:
    - README.md lines 456-508: Vendor interaction specifications
    - README.md lines 522-525: Invoice variance (95 % / 5 %)
    - README.md lines 571-575: Vendor selection Pareto 80/20
    - AAP §0.5.1 Group 7: External World Simulation
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import UUID, uuid4

import numpy as np
import structlog
from pydantic import BaseModel, Field

from app.external_world.behavior_profiles import BehaviorProfile
from app.events.event_types import DocumentGenerated

if TYPE_CHECKING:
    from app.statistical.amount_distributions import AmountDistribution
    from app.statistical.timing_models import ApprovalTimingModel
    from app.statistical.selection_models import SelectionModel
    from app.events.event_bus import EventBus

# ---------------------------------------------------------------------------
# Module-level structured logger (AAP §0.7.6 — structlog to stdout)
# ---------------------------------------------------------------------------
logger = structlog.get_logger(__name__)


# ===========================================================================
# Configuration Model (Pydantic V2)
# ===========================================================================


class VendorSimulatorConfig(BaseModel):
    """Configuration for vendor simulation parameters.

    Provides sensible defaults matching the specification while allowing
    per-simulation overrides through the Pydantic V2 model interface.

    Attributes:
        invoice_timing_days: Mapping of timing category to days delay after
            delivery before the vendor sends an invoice.  Default:
            ``{"immediate": 0, "prompt": 3, "slow": 10}``.
        delivery_lead_time_min: Minimum delivery lead time in calendar days.
        delivery_lead_time_max: Maximum delivery lead time in calendar days.
        partial_delivery_rate: Probability (0.0–1.0) of a partial delivery.
        backorder_rate: Probability (0.0–1.0) of a back-order situation.
        inquiry_response_default_days: Fallback response time when the vendor's
            ``BehaviorProfile`` does not specify ``response_time``.
    """

    invoice_timing_days: Dict[str, int] = Field(
        default_factory=lambda: {"immediate": 0, "prompt": 3, "slow": 10},
        description="Mapping of timing type to days delay after delivery",
    )
    delivery_lead_time_min: int = Field(
        default=3,
        ge=1,
        description="Minimum delivery lead time in calendar days",
    )
    delivery_lead_time_max: int = Field(
        default=30,
        ge=1,
        description="Maximum delivery lead time in calendar days",
    )
    partial_delivery_rate: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability of a partial delivery (5 %)",
    )
    backorder_rate: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description="Probability of a back-order situation (2 %)",
    )
    inquiry_response_default_days: int = Field(
        default=5,
        ge=1,
        description="Default inquiry response time in days",
    )


# ===========================================================================
# Data Models (Pydantic V2 — subsystem boundary contracts)
# ===========================================================================


class VendorInvoice(BaseModel):
    """Vendor invoice generated in response to a purchase order.

    Captures the invoice amount, its relationship to the original PO amount,
    variance details, and processing status.  Used as the cross-subsystem data
    contract between the external-world simulation and the agent system.

    Attributes:
        invoice_id: Unique invoice identifier (UUID4 string).
        vendor_id: Identifier of the issuing vendor.
        po_id: Linked purchase order identifier.
        invoice_date: Date the invoice was issued.
        invoice_amount: Billed amount in USD.
        po_amount: Original PO amount for three-way match comparison.
        variance_amount: Absolute difference between invoice and PO amounts.
            Positive means over-billed, negative means under-billed.
        variance_type: Category of variance (``"quantity"``, ``"price"``, or
            ``None`` when the invoice matches exactly).
        is_accurate: Whether the invoice is considered accurate (matches PO
            within tolerance based on the vendor's ``invoice_accuracy``).
        payment_terms: Payment terms string (e.g. ``"Net 30"``).
        status: Processing status (``"pending"``, ``"matched"``, ``"disputed"``).
    """

    invoice_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique invoice identifier (UUID4)",
    )
    vendor_id: str = Field(
        ...,
        description="Identifier of the issuing vendor",
    )
    po_id: str = Field(
        ...,
        description="Linked purchase order identifier",
    )
    invoice_date: date = Field(
        ...,
        description="Date the invoice was issued",
    )
    invoice_amount: float = Field(
        ...,
        description="Billed amount in USD",
    )
    po_amount: float = Field(
        ...,
        description="Original PO amount for matching",
    )
    variance_amount: float = Field(
        default=0.0,
        description="Difference from PO (positive = over, negative = under)",
    )
    variance_type: Optional[str] = Field(
        default=None,
        description='Variance category: "quantity", "price", or None',
    )
    is_accurate: bool = Field(
        default=True,
        description="Whether the invoice matches PO within tolerance",
    )
    payment_terms: str = Field(
        default="Net 30",
        description="Payment terms string",
    )
    status: str = Field(
        default="pending",
        description='Processing status: "pending", "matched", "disputed"',
    )


class GoodsDelivery(BaseModel):
    """Record of a goods delivery from a vendor against a purchase order.

    Tracks ordered vs. delivered quantities, partial-delivery and back-order
    flags, and delivery status.

    Attributes:
        delivery_id: Unique delivery identifier (UUID4 string).
        vendor_id: Identifier of the delivering vendor.
        po_id: Linked purchase order identifier.
        delivery_date: Date goods were received.
        items_ordered: Quantity ordered on the PO.
        items_delivered: Quantity actually delivered.
        is_partial: True when ``items_delivered < items_ordered``.
        is_backorder: True when the remaining items are on back-order.
        status: Delivery status (``"delivered"``, ``"partial"``, ``"backordered"``).
    """

    delivery_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique delivery identifier (UUID4)",
    )
    vendor_id: str = Field(
        ...,
        description="Identifier of the delivering vendor",
    )
    po_id: str = Field(
        ...,
        description="Linked purchase order identifier",
    )
    delivery_date: date = Field(
        ...,
        description="Date goods were received",
    )
    items_ordered: int = Field(
        ...,
        ge=1,
        description="Quantity ordered on the PO",
    )
    items_delivered: int = Field(
        ...,
        ge=0,
        description="Quantity actually delivered",
    )
    is_partial: bool = Field(
        default=False,
        description="True when items_delivered < items_ordered",
    )
    is_backorder: bool = Field(
        default=False,
        description="True when remaining items are on back-order",
    )
    status: str = Field(
        default="delivered",
        description='Delivery status: "delivered", "partial", "backordered"',
    )


class VendorInquiryResponse(BaseModel):
    """Vendor response to a purchasing inquiry.

    Models the turnaround time for vendor inquiries (pricing, availability,
    status, quality), parameterised by the vendor's ``BehaviorProfile.response_time``.

    Attributes:
        inquiry_id: Unique inquiry identifier (UUID4 string).
        vendor_id: Identifier of the responding vendor.
        inquiry_date: Date the inquiry was sent.
        response_date: Date the vendor responded.
        response_time_days: Elapsed business days between inquiry and response.
        inquiry_type: Category of inquiry (``"pricing"``, ``"availability"``,
            ``"status"``, ``"quality"``).
        status: Response status (``"responded"``, ``"pending"``, ``"escalated"``).
    """

    inquiry_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique inquiry identifier (UUID4)",
    )
    vendor_id: str = Field(
        ...,
        description="Identifier of the responding vendor",
    )
    inquiry_date: date = Field(
        ...,
        description="Date the inquiry was sent",
    )
    response_date: date = Field(
        ...,
        description="Date the vendor responded",
    )
    response_time_days: int = Field(
        ...,
        ge=0,
        description="Elapsed business days between inquiry and response",
    )
    inquiry_type: str = Field(
        ...,
        description='Inquiry category: "pricing", "availability", "status", "quality"',
    )
    status: str = Field(
        default="responded",
        description='Response status: "responded", "pending", "escalated"',
    )


# ===========================================================================
# VendorSimulator
# ===========================================================================


class VendorSimulator:
    """Simulates vendor interactions with the ERP system.

    Generates realistic vendor invoices, goods deliveries, and inquiry
    responses driven by statistical models and entity-level behaviour
    profiles.  All dependencies are injected via the constructor
    (AAP §0.7.1 — constructor injection pattern).

    The simulator integrates with:
    - ``AmountDistribution`` for invoice amount sampling (95 % / 5 % logic).
    - ``SelectionModel`` for Pareto 80/20 vendor selection.
    - ``EventBus`` for publishing ``DocumentGenerated`` events.
    - ``BehaviorProfile`` for per-vendor timing, accuracy, and response.

    All public methods are async coroutines with no blocking I/O.

    Example::

        sim = VendorSimulator(
            amount_distribution=amount_dist,
            selection_model=sel_model,
            event_bus=bus,
            seed=42,
        )
        invoice = await sim.generate_vendor_invoice(vendor, po, today)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[VendorSimulatorConfig] = None,
        amount_distribution: Optional[AmountDistribution] = None,
        selection_model: Optional[SelectionModel] = None,
        event_bus: Optional[EventBus] = None,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the VendorSimulator with injected dependencies.

        Args:
            config: Simulation configuration.  Uses ``VendorSimulatorConfig``
                defaults when *None*.
            amount_distribution: Statistical model for invoice amount
                sampling.  Falls back to simple PO-matching when *None*.
            selection_model: Pareto 80/20 vendor selection model.  Falls
                back to uniform random selection when *None*.
            event_bus: Event bus for publishing ``DocumentGenerated`` events.
                Events are silently skipped when *None*.
            seed: Integer seed for ``numpy.random.RandomState`` to ensure
                reproducible simulation runs.
        """
        self._config = config if config is not None else VendorSimulatorConfig()
        self._amount_distribution = amount_distribution
        self._selection_model = selection_model
        self._event_bus = event_bus
        self.rng = np.random.RandomState(seed)

        logger.info(
            "vendor_simulator_initialized",
            seed=seed,
            delivery_lead_time_min=self._config.delivery_lead_time_min,
            delivery_lead_time_max=self._config.delivery_lead_time_max,
            partial_delivery_rate=self._config.partial_delivery_rate,
            backorder_rate=self._config.backorder_rate,
            has_amount_distribution=amount_distribution is not None,
            has_selection_model=selection_model is not None,
            has_event_bus=event_bus is not None,
        )

    # ------------------------------------------------------------------
    # Invoice Generation (README.md lines 456-508, 522-525)
    # ------------------------------------------------------------------

    async def generate_vendor_invoice(
        self,
        vendor: Dict[str, Any],
        po: Dict[str, Any],
        simulation_date: date,
    ) -> VendorInvoice:
        """Generate a single vendor invoice based on a purchase order.

        Invoice amount determination:
          1. If an ``AmountDistribution`` is injected, delegates to
             ``sample_vendor_invoice_amount(po_amount)`` which internally
             applies the 95 % exact-match / 5 % variance (±5 %) rule.
          2. Otherwise, the PO amount is used directly.

        Invoice accuracy (separate from amount variance):
          - If ``rng.random() > profile.invoice_accuracy``, the invoice is
            marked as inaccurate and an additional intentional error is
            introduced (±3 % of the amount).

        Timing is driven by the vendor's ``BehaviorProfile.invoice_timing``
        mapped through ``VendorSimulatorConfig.invoice_timing_days``.

        A ``DocumentGenerated`` event is published to the ``EventBus`` when
        available.

        Args:
            vendor: Vendor dictionary with at least ``"vendor_id"`` and
                optionally ``"behavior_profile"`` / ``"tier"`` / ``"payment_terms"``.
            po: Purchase order dictionary with at least ``"po_id"`` and
                ``"amount"`` (or ``"total_amount"``).
            simulation_date: Current simulation date for invoice dating.

        Returns:
            A fully populated ``VendorInvoice`` model.
        """
        vendor_id: str = str(vendor.get("vendor_id", vendor.get("id", "unknown")))
        po_id: str = str(po.get("po_id", po.get("id", "unknown")))
        po_amount: float = float(po.get("amount", po.get("total_amount", 0.0)))
        payment_terms: str = str(vendor.get("payment_terms", po.get("payment_terms", "Net 30")))

        # --- Behaviour profile lookups ---
        profile: Optional[BehaviorProfile] = self._get_vendor_profile(vendor)
        invoice_timing: str = self._get_invoice_timing(profile)
        invoice_accuracy: float = profile.invoice_accuracy if profile else 0.95

        # --- Invoice date calculation ---
        timing_delay_days: int = self._config.invoice_timing_days.get(
            invoice_timing,
            self._config.invoice_timing_days.get("prompt", 3),
        )
        invoice_date: date = simulation_date + timedelta(days=timing_delay_days)

        # --- Invoice amount determination ---
        if self._amount_distribution is not None:
            invoice_amount = self._amount_distribution.sample_vendor_invoice_amount(po_amount)
        else:
            # Fallback: simple PO matching with 95 % / 5 % logic
            if self.rng.random() < 0.95:
                invoice_amount = po_amount
            else:
                variance_factor = self.rng.uniform(-0.05, 0.05)
                invoice_amount = po_amount * (1.0 + variance_factor)

        # --- Variance tracking ---
        variance_amount: float = round(invoice_amount - po_amount, 2)
        has_variance: bool = abs(variance_amount) > 0.005
        variance_type: Optional[str] = None
        if has_variance:
            variance_type = "price" if self.rng.random() < 0.6 else "quantity"

        # --- Accuracy check (BehaviorProfile driven) ---
        is_accurate: bool = True
        if self.rng.random() > invoice_accuracy:
            # Vendor produced an inaccurate invoice — introduce intentional error
            is_accurate = False
            if not has_variance:
                # Add a deliberate error when the amount was originally correct
                accuracy_error = self.rng.uniform(-0.03, 0.03)
                invoice_amount = po_amount * (1.0 + accuracy_error)
                variance_amount = round(invoice_amount - po_amount, 2)
                variance_type = "price"

        # Round final amount to 2 decimal places
        invoice_amount = round(invoice_amount, 2)

        invoice = VendorInvoice(
            vendor_id=vendor_id,
            po_id=po_id,
            invoice_date=invoice_date,
            invoice_amount=invoice_amount,
            po_amount=po_amount,
            variance_amount=variance_amount,
            variance_type=variance_type,
            is_accurate=is_accurate,
            payment_terms=payment_terms,
            status="pending",
        )

        # --- Event publishing ---
        if self._event_bus is not None:
            event = DocumentGenerated(
                payload={
                    "document_type": "vendor_invoice",
                    "document_id": invoice.invoice_id,
                    "vendor_id": vendor_id,
                    "po_id": po_id,
                    "invoice_amount": invoice_amount,
                    "has_variance": has_variance,
                },
            )
            try:
                await self._event_bus.publish(event)
            except Exception:
                logger.warning(
                    "vendor_invoice_event_publish_failed",
                    vendor_id=vendor_id,
                    po_id=po_id,
                    exc_info=True,
                )

        logger.info(
            "vendor_invoice_generated",
            vendor_id=vendor_id,
            po_id=po_id,
            invoice_amount=invoice_amount,
            po_amount=po_amount,
            has_variance=has_variance,
            is_accurate=is_accurate,
            invoice_timing=invoice_timing,
        )

        return invoice

    async def generate_batch_invoices(
        self,
        vendors_with_pos: List[Dict[str, Any]],
        simulation_date: date,
    ) -> List[VendorInvoice]:
        """Generate invoices for multiple vendor/PO pairs.

        Iterates over the provided list and delegates to
        ``generate_vendor_invoice`` for each pair.  Each entry in the list
        must contain ``"vendor"`` and ``"po"`` keys.

        Performance target: < 2 seconds for typical batch sizes (AAP criterion #18).

        Args:
            vendors_with_pos: List of dictionaries, each containing
                ``"vendor"`` (vendor dict) and ``"po"`` (purchase order dict).
            simulation_date: Current simulation date.

        Returns:
            List of ``VendorInvoice`` models in the same order as input.
        """
        invoices: List[VendorInvoice] = []
        for entry in vendors_with_pos:
            vendor: Dict[str, Any] = entry.get("vendor", entry)
            po: Dict[str, Any] = entry.get("po", entry)
            invoice = await self.generate_vendor_invoice(vendor, po, simulation_date)
            invoices.append(invoice)

        logger.info(
            "vendor_batch_invoices_generated",
            batch_size=len(invoices),
            simulation_date=simulation_date.isoformat(),
        )
        return invoices

    # ------------------------------------------------------------------
    # Goods Delivery Simulation
    # ------------------------------------------------------------------

    async def simulate_goods_delivery(
        self,
        vendor: Dict[str, Any],
        po: Dict[str, Any],
        simulation_date: date,
    ) -> GoodsDelivery:
        """Simulate a goods delivery from a vendor for a purchase order.

        Delivery lead time varies by vendor tier:
          - **Strategic**: ``[lead_min, lead_min + 5)`` days.
          - **Standard**: ``[lead_min, lead_max // 2)`` days.
          - **Transactional**: ``[lead_max // 2, lead_max)`` days.

        Partial delivery occurs with probability ``config.partial_delivery_rate``
        (default 5 %).  Back-order occurs with probability ``config.backorder_rate``
        (default 2 %).

        A ``DocumentGenerated`` event is published to the ``EventBus`` for
        goods receipt documents.

        Args:
            vendor: Vendor dictionary with ``"vendor_id"`` and optionally
                ``"tier"``.
            po: Purchase order dictionary with ``"po_id"`` and ``"items"``
                (or ``"quantity"``).
            simulation_date: Current simulation date.

        Returns:
            A fully populated ``GoodsDelivery`` model.
        """
        vendor_id: str = str(vendor.get("vendor_id", vendor.get("id", "unknown")))
        po_id: str = str(po.get("po_id", po.get("id", "unknown")))
        items_ordered: int = int(po.get("items", po.get("quantity", 10)))
        tier: str = self._get_vendor_tier(vendor)

        # --- Delivery lead-time by tier ---
        lead_min = self._config.delivery_lead_time_min
        lead_max = self._config.delivery_lead_time_max
        half_max = max(lead_min + 1, lead_max // 2)

        if tier == "strategic":
            upper = min(lead_min + 6, half_max)
            lead_days = int(self.rng.randint(lead_min, max(lead_min + 1, upper)))
        elif tier == "transactional":
            lead_days = int(self.rng.randint(half_max, max(half_max + 1, lead_max + 1)))
        else:
            # "standard" or unknown tiers
            lead_days = int(self.rng.randint(lead_min, max(lead_min + 1, half_max)))

        delivery_date: date = simulation_date + timedelta(days=lead_days)

        # --- Partial delivery determination ---
        is_partial: bool = self.rng.random() < self._config.partial_delivery_rate
        if is_partial and items_ordered > 1:
            items_delivered = int(self.rng.randint(1, items_ordered))
        else:
            items_delivered = items_ordered
            is_partial = False  # Cannot be partial with a single item

        # --- Back-order determination ---
        is_backorder: bool = False
        if is_partial and self.rng.random() < self._config.backorder_rate:
            is_backorder = True

        # --- Delivery status ---
        if is_backorder:
            status = "backordered"
        elif is_partial:
            status = "partial"
        else:
            status = "delivered"

        delivery = GoodsDelivery(
            vendor_id=vendor_id,
            po_id=po_id,
            delivery_date=delivery_date,
            items_ordered=items_ordered,
            items_delivered=items_delivered,
            is_partial=is_partial,
            is_backorder=is_backorder,
            status=status,
        )

        # --- Event publishing ---
        if self._event_bus is not None:
            event = DocumentGenerated(
                payload={
                    "document_type": "goods_receipt",
                    "document_id": delivery.delivery_id,
                    "vendor_id": vendor_id,
                    "po_id": po_id,
                    "items_delivered": items_delivered,
                    "is_partial": is_partial,
                },
            )
            try:
                await self._event_bus.publish(event)
            except Exception:
                logger.warning(
                    "goods_delivery_event_publish_failed",
                    vendor_id=vendor_id,
                    po_id=po_id,
                    exc_info=True,
                )

        logger.info(
            "goods_delivery_simulated",
            vendor_id=vendor_id,
            po_id=po_id,
            items_ordered=items_ordered,
            items_delivered=items_delivered,
            is_partial=is_partial,
            is_backorder=is_backorder,
            tier=tier,
            lead_days=lead_days,
        )

        return delivery

    # ------------------------------------------------------------------
    # Inquiry Response Simulation
    # ------------------------------------------------------------------

    async def simulate_inquiry_response(
        self,
        vendor: Dict[str, Any],
        inquiry_date: date,
        inquiry_type: str = "status",
    ) -> VendorInquiryResponse:
        """Simulate a vendor's response to a purchasing inquiry.

        Response time is driven by the vendor's ``BehaviorProfile.response_time``
        with ±1 day variance.  Falls back to
        ``config.inquiry_response_default_days`` when no profile is available.

        Args:
            vendor: Vendor dictionary with ``"vendor_id"`` and optionally
                ``"behavior_profile"``.
            inquiry_date: Date the inquiry was sent.
            inquiry_type: Category of inquiry.  One of ``"pricing"``,
                ``"availability"``, ``"status"``, ``"quality"``.  Defaults
                to ``"status"``.

        Returns:
            A fully populated ``VendorInquiryResponse`` model.
        """
        vendor_id: str = str(vendor.get("vendor_id", vendor.get("id", "unknown")))
        profile: Optional[BehaviorProfile] = self._get_vendor_profile(vendor)

        # --- Base response time ---
        if profile is not None:
            base_response_days: int = profile.response_time
        else:
            base_response_days = self._config.inquiry_response_default_days

        # --- Add variance: ±1 day (minimum 1 day) ---
        variance = int(self.rng.randint(-1, 2))  # -1, 0, or +1
        response_days: int = max(1, base_response_days + variance)
        response_date: date = inquiry_date + timedelta(days=response_days)

        # --- Validate inquiry type ---
        valid_types = {"pricing", "availability", "status", "quality"}
        if inquiry_type not in valid_types:
            logger.warning(
                "invalid_inquiry_type_defaulting",
                vendor_id=vendor_id,
                inquiry_type=inquiry_type,
                default="status",
            )
            inquiry_type = "status"

        inquiry = VendorInquiryResponse(
            vendor_id=vendor_id,
            inquiry_date=inquiry_date,
            response_date=response_date,
            response_time_days=response_days,
            inquiry_type=inquiry_type,
            status="responded",
        )

        logger.info(
            "vendor_inquiry_response_simulated",
            vendor_id=vendor_id,
            inquiry_type=inquiry_type,
            response_time_days=response_days,
            base_response_days=base_response_days,
        )

        return inquiry

    # ------------------------------------------------------------------
    # Vendor Selection (Pareto 80/20 — README.md lines 571-575)
    # ------------------------------------------------------------------

    async def select_vendor_for_purchase(
        self,
        vendors: List[Dict[str, Any]],
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Select a vendor for a new purchase using Pareto 80/20 weighting.

        Delegates to ``SelectionModel.select_vendor()`` when an injected
        model is available.  Falls back to uniform random selection otherwise.

        Args:
            vendors: Non-empty list of vendor dictionaries.
            category: Optional procurement category to filter by.

        Returns:
            The selected vendor dictionary.

        Raises:
            ValueError: If *vendors* is empty.
        """
        if not vendors:
            raise ValueError("Cannot select from an empty vendor list.")

        if self._selection_model is not None:
            selected = self._selection_model.select_vendor(vendors, category)
        else:
            # Fallback: uniform random selection
            idx = int(self.rng.randint(0, len(vendors)))
            selected = vendors[idx]

        logger.debug(
            "vendor_selected_for_purchase",
            vendor_id=str(selected.get("vendor_id", selected.get("id", "unknown"))),
            category=category,
            pool_size=len(vendors),
            used_selection_model=self._selection_model is not None,
        )
        return selected

    # ------------------------------------------------------------------
    # Private Utility Methods
    # ------------------------------------------------------------------

    def _get_vendor_tier(self, vendor: Dict[str, Any]) -> str:
        """Extract the vendor tier from a vendor dictionary.

        Looks for ``"tier"`` key in the vendor dict.  Falls back to
        ``"standard"`` when absent or unrecognised.

        Args:
            vendor: Vendor dictionary.

        Returns:
            Lowercase tier string: ``"strategic"``, ``"standard"``, or
            ``"transactional"``.
        """
        tier: str = str(vendor.get("tier", "standard")).lower().strip()
        if tier not in {"strategic", "standard", "transactional"}:
            tier = "standard"
        return tier

    def _get_vendor_profile(self, vendor: Dict[str, Any]) -> Optional[BehaviorProfile]:
        """Extract a ``BehaviorProfile`` from a vendor dictionary.

        The profile may be stored under ``"behavior_profile"``, ``"profile"``,
        or a nested key.  Returns *None* when no profile is found.

        Args:
            vendor: Vendor dictionary.

        Returns:
            A ``BehaviorProfile`` instance, or *None*.
        """
        raw_profile = vendor.get("behavior_profile", vendor.get("profile"))

        if raw_profile is None:
            return None

        if isinstance(raw_profile, BehaviorProfile):
            return raw_profile

        # Attempt to construct from dict
        if isinstance(raw_profile, dict):
            try:
                return BehaviorProfile(**raw_profile)
            except Exception:
                logger.warning(
                    "invalid_vendor_behavior_profile",
                    vendor_id=str(vendor.get("vendor_id", vendor.get("id", "unknown"))),
                    exc_info=True,
                )
                return None

        return None

    def _get_invoice_timing(self, profile: Optional[BehaviorProfile]) -> str:
        """Extract invoice timing from a ``BehaviorProfile``.

        Returns ``"prompt"`` as the default when the profile is absent or
        does not specify invoice timing.

        Args:
            profile: Optional behaviour profile.

        Returns:
            One of ``"immediate"``, ``"prompt"``, ``"slow"``.
        """
        if profile is None:
            return "prompt"

        timing: str = profile.invoice_timing
        valid_timings = {"immediate", "prompt", "slow"}
        if timing not in valid_timings:
            return "prompt"
        return timing

"""Specialized ERP agent implementations.

This package contains 12 specialized agent subclasses of
:class:`~app.agents.base_agent.BaseAgent`, each implementing
:meth:`process_work_item` for a specific ERP functional role.

Agent hierarchy by functional area:

- **Accounts Payable**: :class:`APClerkAgent`, :class:`APManagerAgent`
- **Accounts Receivable**: :class:`ARClerkAgent`, :class:`ARManagerAgent`
- **Purchasing**: :class:`PurchasingAgent`, :class:`PurchasingManagerAgent`
- **Warehouse**: :class:`WarehouseClerkAgent`, :class:`WarehouseManagerAgent`
- **Accounting**: :class:`AccountantAgent`, :class:`SeniorAccountantAgent`
- **Financial Control**: :class:`ControllerAgent`, :class:`CFOAgent`

The convenience mapping :data:`AGENT_CLASSES` maps each agent's
``ROLE`` string to its class, enabling the
:class:`~app.agents.agent_registry.AgentRegistry` to instantiate
agents dynamically by role name.

Usage::

    from app.agents.specialized import APClerkAgent, AGENT_CLASSES

    # Instantiate by class directly
    agent = APClerkAgent(config=config, memory=memory,
                         decision_engine=engine,
                         action_registry=registry)

    # Or look up by role string
    cls = AGENT_CLASSES["ap_clerk"]
    agent = cls(config=config, memory=memory,
                decision_engine=engine,
                action_registry=registry)

References:
    - AAP Section 0.5.1 Group 5: "Specialized agent package exports"
    - README.md lines 772-890 (specialized agent requirements)
    - AAP Section 0.7.1 (every specialized agent MUST extend BaseAgent)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Accounts Payable
# ---------------------------------------------------------------------------
from app.agents.specialized.ap_clerk_agent import APClerkAgent
from app.agents.specialized.ap_manager_agent import APManagerAgent

# ---------------------------------------------------------------------------
# Accounts Receivable
# ---------------------------------------------------------------------------
from app.agents.specialized.ar_clerk_agent import ARClerkAgent
from app.agents.specialized.ar_manager_agent import ARManagerAgent

# ---------------------------------------------------------------------------
# Purchasing
# ---------------------------------------------------------------------------
from app.agents.specialized.purchasing_agent import PurchasingAgent
from app.agents.specialized.purchasing_manager_agent import PurchasingManagerAgent

# ---------------------------------------------------------------------------
# Warehouse
# ---------------------------------------------------------------------------
from app.agents.specialized.warehouse_clerk_agent import WarehouseClerkAgent
from app.agents.specialized.warehouse_manager_agent import WarehouseManagerAgent

# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------
from app.agents.specialized.accountant_agent import AccountantAgent
from app.agents.specialized.senior_accountant_agent import SeniorAccountantAgent

# ---------------------------------------------------------------------------
# Financial Control
# ---------------------------------------------------------------------------
from app.agents.specialized.controller_agent import ControllerAgent
from app.agents.specialized.cfo_agent import CFOAgent

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__: list[str] = [
    # Accounts Payable
    "APClerkAgent",
    "APManagerAgent",
    # Accounts Receivable
    "ARClerkAgent",
    "ARManagerAgent",
    # Purchasing
    "PurchasingAgent",
    "PurchasingManagerAgent",
    # Warehouse
    "WarehouseClerkAgent",
    "WarehouseManagerAgent",
    # Accounting
    "AccountantAgent",
    "SeniorAccountantAgent",
    # Financial Control
    "ControllerAgent",
    "CFOAgent",
    # Role-to-class mapping
    "AGENT_CLASSES",
]

# ---------------------------------------------------------------------------
# Role-to-class mapping
# ---------------------------------------------------------------------------
# Maps each agent's ROLE constant (str) to its class object.  Keys match
# the ``ROLE`` class attribute on each agent, the ``ROLE_MAPPING`` in
# :class:`~app.orchestration.workflow_orchestrator.WorkflowOrchestrator`,
# and the role identifiers in ``config/agents/agent_roles.yaml``.
#
# Used by :class:`~app.agents.agent_registry.AgentRegistry` to
# instantiate agents dynamically by role name.
# ---------------------------------------------------------------------------
AGENT_CLASSES: dict[str, type] = {
    "ap_clerk": APClerkAgent,
    "ap_manager": APManagerAgent,
    "ar_clerk": ARClerkAgent,
    "ar_manager": ARManagerAgent,
    "purchasing_agent": PurchasingAgent,
    "purchasing_manager": PurchasingManagerAgent,
    "warehouse_clerk": WarehouseClerkAgent,
    "warehouse_manager": WarehouseManagerAgent,
    "accountant": AccountantAgent,
    "senior_accountant": SeniorAccountantAgent,
    "controller": ControllerAgent,
    "cfo": CFOAgent,
}

from __future__ import annotations

from benchmark.mcp.errors import ToolDisabledError, UnknownToolError
from benchmark.mcp.profiles import McpProfile, _ALL_TOOLS, is_tool_allowed
from benchmark.mcp.tool_contract import TOOL_CONTRACT_MAP, ToolContract


class McpToolRegistry:
    """Profile-aware registry of ToolContracts.

    Uses TOOL_CONTRACT_MAP as the authoritative source of truth.
    Additional contracts can be registered at runtime.
    """

    def __init__(self, extra_contracts: list[ToolContract] | None = None) -> None:
        self._contracts: dict[str, ToolContract] = dict(TOOL_CONTRACT_MAP)
        for contract in (extra_contracts or []):
            self._contracts[contract.name] = contract

    # ------------------------------------------------------------------
    # Core API (task 5.17)
    # ------------------------------------------------------------------

    def list_available_tools(self, profile: McpProfile | None = None) -> list[ToolContract]:
        """Return all contracts allowed by *profile*, or all registered if profile is None."""
        if profile is None:
            return list(self._contracts.values())
        return [c for c in self._contracts.values() if is_tool_allowed(c.name, profile)]

    def list_disabled_tools(self, profile: McpProfile) -> list[ToolContract]:
        """Return contracts NOT allowed by *profile*."""
        return [c for c in self._contracts.values() if not is_tool_allowed(c.name, profile)]

    def get_contract(self, tool_name: str) -> ToolContract:
        """Return the contract for *tool_name*; raises UnknownToolError if not registered."""
        if tool_name not in self._contracts:
            raise UnknownToolError(f"Unknown tool: '{tool_name}'")
        return self._contracts[tool_name]

    def assert_tool_allowed(self, tool_name: str, profile: McpProfile) -> ToolContract:
        """Return the contract if allowed; raises UnknownToolError or ToolDisabledError."""
        contract = self.get_contract(tool_name)
        if not is_tool_allowed(tool_name, profile):
            raise ToolDisabledError(
                f"Tool '{tool_name}' is disabled in profile '{profile.value}'"
            )
        return contract

    def is_allowed(self, tool_name: str, profile: McpProfile) -> bool:
        """Return True if *tool_name* is known and allowed by *profile*."""
        return tool_name in self._contracts and is_tool_allowed(tool_name, profile)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, contract: ToolContract) -> None:
        self._contracts[contract.name] = contract

    # ------------------------------------------------------------------
    # Backward-compat aliases
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolContract:
        return self.get_contract(name)

    def get_allowed(self, name: str, profile: McpProfile) -> ToolContract:
        return self.assert_tool_allowed(name, profile)

    def list_all(self) -> list[ToolContract]:
        return list(self._contracts.values())

    def list_for_profile(self, profile: McpProfile) -> list[ToolContract]:
        return self.list_available_tools(profile)

    def __len__(self) -> int:
        return len(self._contracts)

import sys
from unittest.mock import MagicMock


def test_import_with_mcp_v2(monkeypatch):
    """Test importing FastMCP when mcp.server.mcpserver (v2) is present."""
    mock_mcpserver_module = MagicMock()
    mock_mcpserver_module.MCPServer = MagicMock()

    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", mock_mcpserver_module)
    if "mcp.server.fastmcp" in sys.modules:
        monkeypatch.delitem(sys.modules, "mcp.server.fastmcp")

    # Re-evaluate import logic
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        from mcp.server.fastmcp import FastMCP

    assert FastMCP is mock_mcpserver_module.MCPServer


def test_import_with_mcp_v1_fallback(monkeypatch):
    """Test fallback to mcp.server.fastmcp when mcpserver (v2) is not present."""
    mock_fastmcp_module = MagicMock()
    mock_fastmcp_module.FastMCP = MagicMock()

    # Hide v2 module
    monkeypatch.setitem(sys.modules, "mcp.server.mcpserver", None)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", mock_fastmcp_module)

    try:
        from mcp.server.mcpserver import MCPServer as FastMCP
    except ImportError:
        from mcp.server.fastmcp import FastMCP

    assert FastMCP is mock_fastmcp_module.FastMCP

"""teloscopy.platform — Platform features for extensibility and integration.

This package provides the infrastructure for extending Teloscopy through
a plugin marketplace architecture, enabling custom analysis modules,
export formats, visualization types, and integrations with external
services.  Key features include:

- Plugin discovery, installation, and lifecycle management
- Federated learning for multi-institution collaboration
- Mobile-optimised REST API with JWT auth and offline-sync
- Research collaboration and data export tools
"""

__all__ = [
    "plugin_system",
    "federated",
    "mobile_api",
    "research_tools",
]

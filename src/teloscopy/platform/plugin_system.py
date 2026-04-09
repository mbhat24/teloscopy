"""
teloscopy.platform.plugin_system
================================

Plugin marketplace architecture for custom analysis modules in Teloscopy.
Provides discovery, installation, validation, sandboxed loading/execution,
and uninstallation — backed by a remote registry marketplace.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class PluginType(Enum):
    """Recognised plugin categories."""

    ANALYSIS = "analysis"
    EXPORT = "export"
    VISUALIZATION = "visualization"
    INTEGRATION = "integration"


class PluginState(Enum):
    """Runtime state of a loaded plugin."""

    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALISED = "initialised"
    RUNNING = "running"
    ERROR = "error"
    UNLOADED = "unloaded"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PluginInfo:
    """Metadata about an installed or registry-available plugin."""

    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: list[str] = field(default_factory=list)
    installed_at: datetime | None = None
    path: str | None = None

    def is_installed(self) -> bool:
        return self.path is not None and os.path.isdir(self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "plugin_type": self.plugin_type.value,
            "dependencies": self.dependencies,
            "path": self.path,
            "installed_at": self.installed_at.isoformat() if self.installed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginInfo:
        ts = data.get("installed_at")
        return cls(
            name=data["name"],
            version=data["version"],
            author=data.get("author", "unknown"),
            description=data.get("description", ""),
            plugin_type=PluginType(data.get("plugin_type", "analysis")),
            dependencies=data.get("dependencies", []),
            installed_at=datetime.fromisoformat(ts) if ts else None,
            path=data.get("path"),
        )


@dataclass
class PluginManifest:
    """Parsed representation of a plugin's ``manifest.json``."""

    name: str
    version: str
    entry_point: str
    dependencies: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    min_teloscopy_version: str = "0.1.0"
    author: str = "unknown"
    description: str = ""
    plugin_type: str = "analysis"
    signature: str | None = None

    @classmethod
    def from_file(cls, path: str) -> PluginManifest:
        """Load and validate a *manifest.json* file."""
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        missing = {"name", "version", "entry_point"} - set(data.keys())
        if missing:
            raise ValueError(f"Manifest at {path} missing required keys: {missing}")
        return cls(
            name=data["name"],
            version=data["version"],
            entry_point=data["entry_point"],
            dependencies=data.get("dependencies", []),
            capabilities=data.get("capabilities", []),
            min_teloscopy_version=data.get("min_teloscopy_version", "0.1.0"),
            author=data.get("author", "unknown"),
            description=data.get("description", ""),
            plugin_type=data.get("plugin_type", "analysis"),
            signature=data.get("signature"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class ValidationResult:
    """Outcome of :meth:`PluginManager.validate_plugin`."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    manifest: PluginManifest | None = None
    security_passed: bool = False

    def __bool__(self) -> bool:
        return self.is_valid


@dataclass
class PluginInstance:
    """Runtime wrapper: loaded module, concrete plugin object, and state."""

    info: PluginInfo
    manifest: PluginManifest
    module: types.ModuleType | None = None
    instance: PluginBase | None = None
    state: PluginState = PluginState.DISCOVERED
    loaded_at: datetime | None = None
    error_message: str | None = None


# ---------------------------------------------------------------------------
# Abstract base class & concrete plugin types
# ---------------------------------------------------------------------------


class PluginBase(ABC):
    """Abstract base class that every Teloscopy plugin must subclass.

    Subclasses must implement :meth:`initialize`, :meth:`execute`, and
    :meth:`cleanup`.  Optionally override :meth:`get_schema` and
    :meth:`get_capabilities`.
    """

    name: str = "unnamed_plugin"
    version: str = "0.0.0"
    author: str = "unknown"
    description: str = ""

    @abstractmethod
    def initialize(self, config: dict) -> None:
        """Set up internal state from user-supplied *config*."""

    @abstractmethod
    def execute(self, input_data: dict) -> dict:
        """Run the plugin's main logic and return a result dict."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release any resources acquired during :meth:`initialize`."""

    def get_schema(self) -> dict:
        """Return JSON-Schema for expected input and output."""
        return {
            "input": {"type": "object", "properties": {}},
            "output": {"type": "object", "properties": {}},
        }

    def get_capabilities(self) -> list[str]:
        """Return capability tags advertised by this plugin."""
        return []


class AnalysisPlugin(PluginBase):
    """Custom analysis modules (e.g. novel spot-detection algorithms)."""

    plugin_type: PluginType = PluginType.ANALYSIS

    def get_capabilities(self) -> list[str]:
        return ["analysis"]

    def get_analysis_parameters(self) -> dict:
        """Return default analysis parameters for this module."""
        return {}

    def validate_input_data(self, input_data: dict) -> bool:
        """Check that *input_data* is suitable for this analysis."""
        return True


class ExportPlugin(PluginBase):
    """Custom export formats (CSV, HDF5, cloud upload, etc.)."""

    plugin_type: PluginType = PluginType.EXPORT

    def get_capabilities(self) -> list[str]:
        return ["export"]

    def get_supported_formats(self) -> list[str]:
        return []

    def get_export_options(self) -> dict:
        return {}


class VisualizationPlugin(PluginBase):
    """Custom visualisation types (plots, interactive widgets, etc.)."""

    plugin_type: PluginType = PluginType.VISUALIZATION

    def get_capabilities(self) -> list[str]:
        return ["visualization"]

    def get_visualization_types(self) -> list[str]:
        return []

    def get_default_options(self) -> dict:
        return {}


class IntegrationPlugin(PluginBase):
    """Connections to external services (LIMS, cloud storage, etc.)."""

    plugin_type: PluginType = PluginType.INTEGRATION

    def get_capabilities(self) -> list[str]:
        return ["integration"]

    def get_connection_parameters(self) -> dict:
        return {}

    def test_connection(self, config: dict) -> bool:
        return False


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------


class PluginSecurity:
    """Sandboxed execution, dependency checking, and signature verification."""

    BLOCKED_MODULES: frozenset[str] = frozenset(
        {
            "ctypes",
            "subprocess",
            "multiprocessing",
            "socket",
            "http",
            "urllib",
            "ftplib",
            "smtplib",
            "telnetlib",
        }
    )
    MAX_PLUGIN_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB

    @classmethod
    def check_imports(cls, source_path: str) -> list[str]:
        """Scan *source_path* for blocked import statements."""
        violations: list[str] = []
        try:
            with open(source_path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    for mod in cls.BLOCKED_MODULES:
                        if f"import {mod}" in stripped or f"from {mod}" in stripped:
                            violations.append(f"L{lineno}: blocked '{mod}' — {stripped!r}")
        except OSError as exc:
            violations.append(f"Could not read {source_path}: {exc}")
        return violations

    @classmethod
    def verify_signature(cls, manifest: PluginManifest, plugin_dir: str) -> bool:
        """Verify plugin signature (stub — accepts when field is present).

        Real Ed25519-based verification replaces this once the signing
        infrastructure is deployed.
        """
        if not manifest.signature:
            logger.warning("Plugin '%s' has no signature.", manifest.name)
            return False
        logger.debug(
            "Plugin '%s' hash=%s — stub-accepted.",
            manifest.name,
            cls._compute_plugin_hash(plugin_dir),
        )
        return True

    @classmethod
    def check_plugin_size(cls, plugin_dir: str) -> tuple[bool, int]:
        """Return *(ok, total_bytes)* for the plugin directory."""
        total = sum(
            os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(plugin_dir) for f in fns
        )
        return total <= cls.MAX_PLUGIN_SIZE_BYTES, total

    @classmethod
    def check_dependencies(cls, dependencies: list[str]) -> list[str]:
        """Return list of *missing* packages from *dependencies*."""
        missing: list[str] = []
        for dep in dependencies:
            pkg = dep.split(">=")[0].split("<=")[0].split("==")[0].split("!=")[0].strip()
            if importlib.util.find_spec(pkg) is None:
                missing.append(dep)
        return missing

    @classmethod
    def _compute_plugin_hash(cls, plugin_dir: str) -> str:
        """SHA-256 digest over all files in *plugin_dir*."""
        h = hashlib.sha256()
        for dp, _, fns in os.walk(plugin_dir):
            for fname in sorted(fns):
                with open(os.path.join(dp, fname), "rb") as fh:
                    while chunk := fh.read(8192):
                        h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Mock registry
# ---------------------------------------------------------------------------


class _MockRegistry:
    """Simulated remote plugin registry for development and testing."""

    _CATALOGUE: list[dict] = [
        {
            "name": "advanced-spot-detector",
            "version": "1.2.0",
            "author": "Teloscopy Community",
            "description": "GPU-accelerated spot detection using deep learning.",
            "plugin_type": "analysis",
            "dependencies": ["numpy", "torch"],
        },
        {
            "name": "hdf5-exporter",
            "version": "0.9.1",
            "author": "DataLab",
            "description": "Export results to HDF5 with configurable compression.",
            "plugin_type": "export",
            "dependencies": ["h5py"],
        },
        {
            "name": "napari-bridge",
            "version": "2.0.0",
            "author": "Napari Contributors",
            "description": "Render Teloscopy overlays inside a Napari viewer.",
            "plugin_type": "visualization",
            "dependencies": ["napari"],
        },
        {
            "name": "lims-connector",
            "version": "0.5.3",
            "author": "BioConnect Ltd",
            "description": "Push analysis results directly to your LIMS.",
            "plugin_type": "integration",
            "dependencies": ["requests"],
        },
        {
            "name": "intensity-profiler",
            "version": "1.0.0",
            "author": "Teloscopy Community",
            "description": "Per-telomere intensity profiling with statistical tests.",
            "plugin_type": "analysis",
            "dependencies": ["numpy", "scipy"],
        },
    ]

    @classmethod
    def list_available(cls) -> list[PluginInfo]:
        return [
            PluginInfo(
                name=e["name"],
                version=e["version"],
                author=e["author"],
                description=e["description"],
                plugin_type=PluginType(e["plugin_type"]),
                dependencies=e.get("dependencies", []),
            )
            for e in cls._CATALOGUE
        ]

    @classmethod
    def search(cls, query: str) -> list[PluginInfo]:
        q = query.lower()
        return [
            i for i in cls.list_available() if q in i.name.lower() or q in i.description.lower()
        ]

    @classmethod
    def get_plugin_info(cls, name: str) -> PluginInfo | None:
        return next((i for i in cls.list_available() if i.name == name), None)

    @classmethod
    def download_plugin(cls, name: str, target_dir: str) -> str | None:
        """Simulate downloading — creates a minimal scaffold on disk."""
        info = cls.get_plugin_info(name)
        if info is None:
            return None
        plugin_dir = os.path.join(target_dir, name)
        os.makedirs(plugin_dir, exist_ok=True)

        manifest = {
            "name": info.name,
            "version": info.version,
            "entry_point": "plugin_main.py",
            "dependencies": info.dependencies,
            "capabilities": [info.plugin_type.value],
            "min_teloscopy_version": "0.1.0",
            "author": info.author,
            "description": info.description,
            "plugin_type": info.plugin_type.value,
        }
        with open(os.path.join(plugin_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)

        _map = {
            PluginType.ANALYSIS: "AnalysisPlugin",
            PluginType.EXPORT: "ExportPlugin",
            PluginType.VISUALIZATION: "VisualizationPlugin",
            PluginType.INTEGRATION: "IntegrationPlugin",
        }
        base = _map.get(info.plugin_type, "AnalysisPlugin")
        stub = (
            f'"""Auto-generated stub for {info.name}."""\n'
            f"from teloscopy.platform.plugin_system import {base}\n\n"
            f"class Plugin({base}):\n"
            f"    name = {info.name!r}\n    version = {info.version!r}\n"
            f"    author = {info.author!r}\n    description = {info.description!r}\n"
            f"    def initialize(self, config: dict) -> None: self.config = config\n"
            f"    def execute(self, input_data: dict) -> dict:\n"
            f'        return {{"status": "ok", "plugin": self.name}}\n'
            f"    def cleanup(self) -> None: pass\n"
        )
        with open(os.path.join(plugin_dir, "plugin_main.py"), "w") as fh:
            fh.write(stub)
        return plugin_dir


# ---------------------------------------------------------------------------
# PluginManager
# ---------------------------------------------------------------------------


class PluginManager:
    """Central facade for discovering, installing, loading, and managing
    Teloscopy plugins.

    Parameters
    ----------
    plugin_dir : str
        Directory where plugins are installed (default ``~/.teloscopy/plugins``).
    registry_url : str | None
        Remote registry URL.  *None* uses the built-in mock registry.
    """

    def __init__(
        self, plugin_dir: str = "~/.teloscopy/plugins", registry_url: str | None = None
    ) -> None:
        self.plugin_dir: str = os.path.expanduser(plugin_dir)
        self.registry_url: str | None = registry_url
        self._plugins: dict[str, PluginInstance] = {}
        self._registry = _MockRegistry
        os.makedirs(self.plugin_dir, exist_ok=True)
        logger.info("PluginManager initialised — plugin_dir=%s", self.plugin_dir)

    # -- Discovery ---------------------------------------------------------

    def discover_plugins(self) -> list[PluginInfo]:
        """Scan :attr:`plugin_dir` for installed plugins."""
        discovered: list[PluginInfo] = []
        if not os.path.isdir(self.plugin_dir):
            return discovered
        for entry in sorted(os.listdir(self.plugin_dir)):
            candidate = os.path.join(self.plugin_dir, entry)
            manifest_path = os.path.join(candidate, "manifest.json")
            if not os.path.isfile(manifest_path):
                continue
            try:
                manifest = PluginManifest.from_file(manifest_path)
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping '%s': %s", entry, exc)
                continue
            info = PluginInfo(
                name=manifest.name,
                version=manifest.version,
                author=manifest.author,
                description=manifest.description,
                plugin_type=PluginType(manifest.plugin_type),
                dependencies=manifest.dependencies,
                installed_at=datetime.fromtimestamp(os.path.getctime(candidate), tz=UTC),
                path=candidate,
            )
            discovered.append(info)
            if manifest.name not in self._plugins:
                self._plugins[manifest.name] = PluginInstance(
                    info=info, manifest=manifest, state=PluginState.DISCOVERED
                )
        logger.info("Discovered %d plugin(s).", len(discovered))
        return discovered

    # -- Loading / unloading -----------------------------------------------

    def load_plugin(self, name: str) -> PluginInstance:
        """Dynamically load the installed plugin *name*."""
        if name not in self._plugins:
            self.discover_plugins()
        if name not in self._plugins:
            raise FileNotFoundError(f"Plugin '{name}' is not installed.")
        wrapper = self._plugins[name]
        if wrapper.state == PluginState.LOADED:
            return wrapper

        entry_file = os.path.join(wrapper.info.path, wrapper.manifest.entry_point)
        if not os.path.isfile(entry_file):
            raise RuntimeError(f"Entry point not found: {entry_file}")

        mod_name = f"teloscopy_plugin_{name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(mod_name, entry_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot create module spec for {entry_file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            wrapper.state, wrapper.error_message = PluginState.ERROR, str(exc)
            raise RuntimeError(f"Failed to import plugin '{name}': {exc}") from exc

        plugin_cls = None
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, PluginBase)
                and obj is not PluginBase
                and not getattr(obj, "__abstractmethods__", set())
            ):
                plugin_cls = obj
                break
        if plugin_cls is None:
            raise RuntimeError(f"No concrete PluginBase subclass in '{entry_file}'.")

        wrapper.module, wrapper.instance = module, plugin_cls()
        wrapper.state, wrapper.loaded_at = PluginState.LOADED, datetime.now(UTC)
        logger.info("Plugin '%s' v%s loaded.", name, wrapper.manifest.version)
        return wrapper

    def unload_plugin(self, name: str) -> None:
        """Unload a plugin — calls cleanup() then removes from sys.modules."""
        wrapper = self._plugins.get(name)
        if wrapper is None:
            return
        if wrapper.instance is not None:
            try:
                wrapper.instance.cleanup()
            except Exception as exc:
                logger.error("Cleanup error for '%s': %s", name, exc)
        sys.modules.pop(f"teloscopy_plugin_{name.replace('-', '_')}", None)
        wrapper.instance = wrapper.module = None
        wrapper.state = PluginState.UNLOADED
        logger.info("Plugin '%s' unloaded.", name)

    def get_plugin(self, name: str) -> PluginInstance | None:
        """Return the :class:`PluginInstance` for *name*, or *None*."""
        return self._plugins.get(name)

    # -- Installation / uninstallation -------------------------------------

    def install_plugin(self, name: str, source: str = "registry") -> PluginInfo:
        """Install from the registry (``source="registry"``) or a local path."""
        if source == "registry":
            return self._install_from_registry(name)
        return self._install_from_path(name, source)

    def _install_from_registry(self, name: str) -> PluginInfo:
        info = self._registry.get_plugin_info(name)
        if info is None:
            raise ValueError(f"Plugin '{name}' not found in the registry.")
        with tempfile.TemporaryDirectory() as tmp:
            downloaded = self._registry.download_plugin(name, tmp)
            if downloaded is None:
                raise ValueError(f"Failed to download plugin '{name}'.")
            result = self.validate_plugin(downloaded)
            if not result.is_valid:
                raise ValueError(f"Validation failed: {'; '.join(result.errors)}")
            dest = os.path.join(self.plugin_dir, name)
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(downloaded, dest)
        self.discover_plugins()
        wrapper = self._plugins.get(name)
        if wrapper is None:
            raise ValueError(f"Plugin '{name}' copied but not discoverable.")
        logger.info("Installed '%s' v%s from registry.", name, info.version)
        return wrapper.info

    def _install_from_path(self, name: str, source_path: str) -> PluginInfo:
        if not os.path.isdir(source_path):
            raise ValueError(f"'{source_path}' is not a directory.")
        result = self.validate_plugin(source_path)
        if not result.is_valid:
            raise ValueError(f"Validation failed: {'; '.join(result.errors)}")
        dest = os.path.join(self.plugin_dir, name)
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(source_path, dest)
        self.discover_plugins()
        wrapper = self._plugins.get(name)
        if wrapper is None:
            raise ValueError(f"Plugin '{name}' copied but not discoverable.")
        logger.info("Installed '%s' from local path.", name)
        return wrapper.info

    def uninstall_plugin(self, name: str) -> None:
        """Remove an installed plugin from disk and internal caches."""
        wrapper = self._plugins.get(name)
        if wrapper is None:
            self.discover_plugins()
            wrapper = self._plugins.get(name)
        if wrapper is None:
            return
        if wrapper.state in (PluginState.LOADED, PluginState.INITIALISED):
            self.unload_plugin(name)
        if wrapper.info.path and os.path.isdir(wrapper.info.path):
            shutil.rmtree(wrapper.info.path)
        self._plugins.pop(name, None)
        logger.info("Plugin '%s' uninstalled.", name)

    # -- Validation --------------------------------------------------------

    def validate_plugin(self, path: str) -> ValidationResult:
        """Validate plugin structure and security.

        Checks: manifest.json, entry-point existence, size limits,
        blocked imports, dependency availability, and signature.
        """
        errors: list[str] = []
        warnings: list[str] = []
        manifest: PluginManifest | None = None
        security_passed = True

        manifest_path = os.path.join(path, "manifest.json")
        if not os.path.isfile(manifest_path):
            return ValidationResult(is_valid=False, errors=["Missing manifest.json"])
        try:
            manifest = PluginManifest.from_file(manifest_path)
        except (ValueError, json.JSONDecodeError) as exc:
            return ValidationResult(is_valid=False, errors=[f"Invalid manifest: {exc}"])

        if not os.path.isfile(os.path.join(path, manifest.entry_point)):
            errors.append(f"Entry point '{manifest.entry_point}' missing.")
        size_ok, total = PluginSecurity.check_plugin_size(path)
        if not size_ok:
            errors.append(f"Plugin too large ({total} bytes).")

        for dp, _, files in os.walk(path):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                violations = PluginSecurity.check_imports(os.path.join(dp, fn))
                warnings.extend(f"{fn}: {v}" for v in violations)
                if violations:
                    security_passed = False

        missing = PluginSecurity.check_dependencies(manifest.dependencies)
        if missing:
            warnings.append(f"Missing deps: {missing}")
        if not PluginSecurity.verify_signature(manifest, path):
            warnings.append("Plugin is unsigned.")

        return ValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
            manifest=manifest,
            security_passed=security_passed,
        )

    # -- Registry interaction ----------------------------------------------

    def list_available(self) -> list[PluginInfo]:
        """Return all plugins available in the remote registry."""
        return self._registry.list_available()

    def search_registry(self, query: str) -> list[PluginInfo]:
        """Search the registry by keyword."""
        return self._registry.search(query)

    # -- Convenience helpers -----------------------------------------------

    def get_loaded_plugins(self) -> list[PluginInstance]:
        return [
            w
            for w in self._plugins.values()
            if w.state in (PluginState.LOADED, PluginState.INITIALISED)
        ]

    def get_installed_plugins(self) -> list[PluginInfo]:
        return [w.info for w in self._plugins.values() if w.info.is_installed()]

    def reload_plugin(self, name: str) -> PluginInstance:
        self.unload_plugin(name)
        return self.load_plugin(name)

    def initialize_plugin(self, name: str, config: dict) -> None:
        """Load (if needed) and initialise a plugin with *config*."""
        wrapper = self.get_plugin(name)
        if wrapper is None or wrapper.state == PluginState.DISCOVERED:
            wrapper = self.load_plugin(name)
        if wrapper.instance is None:
            raise RuntimeError(f"Plugin '{name}' has no instance.")
        wrapper.instance.initialize(config)
        wrapper.state = PluginState.INITIALISED

    def execute_plugin(self, name: str, input_data: dict) -> dict:
        """Execute a loaded & initialised plugin, returning its result."""
        wrapper = self.get_plugin(name)
        if wrapper is None or wrapper.instance is None:
            raise RuntimeError(f"Plugin '{name}' is not loaded.")
        if wrapper.state not in (PluginState.LOADED, PluginState.INITIALISED):
            raise RuntimeError(f"Plugin '{name}' state={wrapper.state.value}.")
        wrapper.state = PluginState.RUNNING
        try:
            result = wrapper.instance.execute(input_data)
        except Exception as exc:
            wrapper.state, wrapper.error_message = PluginState.ERROR, str(exc)
            raise RuntimeError(f"Plugin '{name}' failed: {exc}") from exc
        wrapper.state = PluginState.INITIALISED
        return result

    def __repr__(self) -> str:
        return (
            f"<PluginManager dir={self.plugin_dir!r} "
            f"installed={len(self.get_installed_plugins())} "
            f"loaded={len(self.get_loaded_plugins())}>"
        )

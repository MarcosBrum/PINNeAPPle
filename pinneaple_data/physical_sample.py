from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import xarray as xr


@dataclass
class PhysicalSample:
    """
    Unified Physical Sample (UPD-aligned) used across Pinneaple.

    - state: xr.Dataset (preferred for gridded data) OR dict-like for non-grid.
    - geometry: GeometryAsset (optional).
    - schema: governing equations, IC/BC, forcings, units policy, etc.
    - domain: how to interpret sample (grid / mesh / graph / points).
    - provenance: lineage (uid, source, query, tiling, time span, etc.)
    - extras: extensibility (feature caches, mesh labels, sdf, etc.)
    """

    state: xr.Dataset | Mapping[str, Any]
    geometry: Any | None = None
    schema: Mapping[str, Any] = field(default_factory=dict)
    domain: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    # -------------------------
    # Domain helpers
    # -------------------------
    def domain_type(self) -> str:
        t = (self.domain or {}).get("type")
        if t:
            return str(t).lower()
        return "grid" if isinstance(self.state, xr.Dataset) else "unknown"

    def is_grid(self) -> bool:
        return self.domain_type() == "grid"

    def is_mesh(self) -> bool:
        return self.domain_type() == "mesh"

    def is_graph(self) -> bool:
        return self.domain_type() == "graph"

    # -------------------------
    # Introspection
    # -------------------------
    def list_variables(self) -> list[str]:
        if isinstance(self.state, xr.Dataset):
            return list(self.state.data_vars)
        if isinstance(self.state, dict):
            # common case: {"T2M": tensor, ...}
            return [str(k) for k in self.state.keys()]
        return []

    def summary(self) -> dict[str, Any]:
        out = {
            "domain_type": self.domain_type(),
            "has_geometry": self.geometry is not None,
            "schema_keys": sorted(list(self.schema.keys())),
            "provenance_keys": sorted(list(self.provenance.keys())),
            "extras_keys": sorted(list(self.extras.keys())),
        }
        if isinstance(self.state, xr.Dataset):
            out["state"] = {
                "coords": list(self.state.coords),
                "dims": {k: int(v) for k, v in self.state.sizes.items()},
                "vars": list(self.state.data_vars),
            }
        else:
            out["state"] = {
                "type": type(self.state).__name__,
                "vars": self.list_variables(),
            }
        return out

    # -------------------------
    # Training bridge
    # -------------------------
    def to_train_dict(
        self,
        *,
        x_vars: Iterable[str],
        y_vars: Iterable[str] | None = None,
        coords: Iterable[str] | None = None,
        time_dim: str | None = None,
    ) -> dict[str, Any]:
        """
        Converts this sample into a canonical training dict:
          {"x": <tensor>, "y": <tensor?>, "coords": {...}, "meta": {...}}

        Notes:
        - For xr.Dataset, variables are stacked in last dim.
        - For dict-state, assumes each var is already array/tensor-like and stackable.
        """
        x_vars = list(x_vars)
        y_vars = list(y_vars) if y_vars is not None else []
        coords = list(coords) if coords is not None else []

        def _stack_from_xr(ds: xr.Dataset, vars_: list[str]):
            arrs = []
            for v in vars_:
                da = ds[v]
                # if time_dim is given, ensure it is first dimension
                if time_dim and time_dim in da.dims:
                    da = da.transpose(time_dim, ...)
                arrs.append(da.data)  # numpy-like
            # stack on last dim
            import numpy as np

            return np.stack(arrs, axis=-1)

        def _stack_from_dict(d: dict[str, Any], vars_: list[str]):
            import numpy as np

            arrs = [d[v] for v in vars_]
            # if tensors, torch.stack works; if numpy, np.stack works
            if hasattr(arrs[0], "shape") and "torch" in str(type(arrs[0])):
                import torch

                return torch.stack(arrs, dim=-1)
            return np.stack(arrs, axis=-1)

        if isinstance(self.state, xr.Dataset):
            x = _stack_from_xr(self.state, x_vars)
            y = _stack_from_xr(self.state, y_vars) if y_vars else None
            c = {k: self.state.coords[k].data for k in coords if k in self.state.coords}
        else:
            assert isinstance(self.state, dict), "state must be xr.Dataset or dict"
            x = _stack_from_dict(self.state, x_vars)
            y = _stack_from_dict(self.state, y_vars) if y_vars else None
            c = {}

        out = {
            "x": x,
            "coords": c,
            "schema": self.schema,
            "domain": self.domain,
            "provenance": self.provenance,
            "geometry": self.geometry,
        }
        if y is not None:
            out["y"] = y
        return out

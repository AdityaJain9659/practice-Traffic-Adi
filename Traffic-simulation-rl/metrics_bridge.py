#!/usr/bin/env python3
"""
Dashboard Bridge
================

Bridges metrics from the RL + SUMO training to the Streamlit dashboard JSON
in Repo B, matching its exact schema (per README and sample data).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class DashboardBridge:
    """
    Writes metrics to the dashboard JSON with the exact schema expected by
    AI-Powered-Traffic-Management-System- (Repo B):

    {
      "ts": int,
      "avg_travel_time": float | null,
      "avg_wait_time": float | null,
      "vehicles_in_system": int | null,
      "baseline_avg_travel_time": float | null,
      "selected_intersection": str,
      "intersections": {
        "<id>": { "current_phase": int, "queues": [int,int,int,int], "name": str },
        ...
      },
      "time_series": {
        "t": [int,...],
        "rl_avg_travel_time": [float|null,...],
        "baseline_avg_travel_time": [float|null,...]
      },
      "latest_frame_path": str,
      "traffic_phases": { "0": str, ... }
    }
    """

    def __init__(self, output_path: Optional[str] = None) -> None:
        # Default to Repo B sibling path ../AI-Powered-Traffic-Management-System-/data/dashboard_data.json
        if output_path is None:
            target = (
                Path(__file__).resolve().parent
                / ".."
                / "AI-Powered-Traffic-Management-System-"
                / "data"
                / "dashboard_data.json"
            )
        else:
            target = Path(output_path)

        self.output_path: Path = target.resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._data: Dict[str, Any] = {
            "ts": 0,
            "avg_travel_time": None,
            "avg_wait_time": None,
            "vehicles_in_system": None,
            "baseline_avg_travel_time": None,
            "selected_intersection": "",
            "intersections": {},
            "time_series": {
                "t": [],
                "rl_avg_travel_time": [],
                "baseline_avg_travel_time": [],
            },
            "latest_frame_path": "",
            "traffic_phases": {
                "0": "North-South Green",
                "1": "East-West Green",
                "2": "All Red (Transition)",
                "3": "North-South Yellow",
                "4": "East-West Yellow",
            },
        }

        # Initialize file so dashboard can start reading
        self._write()

    def set_traffic_phases(self, mapping: Dict[int, str]) -> None:
        self._data["traffic_phases"] = {str(k): str(v) for k, v in mapping.items()}
        self._write()

    def push_step(
        self,
        *,
        t: int,
        avg_wait_time: Optional[float],
        intersections: Dict[str, Dict[str, Any]],
        vehicles_in_system: Optional[int],
        rl_avg_travel_time: Optional[float] = None,
        baseline_avg_travel_time: Optional[float] = None,
        selected_intersection: Optional[str] = None,
    ) -> None:
        """
        Update the dashboard JSON for one step/episode.
        """
        # Top-level snapshot
        self._data["ts"] = int(t)
        self._data["avg_travel_time"] = (
            float(rl_avg_travel_time) if rl_avg_travel_time is not None else None
        )
        self._data["avg_wait_time"] = float(avg_wait_time) if avg_wait_time is not None else None
        self._data["vehicles_in_system"] = (
            int(vehicles_in_system) if vehicles_in_system is not None else None
        )
        self._data["baseline_avg_travel_time"] = (
            float(baseline_avg_travel_time) if baseline_avg_travel_time is not None else None
        )
        self._data["intersections"] = intersections or {}

        # Selected intersection default: first key
        if selected_intersection is not None:
            self._data["selected_intersection"] = selected_intersection
        else:
            self._data["selected_intersection"] = next(iter(self._data["intersections"].keys()), "")

        # Time series arrays
        self._data["time_series"]["t"].append(int(t))
        self._data["time_series"]["rl_avg_travel_time"].append(
            float(rl_avg_travel_time) if rl_avg_travel_time is not None else None
        )
        self._data["time_series"]["baseline_avg_travel_time"].append(
            float(baseline_avg_travel_time) if baseline_avg_travel_time is not None else None
        )

        self._write()

    def _write(self) -> None:
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp_path), str(self.output_path))



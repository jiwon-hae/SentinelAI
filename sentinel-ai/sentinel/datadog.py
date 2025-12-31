from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import time
import requests


@dataclass
class DatadogConfig:
    api_key: str
    site: str = "datadoghq.com"
    service: str = "sentinel-ai"
    env: str = "demo"


class DatadogClient:
    """
    Sends:  
      - Custom metrics via Datadog v2 metrics endpoint
      - Logs via Datadog v2 logs endpoint

    Metric types:
      - gauge: instantaneous values (latency_ms, hallucination_rate)
      - count: event counts (request_count, error_count, good_count)
    """
    def __init__(self, cfg: DatadogConfig):
        self.cfg = cfg
        self.metrics_url = f"https://api.{cfg.site}/api/v2/series"
        self.logs_url = f"https://http-intake.logs.{cfg.site}/api/v2/logs"
        self.headers = {
            "DD-API-KEY": cfg.api_key,
            "Content-Type": "application/json",
        }

    def send_metric(self, name: str, value: float, *, tags: Optional[List[str]] = None, metric_type: str = "gauge") -> None:
        ts = int(time.time())
        dd_type = 0 if metric_type == "gauge" else 1  # 0=gauge, 1=count

        payload = {
            "series": [{
                "metric": name,
                "type": dd_type,
                "points": [{"timestamp": ts, "value": float(value)}],
                "tags": tags or [],
            }]
        }
        r = requests.post(self.metrics_url, headers=self.headers, json=payload, timeout=10)
        r.raise_for_status()

    def send_log(self, log: Dict[str, Any]) -> None:
        r = requests.post(self.logs_url, headers=self.headers, json=[log], timeout=10)
        r.raise_for_status()
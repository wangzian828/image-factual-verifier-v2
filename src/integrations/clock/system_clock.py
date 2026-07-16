from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict


@dataclass
class SystemClockClient:
    def now(self) -> Dict[str, str]:
        current = datetime.now().astimezone()
        override = os.getenv("IFV_RUNTIME_DATE", "").strip()
        if override:
            runtime_date = date.fromisoformat(override)
            current = current.replace(
                year=runtime_date.year,
                month=runtime_date.month,
                day=runtime_date.day,
            )
        return {
            "current_date": current.date().isoformat(),
            "current_datetime": current.isoformat(),
            "timezone": str(current.tzinfo),
            "source": (
                "runtime_date_override"
                if override
                else "system_clock"
            ),
        }

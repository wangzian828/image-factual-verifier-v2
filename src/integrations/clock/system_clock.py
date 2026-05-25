from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class SystemClockClient:
    def now(self) -> Dict[str, str]:
        current = datetime.now().astimezone()
        return {
            "current_date": current.date().isoformat(),
            "current_datetime": current.isoformat(),
            "timezone": str(current.tzinfo),
            "source": "system_clock",
        }

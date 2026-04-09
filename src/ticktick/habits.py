"""TickTick Habits — V2 API経由で習慣情報を取得."""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ticktick_sdk import TickTickClient

JST = ZoneInfo("Asia/Tokyo")
_WEEKDAY_MAP = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]

logger = logging.getLogger(__name__)
TOKEN_FILE = Path(os.environ.get("TOKEN_FILE", ".tokens.json"))


def _run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def _get_v1_token() -> str | None:
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        return data.get("access_token")
    return None


def _is_today_habit(repeat_rule: str | None) -> bool:
    if not repeat_rule:
        return True
    today_byday = _WEEKDAY_MAP[datetime.now(JST).weekday()]
    byday_match = re.search(r"BYDAY=([A-Z,]+)", repeat_rule)
    if not byday_match:
        return True
    return today_byday in byday_match.group(1).split(",")


async def _fetch_habits() -> list[dict]:
    v1_token = _get_v1_token()
    client = TickTickClient(
        client_id=os.environ["TICKTICK_CLIENT_ID"],
        client_secret=os.environ["TICKTICK_CLIENT_SECRET"],
        username=os.environ.get("TICKTICK_USERNAME", ""),
        password=os.environ.get("TICKTICK_PASSWORD", ""),
        v1_access_token=v1_token,
    )
    try:
        await client.connect()
        habits = await client.get_all_habits()
        today_stamp = int(datetime.now(JST).strftime("%Y%m%d"))
        habit_ids = [h.id for h in habits]
        checkins_dict = await client.get_habit_checkins(habit_ids) if habit_ids else {}
        result = []
        for h in habits:
            if not _is_today_habit(getattr(h, "repeat_rule", None)):
                continue
            checkins = checkins_dict.get(h.id, [])
            checked_today = any(c.checkin_stamp == today_stamp for c in checkins)
            result.append({
                "id": h.id,
                "name": h.name,
                "status": getattr(h, "status", None),
                "goal": getattr(h, "goal", None),
                "frequency": getattr(h, "frequency", None),
                "checked_today": checked_today,
            })
        return result
    finally:
        await client.disconnect()


def get_habits() -> list[dict]:
    if not os.environ.get("TICKTICK_USERNAME") or not os.environ.get("TICKTICK_PASSWORD"):
        logger.info("TickTick V2 credentials not set, skipping habits")
        return []
    try:
        return _run_async(_fetch_habits())
    except Exception:
        logger.exception("Failed to fetch habits")
        return []

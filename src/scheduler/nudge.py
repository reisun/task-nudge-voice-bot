"""定時Nudgeスケジューラー — 指定時間帯に毎時タスク通知を発話."""

import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


def create_scheduler(on_nudge: callable) -> BackgroundScheduler:
    """定時Nudge用のスケジューラーを作成.

    on_nudge() は毎時呼ばれ、会話セッションを開始するコールバック。
    """
    start_hour = int(os.environ.get("NUDGE_START_HOUR", "7"))
    end_hour = int(os.environ.get("NUDGE_END_HOUR", "23"))

    scheduler = BackgroundScheduler(timezone="Asia/Tokyo")
    scheduler.add_job(
        on_nudge,
        CronTrigger(hour=f"{start_hour}-{end_hour}", minute=0, timezone="Asia/Tokyo"),
        name="Hourly voice nudge",
    )
    logger.info("Scheduler configured: every hour %d:00-%d:00 (Asia/Tokyo)",
                start_hour, end_hour)
    return scheduler

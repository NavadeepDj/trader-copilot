"""NSE market status, trading hours, and holiday calendar."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Dict

IST = timezone(timedelta(hours=5, minutes=30))

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

NSE_HOLIDAYS_2025 = {
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Id-Ul-Fitr (Ramadan Eid)
    date(2025, 4, 10),   # Shri Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 6, 7),    # Bakri Id (Eid ul-Adha)
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 16),   # Ashura
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Mahatma Gandhi Jayanti
    date(2025, 10, 21),  # Diwali (Lakshmi Puja)
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
}

NSE_HOLIDAYS_2026 = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 17),   # Mahashivratri
    date(2026, 3, 3),    # Holi
    date(2026, 3, 20),   # Id-Ul-Fitr (Ramadan Eid)
    date(2026, 3, 30),   # Shri Ram Navami
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 16),   # Buddha Pournima
    date(2026, 5, 28),   # Bakri Id (Eid ul-Adha)
    date(2026, 6, 27),   # Muharram
    date(2026, 8, 15),   # Independence Day
    date(2026, 8, 27),   # Milad-un-Nabi (Prophet Mohammed Birthday)
    date(2026, 9, 16),   # Ganesh Chaturthi (tentative)
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 10, 9),   # Dussehra
    date(2026, 11, 9),   # Diwali (Lakshmi Puja) (tentative)
    date(2026, 11, 10),  # Diwali Balipratipada (tentative)
    date(2026, 11, 27),  # Guru Nanak Jayanti
    date(2026, 12, 25),  # Christmas
}

ALL_HOLIDAYS = NSE_HOLIDAYS_2025 | NSE_HOLIDAYS_2026


def _is_trading_day(d: date) -> bool:
    if d.weekday() >= 5:
        return False
    if d in ALL_HOLIDAYS:
        return False
    return True


def _next_trading_day(from_date: date) -> date:
    d = from_date + timedelta(days=1)
    while not _is_trading_day(d):
        d += timedelta(days=1)
    return d


def _prev_trading_day(from_date: date) -> date:
    d = from_date - timedelta(days=1)
    while not _is_trading_day(d):
        d -= timedelta(days=1)
    return d


def get_market_status() -> Dict:
    """Get the current NSE market status, trading hours, and next trading day.

    Returns:
        dict with market status (OPEN/CLOSED/PRE_MARKET/POST_MARKET),
        current IST time, whether today is a trading day, next trading day,
        and previous trading day.
    """
    now_ist = datetime.now(IST)
    today = now_ist.date()
    current_time = now_ist.time()

    is_today_trading = _is_trading_day(today)
    next_td = _next_trading_day(today)
    prev_td = _prev_trading_day(today)

    if not is_today_trading:
        if today.weekday() >= 5:
            reason = f"Weekend ({today.strftime('%A')})"
        else:
            reason = "NSE holiday"
        status = "CLOSED"
    elif current_time < MARKET_OPEN:
        status = "PRE_MARKET"
        reason = f"Market opens at 9:15 AM IST (in {_time_diff(current_time, MARKET_OPEN)})"
    elif current_time > MARKET_CLOSE:
        status = "POST_MARKET"
        reason = "Market closed at 3:30 PM IST today"
    else:
        status = "OPEN"
        reason = f"Market closes at 3:30 PM IST (in {_time_diff(current_time, MARKET_CLOSE)})"

    return {
        "status": "success",
        "market_status": status,
        "reason": reason,
        "current_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST (%A)"),
        "is_trading_day": is_today_trading,
        "next_trading_day": next_td.strftime("%Y-%m-%d (%A)"),
        "previous_trading_day": prev_td.strftime("%Y-%m-%d (%A)"),
        "market_hours": "9:15 AM - 3:30 PM IST",
    }


def _time_diff(t1: time, t2: time) -> str:
    d1 = timedelta(hours=t1.hour, minutes=t1.minute)
    d2 = timedelta(hours=t2.hour, minutes=t2.minute)
    diff = d2 - d1
    hours, remainder = divmod(int(diff.total_seconds()), 3600)
    minutes = remainder // 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"

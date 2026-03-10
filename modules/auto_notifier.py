"""
Auto Notifier — tự động fetch data lúc 14:55, gửi Discord lúc 15:00
Chạy trong background thread, không block Streamlit.

Cách dùng trong app.py:
    from modules import auto_notifier
    auto_notifier.start(session_state=st.session_state)
"""

import threading
import time
import logging
from datetime import datetime, date, timedelta

logger = logging.getLogger(__name__)

# ── State ────────────────────────────────────────────────
_thread: threading.Thread | None = None
_last_fetched_date: date | None = None
_last_sent_date:    date | None = None
_lock = threading.Lock()

# ── Config ───────────────────────────────────────────────
FETCH_HOUR,   FETCH_MINUTE   = 14, 55   # tự động fetch data
SEND_HOUR,    SEND_MINUTE    = 15,  0   # gửi Discord
CHECK_EVERY   = 20                      # kiểm tra mỗi 20 giây

DEFAULT_EXCHANGE   = "ALL"
DEFAULT_MA_PERIODS = [10, 20, 50]
DEFAULT_LOOKBACK   = 60
DEFAULT_MIN_LIQ    = 1.0    # Bn VND


# ── Helpers ──────────────────────────────────────────────

def _is_weekday(now: datetime) -> bool:
    return now.weekday() < 5

def _past_time(now: datetime, hour: int, minute: int) -> bool:
    return (now.hour, now.minute) >= (hour, minute)

def _should_fetch(now: datetime) -> bool:
    with _lock:
        already = _last_fetched_date == now.date()
    return _is_weekday(now) and _past_time(now, FETCH_HOUR, FETCH_MINUTE) and not already

def _should_send(now: datetime) -> bool:
    with _lock:
        already = _last_sent_date == now.date()
    return _is_weekday(now) and _past_time(now, SEND_HOUR, SEND_MINUTE) and not already


def _auto_fetch(session_state) -> bool:
    """Fetch full market data và lưu vào session_state['stats']."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from modules import fetcher, calculator
        from modules import tickers as tk

        logger.info("Auto-fetch bat dau luc %s", datetime.now().strftime("%H:%M:%S"))

        end_date   = date.today()
        start_date = end_date - timedelta(days=720)
        tickers_to_scan = tk.get_tickers(DEFAULT_EXCHANGE)

        # 1. Fetch aggressive → lọc thanh khoản
        agg_results = fetcher.batch_fetch_aggressive(tickers_to_scan)
        agg_liq_map = calculator.compute_liquidity_map(agg_results)
        fetcher.save_liquidity_cache(agg_liq_map)

        # 2. Filter winners
        winners = []
        for t in tickers_to_scan:
            if calculator.is_index_ticker(t):
                winners.append(t)
                continue
            if agg_liq_map.get(t, 0) >= DEFAULT_MIN_LIQ:
                winners.append(t)

        if len(winners) <= 1:
            logger.warning("Auto-fetch: khong co ticker nao pass filter")
            return False

        logger.info("Auto-fetch: %d tickers, loading history...", len(winners))

        # 3. Fetch lịch sử giá
        raw_results = fetcher.batch_fetch(
            winners,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
        )
        prices_dict = fetcher.parse_results(raw_results)

        if not prices_dict:
            logger.error("Auto-fetch: parse_results rong")
            return False

        for t, data in prices_dict.items():
            if t in agg_liq_map:
                data['avg_liquidity_bn'] = agg_liq_map[t]

        # 4. Tính stats — dùng lại hàm compute_all_stats từ app.py
        import importlib.util, pathlib
        app_path = pathlib.Path(__file__).parent.parent / 'app.py'
        spec = importlib.util.spec_from_file_location("app", app_path)
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)
        stats = app_mod.compute_all_stats(
            prices_dict, DEFAULT_MA_PERIODS, DEFAULT_LOOKBACK,
            agg_liq_map=agg_liq_map,
        )

        # 5. Lưu vào session_state
        session_state['stats']       = stats
        session_state['agg_liq_map'] = agg_liq_map
        session_state['agg_results'] = agg_results
        session_state['data_loaded'] = True

        global _last_fetched_date
        with _lock:
            _last_fetched_date = date.today()

        logger.info("Auto-fetch xong: %d tickers", len(prices_dict))
        return True

    except Exception as e:
        logger.exception("Auto-fetch error: %s", e)
        return False


def _scheduler_loop(session_state):
    """Background loop chính."""
    logger.info(
        "Auto-notifier started | fetch=%02d:%02d | send=%02d:%02d",
        FETCH_HOUR, FETCH_MINUTE, SEND_HOUR, SEND_MINUTE
    )

    while True:
        try:
            now = datetime.now()

            # Job 1: Auto-fetch 14:55
            if _should_fetch(now):
                logger.info("Den gio fetch data (%02d:%02d)", FETCH_HOUR, FETCH_MINUTE)
                if not _auto_fetch(session_state):
                    time.sleep(120)
                    continue

            # Job 2: Gửi Discord 15:00
            if _should_send(now):
                stats = session_state.get('stats')
                if stats is None:
                    logger.warning("Den gio gui nhung stats=None, thu lai sau 30s")
                    time.sleep(30)
                    continue

                logger.info("Dang gui Discord luc %s", now.strftime("%H:%M:%S"))
                from modules import notifier
                success = notifier.send_market_summary_sync(stats)

                if success:
                    global _last_sent_date
                    with _lock:
                        _last_sent_date = date.today()
                    logger.info("Gui Discord thanh cong")
                else:
                    logger.error("Gui Discord that bai, thu lai sau 5 phut")
                    time.sleep(300)
                    continue

        except Exception as e:
            logger.exception("Scheduler error: %s", e)

        time.sleep(CHECK_EVERY)


# ── Public API ────────────────────────────────────────────

def start(session_state):
    """
    Khởi động background scheduler.
    Gọi 1 lần trong app.py sau keep_alive.start_keep_alive():
        auto_notifier.start(session_state=st.session_state)
    """
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _thread = threading.Thread(
        target=_scheduler_loop,
        args=(session_state,),
        daemon=True,
        name="auto-notifier",
    )
    _thread.start()
    logger.info("Auto-notifier thread started")


def status() -> dict:
    """Trạng thái hiện tại để hiển thị trên sidebar."""
    now = datetime.now()
    with _lock:
        last_fetch = _last_fetched_date
        last_send  = _last_sent_date

    fetched_today = last_fetch == date.today() if last_fetch else False
    sent_today    = last_send  == date.today() if last_send  else False

    if not _past_time(now, FETCH_HOUR, FETCH_MINUTE):
        next_action = f"Fetch luc {FETCH_HOUR:02d}:{FETCH_MINUTE:02d}"
    elif not fetched_today:
        next_action = f"Dang cho fetch..."
    elif not _past_time(now, SEND_HOUR, SEND_MINUTE):
        next_action = f"Gui Discord luc {SEND_HOUR:02d}:{SEND_MINUTE:02d}"
    elif not sent_today:
        next_action = f"Dang cho gui..."
    else:
        next_action = f"Ngay mai {FETCH_HOUR:02d}:{FETCH_MINUTE:02d}"

    return {
        "running":       _thread is not None and _thread.is_alive(),
        "fetched_today": fetched_today,
        "sent_today":    sent_today,
        "last_fetch":    last_fetch.strftime("%Y-%m-%d") if last_fetch else "—",
        "last_send":     last_send.strftime("%Y-%m-%d")  if last_send  else "—",
        "next_action":   next_action,
    }
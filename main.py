import os
import json
import time
from datetime import datetime, time as dtime
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import schedule
import pytz

# -----------------------------
# CONFIG
# -----------------------------
# Telegram (ortam değişkeni veya doğrudan buraya koyabilirsin)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "8063930487:AAENeimw2WCdHupGCQ-o65YK4u0KXP9Q4lk")
CHAT_ID = os.environ.get("CHAT_ID", "5382853959")

# Hareket bildirimi eşiği (örnek: 1.0 => %1)
MOVEMENT_NOTIFY_DWN = float(os.environ.get("MOVEMENT_NOTIFY_DWN", 1))
MOVEMENT_NOTIFY_UP = float(os.environ.get("MOVEMENT_NOTIFY_UP", 2.5))

# Periyodik kontrol aralığı (dakika)
CHECK_INTERVAL_MINUTES = int(os.environ.get("CHECK_INTERVAL_MINUTES", 1))

# Borsa saatleri opsiyonu: True => sadece MARKET_OPEN..MARKET_CLOSE arasında kontrol yap
USE_MARKET_HOURS = os.environ.get("USE_MARKET_HOURS", "True").lower() in ("1", "true", "yes")
#USE_MARKET_HOURS = False
# Market timezone ve saatler (BIST örneği — istersen değiştir)
MARKET_TZ = os.environ.get("MARKET_TZ", "Europe/Istanbul")
MARKET_OPEN_HH = int(os.environ.get("MARKET_OPEN_HH", 9))
MARKET_OPEN_MM = int(os.environ.get("MARKET_OPEN_MM", 40))
MARKET_CLOSE_HH = int(os.environ.get("MARKET_CLOSE_HH", 18))
MARKET_CLOSE_MM = int(os.environ.get("MARKET_CLOSE_MM", 10))
MARKET_OPEN = dtime(hour=MARKET_OPEN_HH, minute=MARKET_OPEN_MM)
MARKET_CLOSE = dtime(hour=MARKET_CLOSE_HH, minute=MARKET_CLOSE_MM)

# Kaç top listesi isteriz? (her model için top N)
TOP_N = int(os.environ.get("TOP_N", 5))

# Dosya isimleri (columns.json ve data.json senin verilerin)
COLUMNS_JSON = os.environ.get("COLUMNS_JSON", "columns.json")
DATA_JSON = os.environ.get("DATA_JSON", "data.json")

# Kullanıcı bilgilendirmesi
print("CONFIG:")
print(f"  MOVEMENT_NOTIFY_DWN = {MOVEMENT_NOTIFY_DWN}%")
print(f"  MOVEMENT_NOTIFY_UP = {MOVEMENT_NOTIFY_UP}%")
print(f"  CHECK_INTERVAL_MINUTES = {CHECK_INTERVAL_MINUTES} minutes")
print(f"  USE_MARKET_HOURS = {USE_MARKET_HOURS}")
print(f"  MARKET HOURS = {MARKET_OPEN} -> {MARKET_CLOSE} ({MARKET_TZ})")
print(f"  Columns file = {COLUMNS_JSON}, Data file = {DATA_JSON}")
print("--------------------------------------------------\n")


# -----------------------------
# UTIL: Telegram & price fetch
# -----------------------------
def send_telegram_message(text: str):
    """Kısa ve güvenli Telegram gönderimi."""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.startswith("YOUR_"):
        print("[WARN] TELEGRAM_TOKEN ayarlı değil. Telegram mesajı gönderilmeyecek. Mesaj içeriği:\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print("[Telegram error]", r.status_code, r.text)
    except Exception as e:
        print("[Telegram exception]", e)


def safe_get_last_price(symbol: str):
    """Yahoo Finance üzerinden son kapanış fiyatını çek.
    Varsayılan BIST için '.IS' eklenir. Eğer sembol NASDAQ gibi ise, sembolü doğrudan kullan."""
    if not symbol:
        return None
    # Basit heuristic: eğer sembol içinde '.' veya '-' ya da büyük harfle NASDAQ/NYSE ise ayrı kullanım gerekebilir.
    # Burada senin verilerin BIST ise ".IS" ekliyoruz. İstersen sembol formatına göre değiştir.
    ticker_symbol = f"{symbol}.IS"
    try:
        t = yf.Ticker(ticker_symbol)
        hist = t.history(period="1d", interval="1d")
        if hist is None or hist.empty:
            return None
        last_close = hist["Close"].iloc[-1]
        if pd.isna(last_close):
            return None
        return float(last_close)
    except Exception as e:
        print(f"[price fetch error] {symbol}: {e}")
        return None


# -----------------------------
# ANALYZE ONCE (ilk hesaplama)
# -----------------------------
def analyze_once():
    """columns.json ve data.json okuyup combined_df oluşturur,
       Balanced/RSI skorlarını hesaplar,
       current_price ve target price hesaplayıp top listeleri döner."""
    # --- read files
    if not os.path.exists(COLUMNS_JSON):
        raise FileNotFoundError(f"{COLUMNS_JSON} bulunamadı.")
    if not os.path.exists(DATA_JSON):
        raise FileNotFoundError(f"{DATA_JSON} bulunamadı.")

    with open(COLUMNS_JSON, "r", encoding="utf-8") as f:
        cols_obj = json.load(f)
        columns = cols_obj.get("columns", [])

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data_obj = json.load(f)
        rows = data_obj.get("data", [])

    # --- map rows to dicts using columns
    rows_mapped = []
    for item in rows:
        sym = item.get("s", "")
        arr = item.get("d", [])
        # flatten nested lists if any
        flat = []
        for v in arr:
            if isinstance(v, list):
                flat.extend(v)
            else:
                flat.append(v)
        # pad/truncate to length of columns
        flat = flat[: len(columns)] + [None] * max(0, len(columns) - len(flat))
        row_dict = dict(zip(columns, flat))
        row_dict["symbol"] = sym.split(":")[-1] if sym else ""
        rows_mapped.append(row_dict)

    df = pd.DataFrame(rows_mapped)

    # --- rating map (string to numeric) ---
    rating_map = {"StrongBuy": 2.0, "Buy": 1.0, "Neutral": 0.0, "Sell": -1.0, "StrongSell": -2.0}
    for col in ["TechRating_1D", "MARating_1D", "OsRating_1D"]:
        if col in df.columns:
            df[col] = df[col].map(rating_map)

    # numeric conversions
    numeric_cols = ["RSI", "Mom", "Stoch.K", "Stoch.D", "AO", "CCI20"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- RSI special score (kısmi) ---
    rsi = df["RSI"]
    # avoid warnings for all-NaN
    rsi_score = -((rsi - 60.0).abs() / 60.0)
    rsi_bonus = np.select(
        [rsi.between(50, 70, inclusive="both"), rsi.between(70, 80, inclusive="left"), rsi > 80],
        [0.6, 0.2, -0.3],
        default=-0.1,
    )
    df["rsi_score"] = rsi_score + rsi_bonus


    # Balanced and RSI weighted
    df["BalancedScore"] = df["RSI"].fillna(0) * 0.25 + df["OsRating_1D"].fillna(0) * 0.25 + df["TechRating_1D"].fillna(0) * 0.25 + df["MARating_1D"].fillna(0) * 0.25
    df["RSIWeightedScore"] = df["RSI"].fillna(0) * 0.6 + df["OsRating_1D"].fillna(0) * 0.1333 + df["TechRating_1D"].fillna(0) * 0.1333 + df["MARating_1D"].fillna(0) * 0.1333

    # --- current price for each symbol (ilk an) ---
    unique_syms = df["symbol"].unique().tolist()
    prices_map = {}
    for s in unique_syms:
        prices_map[s] = safe_get_last_price(s)

    df["current_price"] = df["symbol"].map(prices_map)

    # --- target price calculation (model-specific) ---
    def target_by_score(price, score):
        if price is None or pd.isna(price) or score is None or pd.isna(score):
            return None
        if score >= 4:
            return price * 1.15
        elif score >= 2:
            return price * 1.08
        elif score >= 1:
            return price * 1.04
        elif score < 0:
            return price * 0.95
        else:
            return price

    df["target_price_balanced"] = df.apply(lambda r: target_by_score(r["current_price"], r["BalancedScore"]), axis=1)

    # RSI-target logic
    def target_by_rsi(price, rsi_val):
        if price is None or pd.isna(price) or rsi_val is None or pd.isna(rsi_val):
            return None
        if 50 <= rsi_val < 70:
            return price * 1.05
        elif 70 <= rsi_val < 80:
            return price * 1.08
        elif rsi_val >= 80:
            return price * 0.95
        elif 40 <= rsi_val < 50:
            return price * 1.03
        else:
            return price * 0.90

    df["target_price_rsi"] = df.apply(lambda r: target_by_rsi(r["current_price"], r["RSI"]), axis=1)

    # expected change formatting
    def format_expected_change(row, tp_col):
        try:
            price = row["current_price"]
            tp = row.get(tp_col, None)
            if price is None or tp is None or pd.isna(price) or pd.isna(tp):
                return "-"
            pct = (tp - price) / price * 100.0
            return f"{pct:.1f}%"
        except Exception:
            return "-"

    df["expected_change_balanced"] = df.apply(lambda r: format_expected_change(r, "target_price_balanced"), axis=1)
    df["expected_change_rsi"] = df.apply(lambda r: format_expected_change(r, "target_price_rsi"), axis=1)

    # --- pick top lists ---
    top_balanced = df.sort_values("BalancedScore", ascending=False).head(TOP_N)
    top_rsi = df.sort_values("RSIWeightedScore", ascending=False).head(TOP_N)

    # terminal output (detay)
    print("\n=== İlk Analiz - Top Lists (terminal output) ===")
    def print_top(df_, model_name):
        cols_to_show = ["symbol", "RSI", "BalancedScore", "current_price", f"target_price_{model_name.lower()}", f"expected_change_{model_name.lower()}"]
        print(f"\n--- {model_name} Top {TOP_N} ---")
        # bazı sütunlar eksikse esnek davran
        show_cols = [c for c in cols_to_show if c in df_.columns]
        if df_.empty:
            print("boş")
            return
        print(df_[show_cols].to_string(index=False))
    print_top(top_balanced, "balanced")
    print_top(top_rsi, "rsi")

    # Telegram initial message (tek mesajda üç liste)
    def make_initial_message(top_bal, top_rsi):
        msg = "📌 İlk analiz sonuçları — Takip edilecek hisseler (top lists):\n\n"
        for df_top, model in [ (top_bal, "Balanced"), (top_rsi, "RSI")]:
            msg += f"📊 {model} Top {TOP_N}:\n"
            if df_top.empty:
                msg += " (yok)\n\n"
                continue
            for i, r in df_top.iterrows():
                sym = r["symbol"]
                #rsi_val = f"{r['RSI']:.2f}" if pd.notna(r.get("RSI")) else "-"
                bal = f"{r['BalancedScore']:.2f}" if pd.notna(r.get("BalancedScore")) else "-"
                price = f"{r['current_price']:.4f}" if pd.notna(r.get("current_price")) else "-"
                # target col name consistent
                tp_field = f"target_price_{model.lower()}"
                tp = f"{r.get(tp_field):.4f}" if pd.notna(r.get(tp_field)) else "-"
                exp_field = f"expected_change_{model.lower()}"
                exp = r.get(exp_field, "-")
                msg += f"{i+1}. {sym} |  Bal:{bal} | Price:{price} | Target:{tp} | Δ:{exp}\n"
            msg += "\n"
        return msg

    initial_msg = make_initial_message(top_balanced, top_rsi)
    send_telegram_message(initial_msg)

    # prepare monitored dict (initial baseline + targets + flags)
    monitored = {}
    for df_top in ( top_balanced, top_rsi):
        for _, row in df_top.iterrows():
            sym = row["symbol"]
            if not sym or pd.isna(sym):
                continue
            if sym not in monitored:
                monitored[sym] = {
                    "baseline_price": float(row["current_price"]) if pd.notna(row.get("current_price")) else None,
                    "target_price_balanced": float(row["target_price_balanced"]) if pd.notna(row.get("target_price_balanced")) else None,
                    "target_price_rsi": float(row["target_price_rsi"]) if pd.notna(row.get("target_price_rsi")) else None,
                    "alerts": {"balanced": False, "rsi": False},
                    "last_movement_dir": None,  # "up"/"down"/None
                }
            else:
                # update missing targets if any
                if pd.notna(row.get("target_price_balanced")):
                    monitored[sym]["target_price_balanced"] = float(row["target_price_balanced"])
                if pd.notna(row.get("target_price_rsi")):
                    monitored[sym]["target_price_rsi"] = float(row["target_price_rsi"])

    # return full df + monitored dictionary + top lists
    top_dict = { "Balanced": top_balanced, "RSI": top_rsi}
    return df, monitored, top_dict


# -----------------------------
# Price checker factory
# -----------------------------
def create_price_checker(monitored_dict):
    def check_prices():
        tz = pytz.timezone(MARKET_TZ)
        now = datetime.now(tz)
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S %Z')}] Fiyat kontrolü başlıyor...")

        # Borsa saatleri kontrolü
        if USE_MARKET_HOURS:
            if not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
                print("⏸ Market saatleri dışında. Kontrol atlandı.")
                return

        for sym, meta in monitored_dict.items():
            latest = safe_get_last_price(sym)
            if latest is None:
                print(f"  {sym}: fiyat alınamadı.")
                continue

            baseline = meta.get("baseline_price")
            if baseline is None:
                print(f"  {sym}: baseline yok, atlandı.")
                continue

            # yüzde değişim
            pct = (latest - baseline) / baseline * 100.0

            # 📌 Model bazlı hedef fiyat kontrolü
            for mkey, tkey, label in [
                ("balanced", "target_price_balanced", "Balanced"),
                ("rsi", "target_price_rsi", "RSI"),
            ]:
                tp = meta.get(tkey)
                if tp is not None and not meta["alerts"].get(mkey, False) and latest >= tp:
                    send_telegram_message(
                        f"🚨 {sym} {label} hedefe ulaştı!\n"
                        f"Şu an: {latest:.4f} ₺ (baseline: {baseline:.4f})\n"
                        f"Hedef: {tp:.4f} ₺"
                    )
                    meta["alerts"][mkey] = True

            # 📌 Yükseliş kademeli bildirimi
            if pct >= MOVEMENT_NOTIFY_UP:
                steps_up = int(pct // MOVEMENT_NOTIFY_UP)
                last_up = meta.get("last_threshold_up", 0)
                if steps_up > last_up:
                    send_telegram_message(
                        f"📈 {sym} yükseliş: +{pct:.2f}% "
                        f"(baseline {baseline:.4f} → {latest:.4f})"
                    )
                    meta["last_threshold_up"] = steps_up

            # 📌 Düşüş kademeli bildirimi
            elif pct <= -MOVEMENT_NOTIFY_DWN:
                steps_down = int(abs(pct) // MOVEMENT_NOTIFY_DWN)
                last_down = meta.get("last_threshold_down", 0)
                if steps_down > last_down:
                    send_telegram_message(
                        f"📉 {sym} düşüş: {pct:.2f}% "
                        f"(baseline {baseline:.4f} → {latest:.4f})"
                    )
                    meta["last_threshold_down"] = steps_down

            print(
                f"  {sym}: latest={latest:.4f}, pct={pct:.2f}%, "
                f"alerts={meta['alerts']}, "
                f"up_steps={meta.get('last_threshold_up',0)}, "
                f"down_steps={meta.get('last_threshold_down',0)}"
            )

        print("✅ Kontrol tamamlandı.")
    return check_prices


# -----------------------------
# MAIN
# -----------------------------
def main():
    tz = pytz.timezone(MARKET_TZ)

    # --- RUNNING flag başlangıçta kontrol ---
    RUNNING = os.getenv("RUNNING", "true").lower() in ("1", "true", "yes")

    if RUNNING:
        # ANALIZ ve INITIAL CHECKER
        combined_df, monitored, top_dict = analyze_once()
        monitored_valid = {s: m for s, m in monitored.items() if m.get("baseline_price") is not None}

        if monitored_valid:
            checker = create_price_checker(monitored_valid)
            checker()  # ilk kontrol ve mesaj gönderimi

            # scheduler'a ekle
            schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(checker)
            print(f"✅ RUNNING=True, izleme başladı. Her {CHECK_INTERVAL_MINUTES} dakikada bir kontrol edilecek.")
        else:
            print("Geçerli baseline fiyatı olan izlenecek sembol yok. Program beklemeye alındı.")
            checker = None
    else:
        print("⏸ RUNNING=False, analiz ve mesaj gönderimi atlandı.")
        checker = None
        monitored_valid = {}

    try:
        while True:
            # --- MARKET HOURS kontrolü ---
            now = datetime.now(tz)
            if USE_MARKET_HOURS and not (MARKET_OPEN <= now.time() <= MARKET_CLOSE):
                print(f"⏸ Market saatleri dışında ({now.strftime('%H:%M:%S')})")
                time.sleep(60)
                continue

            # --- RUNNING=false ise sadece bekle ---
            RUNNING = os.getenv("RUNNING", "true").lower() in ("1", "true", "yes")
            if not RUNNING or checker is None:
                time.sleep(60)
                continue

            # periyodik kontrol
            schedule.run_pending()
            time.sleep(1)

    except KeyboardInterrupt:
        print("Program manuel olarak durduruldu.")
    except Exception as e:
        print("Ana döngüde hata:", e)

if __name__ == "__main__":
    main()

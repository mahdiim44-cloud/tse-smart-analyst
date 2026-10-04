from __future__ import annotations
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# TSE SMART ANALYST V2 - MOBILE / TWO FILE EDITION
# فقط app.py + requirements.txt
# ============================================================

BASE = "https://cdn.tsetmc.com/api"

class TSETMCError(RuntimeError):
    pass

class TSETMCClient:
    def __init__(self, timeout=20):
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"})

    def _get(self, path):
        try:
            r = self.s.get(f"{BASE}/{path.lstrip('/')}", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            raise TSETMCError(f"دریافت داده TSETMC ناموفق بود: {e}")

    @staticmethod
    def _unwrap(obj):
        if isinstance(obj, dict) and len(obj) == 1:
            return next(iter(obj.values()))
        return obj

    def search_symbol(self, query):
        data = self._unwrap(self._get(f"Instrument/GetInstrumentSearch/{query}"))
        rows = data if isinstance(data, list) else []
        if not rows:
            raise TSETMCError(f"نماد «{query}» پیدا نشد.")
        row = rows[0]
        ins = str(row.get("insCode") or row.get("inscode") or row.get("instrumentId") or "")
        if not ins:
            for v in row.values():
                if str(v).isdigit() and len(str(v)) >= 8:
                    ins = str(v)
                    break
        if not ins:
            raise TSETMCError("کد معاملاتی نماد پیدا نشد.")
        return {
            "ins_code": ins,
            "symbol": row.get("lVal18AFC") or row.get("symbol") or query,
            "name": row.get("lVal30") or row.get("name") or "",
        }

    def quote(self, ins):
        return self._unwrap(self._get(f"ClosingPrice/GetClosingPriceInfo/{ins}")) or {}

    def history(self, ins, days=180):
        data = self._unwrap(self._get(f"ClosingPrice/GetClosingPriceDailyList/{ins}/{days}"))
        if isinstance(data, dict):
            for k in ("closingPriceDaily", "closingPriceDailyList", "closingPriceHistory"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        rows = []
        for x in data if isinstance(data, list) else []:
            def g(*keys, default=0):
                for k in keys:
                    if k in x and x[k] is not None:
                        return x[k]
                return default
            rows.append({
                "date_raw": g("dEven", "date"),
                "open": float(g("priceFirst", "pFirst", "pf")),
                "high": float(g("priceMax", "pMax", "pmax")),
                "low": float(g("priceMin", "pMin", "pmin")),
                "close": float(g("pClosing", "close", "pc")),
                "last": float(g("pDrCotVal", "last", "pl")),
                "volume": float(g("qTotTran5J", "volume", "tvol")),
                "value": float(g("qTotCap", "value", "tval")),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise TSETMCError("تاریخچه قیمت خالی است.")
        df["date"] = pd.to_datetime(df["date_raw"].astype(str), format="%Y%m%d", errors="coerce")
        return df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)

    def client_type_history(self, ins, days=30):
        data = self._unwrap(self._get(f"ClientType/GetClientTypeHistory/{ins}"))
        if isinstance(data, dict):
            for k in ("clientType", "clientTypeHistory", "clientTypeAllDto"):
                if isinstance(data.get(k), list):
                    data = data[k]
                    break
        rows = data if isinstance(data, list) else []
        out = []
        for x in rows[-days:]:
            def g(*keys):
                for k in keys:
                    if k in x:
                        return float(x[k] or 0)
                return 0.0
            out.append({
                "date": x.get("dEven"),
                "real_buy_volume": g("buy_I_Volume"),
                "real_sell_volume": g("sell_I_Volume"),
                "legal_buy_volume": g("buy_N_Volume"),
                "legal_sell_volume": g("sell_N_Volume"),
                "real_buy_count": g("buy_I_Count"),
                "real_sell_count": g("sell_I_Count"),
            })
        return pd.DataFrame(out)


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def add_indicators(df):
    x = df.copy()
    d = x.close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = (100 - 100 / (1 + rs)).fillna(50)
    x["macd"] = ema(x.close, 12) - ema(x.close, 26)
    x["macd_signal"] = ema(x.macd, 9)
    x["vol_ma20"] = x.volume.rolling(20).mean()
    x["volume_ratio"] = (x.volume / x.vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(1)
    x["ma50"] = x.close.rolling(50).mean()
    return x


def levels(df):
    r = df.loc[df.high.eq(df.high.rolling(20, min_periods=20).max()), "high"].dropna()
    s = df.loc[df.low.eq(df.low.rolling(20, min_periods=20).min()), "low"].dropna()
    cur = float(df.close.iloc[-1])
    supports = sorted(set(float(v) for v in s if v < cur), reverse=True)[:5]
    resistances = sorted(set(float(v) for v in r if v > cur))[:5]
    return supports, resistances


def flow(flow_df):
    if flow_df is None or flow_df.empty:
        return dict(real_buy_volume=0, real_sell_volume=0, legal_buy_volume=0, legal_sell_volume=0, buyer_power=1, real_money_net=0)
    f = flow_df.iloc[-1]
    rb, rs = f.real_buy_volume, f.real_sell_volume
    rbc, rsc = f.real_buy_count, f.real_sell_count
    p = (rb / rbc) / (rs / rsc) if rbc > 0 and rsc > 0 and rs > 0 else 1
    return dict(real_buy_volume=rb, real_sell_volume=rs, legal_buy_volume=f.legal_buy_volume,
                legal_sell_volume=f.legal_sell_volume, buyer_power=float(p), real_money_net=float(rb - rs))


def analyze_symbol(client, symbol, days=180):
    info = client.search_symbol(symbol)
    df = add_indicators(client.history(info["ins_code"], days))
    fl = flow(client.client_type_history(info["ins_code"], 30))
    q = client.quote(info["ins_code"])
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    change = ((last.close / prev.close) - 1) * 100 if prev.close else 0
    total = 0
    components = []

    def add(name, points, reason):
        nonlocal total
        total += points
        components.append({"فاکتور": name, "امتیاز": points, "توضیح": reason})

    if fl["buyer_power"] >= 1.5:
        add("قدرت خریدار", 22, "سرانه خرید حقیقی قوی است")
    elif fl["buyer_power"] >= 1.15:
        add("قدرت خریدار", 10, "قدرت خریدار مثبت است")
    elif fl["buyer_power"] < 0.85:
        add("قدرت خریدار", -18, "قدرت فروشنده غالب است")

    if fl["real_money_net"] > 0:
        add("پول حقیقی", 18, "خالص جریان حقیقی مثبت است")
    elif fl["real_money_net"] < 0:
        add("پول حقیقی", -18, "خالص جریان حقیقی منفی است")

    if last.rsi < 35:
        add("RSI", 14, "RSI در اشباع فروش")
    elif last.rsi > 70:
        add("RSI", -14, "RSI در اشباع خرید")
    elif last.rsi >= 50:
        add("RSI", 7, "RSI بالای 50")

    if last.macd > last.macd_signal:
        add("MACD", 12, "MACD بالاتر از سیگنال")
    else:
        add("MACD", -10, "MACD پایین‌تر از سیگنال")

    if last.volume_ratio >= 1.5 and change > 0:
        add("حجم", 12, "حجم قوی همراه رشد")
    elif last.volume_ratio >= 1.5 and change < 0:
        add("حجم", -10, "حجم قوی همراه افت")

    trend = bool(last.close > last.ma50) if pd.notna(last.ma50) else False
    add("روند", 10 if trend else -8, "قیمت بالای MA50" if trend else "قیمت زیر MA50")
    total = max(-100, min(100, int(total)))

    if total >= 60:
        label, summary = "🟢 خرید قوی", "هم‌جهتی چند عامل مهم به نفع خریدار است."
    elif total >= 25:
        label, summary = "🟢 خرید", "برآیند فعلی متمایل به صعود است."
    elif total <= -60:
        label, summary = "🔴 فروش قوی", "فشار فروش در چند عامل هم‌زمان غالب است."
    elif total <= -25:
        label, summary = "🔴 فروش", "برآیند فعلی متمایل به نزول است."
    else:
        label, summary = "🟡 خنثی", "مزیت آماری کافی برای ورود جهت‌دار دیده نمی‌شود."

    supports, resistances = levels(df)
    current = float(last.close)
    atr = float((df.high - df.low).rolling(14).mean().iloc[-1])
    if not np.isfinite(atr) or atr <= 0:
        atr = current * 0.03

    if total >= 25:
        entry_low, entry_high = current * 0.995, current * 1.005
        stop = max(current - 1.5 * atr, supports[0] if supports else current - 2 * atr)
        risk = max(current - stop, atr * 0.5)
        tps = [current + risk * 1.5, current + risk * 2.5, current + risk * 3.5]
    elif total <= -25:
        entry_low, entry_high = current * 0.995, current * 1.005
        stop = current + 1.5 * atr
        risk = max(stop - current, atr * 0.5)
        tps = [current - risk * 1.5, current - risk * 2.5, current - risk * 3.5]
    else:
        entry_low = entry_high = current
        stop = current
        tps = [resistances[0] if resistances else current, current, current]

    return {
        "symbol": info["symbol"], "name": info["name"], "history": df,
        "quote": {"last": float(q.get("pDrCotVal") or last.last or current), "change_pct": change},
        "metrics": {"buyer_power": fl["buyer_power"], "real_money_net": fl["real_money_net"],
                     "rsi": float(last.rsi), "macd": float(last.macd), "volume_ratio": float(last.volume_ratio),
                     "trend_up": trend},
        "score": {"total": total, "components": components},
        "signal": {"label": label, "summary": summary,
                    "confidence": round(min(95, 50 + abs(total) * 0.45)),
                    "reasons": [z["توضیح"] for z in components if z["امتیاز"]]},
        "supports": supports, "resistances": resistances,
        "trade_plan": {"entry": f"{entry_low:,.0f} – {entry_high:,.0f}", "stop": f"{stop:,.0f}",
                        "tp1": f"{tps[0]:,.0f}", "tp2": f"{tps[1]:,.0f}", "tp3": f"{tps[2]:,.0f}",
                        "rr": "1 : 1.5 / 2.5 / 3.5"}
    }


@st.cache_resource
def get_client():
    return TSETMCClient()


@st.cache_data(ttl=90, show_spinner=False)
def analyze_cached(symbol, days=180):
    return analyze_symbol(get_client(), symbol, days)


def scan_symbols(symbols, days=120):
    out = []
    for sym in symbols:
        try:
            a = analyze_cached(sym, days)
            out.append({
                "نماد": a["symbol"], "سیگنال": a["signal"]["label"], "امتیاز": a["score"]["total"],
                "تغییر٪": round(a["quote"]["change_pct"], 2), "قدرت خریدار": round(a["metrics"]["buyer_power"], 2),
                "RSI": round(a["metrics"]["rsi"], 1), "حجم/میانگین": round(a["metrics"]["volume_ratio"], 2),
                "خالص پول حقیقی": round(a["metrics"]["real_money_net"]),
            })
        except Exception:
            continue
    return out


def telegram_configured():
    try:
        return bool(st.secrets.get("TELEGRAM_BOT_TOKEN")) and bool(st.secrets.get("TELEGRAM_CHAT_ID"))
    except Exception:
        return False


def send_telegram(text):
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat_id, "text": text}, timeout=15)
        return (True, "پیام با موفقیت ارسال شد.") if r.ok else (False, f"Telegram error: {r.text[:300]}")
    except Exception as e:
        return False, str(e)


# ---------------- UI ----------------
st.set_page_config(page_title="TSE Smart Analyst V2", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1400px;padding:1rem .8rem 5rem}
button{min-height:44px!important}
@media(max-width:700px){.block-container{padding:.65rem .55rem 4rem}h1{font-size:1.5rem!important}h2{font-size:1.25rem!important}h3{font-size:1.05rem!important}}
</style>
""", unsafe_allow_html=True)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["فملی", "فولاد", "شستا"]
if "last_symbol" not in st.session_state:
    st.session_state.last_symbol = "فملی"

st.title("📈 TSE Smart Analyst")
st.caption("تحلیل هوشمند بازار سرمایه ایران — نسخه ۲ موبایل")

# Six simple tabs replace the old multi-page folder, so GitHub only needs 2 files.
tabs = st.tabs(["🏠 داشبورد", "📊 اسکنر", "⭐ واچ‌لیست", "🎯 تحلیل پیشرفته", "🔔 هشدار", "⚙️ تنظیمات"])

with tabs[0]:
    st.subheader("🔎 تحلیل سریع")
    symbol = st.text_input("نماد", value=st.session_state.last_symbol, placeholder="مثلاً فملی", key="dash_symbol")
    days = st.select_slider("تاریخچه", options=[60,90,120,180,250,365], value=180, key="dash_days")
    if st.button("🚀 تحلیل سهم", type="primary", use_container_width=True, key="dash_run"):
        try:
            with st.spinner("در حال دریافت و تحلیل داده‌ها..."):
                st.session_state.last_analysis = analyze_cached(symbol.strip(), days)
                st.session_state.last_symbol = symbol.strip()
        except Exception as e:
            st.error(str(e))
    if "last_analysis" in st.session_state:
        a = st.session_state.last_analysis
        q, m, s = a["quote"], a["metrics"], a["signal"]
        st.markdown(f"## {a['symbol']} — {a['name']}")
        c1,c2,c3 = st.columns(3)
        c1.metric("سیگنال", s["label"])
        c2.metric("امتیاز", f"{a['score']['total']:+d}/100")
        c3.metric("قیمت", f"{q['last']:,.0f}")
        c1,c2,c3 = st.columns(3)
        c1.metric("تغییر", f"{q['change_pct']:+.2f}%")
        c2.metric("قدرت خریدار", f"{m['buyer_power']:.2f}x")
        c3.metric("اعتماد نسبی", f"{s['confidence']}%")
        st.info(f"{s['summary']}")
        st.markdown("### 📌 عوامل اصلی")
        for reason in s["reasons"][:6]: st.write("•", reason)
        st.markdown("### 🎯 سناریوی معاملاتی")
        sc=a["trade_plan"]
        c1,c2,c3=st.columns(3)
        c1.metric("ورود",sc["entry"]); c2.metric("حد ضرر",sc["stop"]); c3.metric("R/R",sc["rr"])
        st.write(f"**هدف ۱:** {sc['tp1']}  |  **هدف ۲:** {sc['tp2']}  |  **هدف ۳:** {sc['tp3']}")
        st.caption("اهداف و حدضرر محاسباتی هستند و جایگزین مدیریت ریسک شخصی نیستند.")
        st.markdown("### 📊 وضعیت")
        st.write(f"RSI: **{m['rsi']:.1f}** · MACD: **{m['macd']:.2f}** · حجم: **{m['volume_ratio']:.2f}x**")
        st.write(f"خالص حجم حقیقی: **{m['real_money_net']:,.0f}** · روند MA50: **{'مثبت' if m['trend_up'] else 'منفی'}**")

with tabs[1]:
    st.subheader("📊 اسکنر بازار")
    default="فملی,فولاد,شستا,خودرو,خساپا,وبملت,وتجارت,شپنا,شبندر,شتران,ذوب,کگل,کچاد,فارس,نوری,پارس,تاپیکو"
    raw=st.text_area("نمادها",value=default,height=90)
    symbols=[x.strip() for x in raw.replace("،",",").split(",") if x.strip()]
    days_scan=st.select_slider("تاریخچه اسکن",options=[60,90,120,180],value=120,key="scan_days")
    if st.button("🔎 شروع اسکن",type="primary",use_container_width=True):
        with st.spinner(f"در حال اسکن {len(symbols)} نماد..."):
            rows=scan_symbols(symbols,days_scan)
        if rows:
            df=pd.DataFrame(rows).sort_values("امتیاز",ascending=False)
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.markdown("### ⭐ برترین‌های فعلی")
            for _,r in df.head(5).iterrows(): st.write(f"**{r['نماد']}** — {r['سیگنال']} — امتیاز {r['امتیاز']:+d} — قدرت خریدار {r['قدرت خریدار']:.2f}x")
        else: st.warning("داده‌ای دریافت نشد.")

with tabs[2]:
    st.subheader("⭐ واچ‌لیست من")
    c1,c2=st.columns([2,1])
    new= c1.text_input("افزودن نماد",placeholder="مثلاً فملی",label_visibility="collapsed",key="new_watch")
    if c2.button("➕ افزودن",use_container_width=True):
        if new.strip() and new.strip() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new.strip()); st.rerun()
    for sym in list(st.session_state.watchlist):
        try:
            a=analyze_cached(sym,120)
            c1,c2,c3=st.columns([1.2,1,0.8])
            c1.markdown(f"### {a['symbol']}"); c1.caption(a['name'])
            c2.metric("سیگنال",a['signal']['label'])
            c3.metric("امتیاز",f"{a['score']['total']:+d}")
            if c3.button("حذف",key=f"del_{sym}",use_container_width=True):
                st.session_state.watchlist.remove(sym); st.rerun()
            st.divider()
        except Exception as e: st.warning(f"{sym}: {e}")
    st.caption("واچ‌لیست V2 تا وقتی نشست مرورگر فعال است نگهداری می‌شود.")

with tabs[3]:
    st.subheader("🎯 تحلیل پیشرفته")
    sym=st.text_input("نماد",value=st.session_state.last_symbol,key="adv_symbol")
    adv_days=st.select_slider("تاریخچه",options=[120,180,250,365],value=365,key="adv_days")
    if st.button("📈 اجرای تحلیل",type="primary",use_container_width=True):
        try: st.session_state.advanced=analyze_cached(sym.strip(),adv_days); st.session_state.last_symbol=sym.strip()
        except Exception as e: st.error(str(e))
    if "advanced" in st.session_state:
        a=st.session_state.advanced; df=a["history"]
        fig=go.Figure(go.Candlestick(x=df.date,open=df.open,high=df.high,low=df.low,close=df.close,name="Price"))
        for x in a["supports"][:3]: fig.add_hline(y=x,line_dash="dot",annotation_text=f"S {x:,.0f}")
        for x in a["resistances"][:3]: fig.add_hline(y=x,line_dash="dot",annotation_text=f"R {x:,.0f}")
        fig.update_layout(height=520,xaxis_rangeslider_visible=False,template="plotly_dark",margin=dict(l=5,r=5,t=10,b=5))
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("### 🧩 اجزای امتیاز")
        for row in a["score"]["components"]: st.write(f"**{row['فاکتور']}** → {row['امتیاز']:+d} — {row['توضیح']}")
        st.markdown("### 🛡️ مدیریت ریسک")
        st.json(a["trade_plan"])

with tabs[4]:
    st.subheader("🔔 هشدارها")
    st.caption("ارسال سیگنال به تلگرام از طریق Bot API؛ توکن را فقط در Secrets نگهداری کن.")
    if telegram_configured():
        st.success("اتصال تلگرام پیکربندی شده است.")
        msg=st.text_area("متن آزمایشی",value="TSE Smart Analyst: test alert")
        if st.button("📨 ارسال تست",type="primary",use_container_width=True):
            ok,detail=send_telegram(msg)
            st.success(detail) if ok else st.error(detail)
    else:
        st.warning("تلگرام هنوز تنظیم نشده است.")
        st.code('TELEGRAM_BOT_TOKEN = "..."\nTELEGRAM_CHAT_ID = "..."',language="toml")
    st.markdown("### منطق هشدار پیشنهادی V3")
    st.write("• امتیاز ≥ +60 → سیگنال صعودی قوی\n• امتیاز ≤ -60 → سیگنال نزولی قوی\n• تغییر ناگهانی قدرت خریدار\n• ورود پول حقیقی + شکست مقاومت")

with tabs[5]:
    st.subheader("⚙️ تنظیمات")
    st.info("V2 برای اجرای ساده روی موبایل در دو فایل فشرده شده است. برای دیتابیس، حساب کاربری و تاریخچه دائمی سیگنال‌ها، V3 به پایگاه‌داده نیاز دارد.")
    st.markdown("### معماری")
    st.write("Data: TSETMC · Cache: Streamlit · State: Session State · Analysis: Technical + Money Flow + Buyer Power + Volume + Trend · UI: Mobile-first · Alerts: Telegram-ready")
    st.markdown("### نکته امنیتی")
    st.write("توکن Telegram را داخل کد GitHub قرار نده. در Streamlit Cloud از بخش Secrets وارد کن.")

st.divider()
st.caption("TSE Smart Analyst V2 — ابزار تحقیق و تحلیل؛ نه توصیه قطعی سرمایه‌گذاری.")

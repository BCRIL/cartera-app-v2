import streamlit as st
from supabase import create_client, Client
import pandas as pd
import yfinance as yf
from yahooquery import search
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from datetime import datetime, timedelta
import re
from duckduckgo_search import DDGS
import openai
import io

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Carterapro Ultra", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# ==============================================================================
# DISEÑO DARK MODE PROFESIONAL
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
    h1, h2, h3, p, div, span, label { color: #E6E6E6 !important; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }

    /* KPIs */
    div[data-testid="stMetric"] {
        background-color: #21262D; border: 1px solid #30363D; border-radius: 12px;
        padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] { font-size: 1.5rem !important; color: white !important; font-weight: 700; }
    div[data-testid="stMetricDelta"] svg { display: none; }

    /* Botones */
    .stButton>button { border-radius: 8px; font-weight: 600; border: 1px solid #30363D; background-color: #21262D; color: white; transition: 0.3s; }
    .stButton>button:hover { border-color: #00CC96; color: #00CC96; }
    .stButton>button[kind="primary"] { background: linear-gradient(135deg, #00CC96 0%, #007d5c 100%); border: none; color: black !important; }

    /* Noticias */
    .news-scroll-area {
        height: 68vh; overflow-y: auto; padding: 10px; background-color: #161B22;
        border: 1px solid #30363D; border-radius: 12px; margin-top: 15px;
    }
    .news-scroll-area::-webkit-scrollbar { width: 6px; }
    .news-scroll-area::-webkit-scrollbar-thumb { background: #30363D; border-radius: 4px; }
    .news-card {
        background-color: #21262D; border-radius: 8px; padding: 12px; margin-bottom: 12px;
        border: 1px solid #30363D; display: flex; flex-direction: column; transition: transform 0.2s;
    }
    .news-card:hover { transform: translateY(-2px); border-color: #00CC96; }
    .news-img { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; margin-bottom: 8px; opacity: 0.8; }
    .news-title a { color: #FFFFFF !important; text-decoration: none; font-weight: 600; font-size: 0.95rem; display: block; line-height: 1.3; }
    .news-title a:hover { color: #00CC96 !important; }
    .news-source { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; font-weight: bold; }

    /* Tablas */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Gráficos Limpios */
    .js-plotly-plot .plotly .main-svg { background-color: rgba(0,0,0,0) !important; }

    /* Stat Card */
    .stat-card {
        background: linear-gradient(135deg, #21262D 0%, #161B22 100%);
        border: 1px solid #30363D; border-radius: 12px; padding: 20px;
        text-align: center; transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-3px); border-color: #00CC96; }
    .stat-card h2 { margin: 0; font-size: 2rem; }
    .stat-card p { margin: 5px 0 0 0; color: #8b949e !important; font-size: 0.85rem; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #21262D; border-radius: 8px; border: 1px solid #30363D; }
    .stTabs [aria-selected="true"] { border-color: #00CC96 !important; }
</style>
""", unsafe_allow_html=True)

# --- CONEXIONES ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    HAS_GROQ = "GROQ_API_KEY" in st.secrets
except Exception:
    st.error("Error Crítico: Faltan secretos en la configuración."); st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- SESIÓN ---
defaults = {
    'user': None, 'show_news': True, 'messages': [],
    'watchlist': [], 'tx_log': []
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# LOGIN
# ==============================================================================
query_params = st.query_params
if "code" in query_params and not st.session_state['user']:
    try:
        session = supabase.auth.exchange_code_for_session({"auth_code": query_params["code"]})
        st.session_state['user'] = session.user
        st.query_params.clear()
        st.rerun()
    except Exception:
        pass

if not st.session_state['user']:
    _c1, _c2, _c3 = st.columns([1, 2, 1])
    with _c2:
        st.markdown("<br><h1 style='text-align: center; color: #00CC96;'>🏦 Carterapro Ultra</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#8b949e !important;'>Gestión patrimonial inteligente</p>", unsafe_allow_html=True)
        tab_login, tab_signup = st.tabs(["🔐 Entrar", "📝 Registro"])
        with tab_login:
            if st.button("🇬 Entrar con Google", type="primary", use_container_width=True):
                try:
                    data = supabase.auth.sign_in_with_oauth({"provider": "google", "options": {"redirect_to": "https://carterapro.streamlit.app"}})
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error al conectar con Google: {e}")
            st.divider()
            with st.form("login_form"):
                em = st.text_input("Email")
                pa = st.text_input("Contraseña", type="password")
                if st.form_submit_button("Entrar", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": em, "password": pa})
                        st.session_state['user'] = res.user
                        st.rerun()
                    except Exception:
                        st.error("Credenciales incorrectas. Inténtalo de nuevo.")
        with tab_signup:
            with st.form("signup_form"):
                em2 = st.text_input("Email")
                pa2 = st.text_input("Contraseña", type="password")
                pa3 = st.text_input("Confirmar Contraseña", type="password")
                if st.form_submit_button("Crear Cuenta", use_container_width=True):
                    if pa2 != pa3:
                        st.error("Las contraseñas no coinciden.")
                    elif len(pa2) < 6:
                        st.error("La contraseña debe tener al menos 6 caracteres.")
                    else:
                        try:
                            supabase.auth.sign_up({"email": em2, "password": pa2})
                            st.success("Cuenta creada. Revisa tu email para confirmar.")
                        except Exception as e:
                            st.error(f"Error al crear cuenta: {e}")
    st.stop()

user = st.session_state['user']

# ==============================================================================
# MOTOR DE DATOS
# ==============================================================================

def sanitize_input(text):
    return re.sub(r'[^\w\s\-\.]', '', str(text)).strip().upper()

def safe_metric_calc(price_series):
    """Calcula métricas de riesgo/retorno a partir de una serie de precios."""
    clean = price_series.dropna()
    if len(clean) < 5:
        return 0, 0, 0, 0
    returns = clean.pct_change().dropna()
    if returns.empty or len(returns) < 2:
        return 0, 0, 0, 0
    try:
        total_ret = (clean.iloc[-1] / clean.iloc[0]) - 1
    except (ZeroDivisionError, IndexError):
        total_ret = 0
    vol = returns.std() * np.sqrt(252)
    mean_ret = returns.mean() * 252
    sharpe = (mean_ret - 0.03) / vol if vol > 0.001 else 0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min() if not dd.empty else 0
    return total_ret, vol, max_dd, sharpe

def calc_sortino(returns, rf=0.03):
    """Ratio de Sortino: penaliza solo volatilidad bajista."""
    excess = returns - rf / 252
    downside = returns[returns < 0]
    down_std = downside.std() * np.sqrt(252)
    if down_std < 0.001:
        return 0
    return (returns.mean() * 252 - rf) / down_std

def calc_calmar(returns, max_dd):
    """Ratio de Calmar: retorno anualizado / max drawdown."""
    ann_ret = returns.mean() * 252
    if abs(max_dd) < 0.001:
        return 0
    return ann_ret / abs(max_dd)

def calc_var(returns, confidence=0.05):
    """Value at Risk paramétrico."""
    if len(returns) < 10:
        return 0
    return np.percentile(returns, confidence * 100)

def diversification_score(weights):
    """Índice Herfindahl-Hirschman invertido (0-100, mayor = más diversificado)."""
    if len(weights) == 0:
        return 0
    hhi = sum(w ** 2 for w in weights)
    max_hhi = 1.0
    min_hhi = 1.0 / len(weights) if len(weights) > 0 else 1.0
    if max_hhi == min_hhi:
        return 100
    return max(0, min(100, (1 - (hhi - min_hhi) / (max_hhi - min_hhi)) * 100))

@st.cache_data(ttl=60)
def get_user_data(uid):
    try:
        assets = pd.DataFrame(supabase.table('assets').select("*").eq('user_id', uid).execute().data)
    except Exception:
        assets = pd.DataFrame()
    try:
        liq_res = supabase.table('liquidity').select("*").eq('user_id', uid).execute().data
    except Exception:
        liq_res = []
    if not liq_res:
        try:
            supabase.table('liquidity').insert({"user_id": uid, "name": "Principal", "amount": 0.0}).execute()
        except Exception:
            pass
        liquidity = 0.0
        liq_id = 0
    else:
        liquidity = liq_res[0]['amount']
        liq_id = liq_res[0]['id']
    return assets, liquidity, liq_id

def get_real_time_prices(tickers):
    """Obtiene precios actuales de mercado."""
    if not tickers:
        return {}
    prices = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            if 'last_price' in info and info['last_price'] is not None:
                prices[t] = info['last_price']
            else:
                hist = yf.Ticker(t).history(period='2d')
                prices[t] = hist['Close'].iloc[-1] if not hist.empty else 0.0
        except Exception:
            prices[t] = 0.0
    return prices

def get_ticker_info(ticker):
    """Obtiene información detallada de un ticker."""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        return {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'currency': info.get('currency', 'N/A'),
            'country': info.get('country', 'N/A'),
            'dividend_yield': info.get('dividendYield', 0) or 0,
            'market_cap': info.get('marketCap', 0) or 0,
            'pe_ratio': info.get('trailingPE', 0) or 0,
            'beta': info.get('beta', 1) or 1,
            'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0) or 0,
            'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0) or 0,
            'name': info.get('longName', info.get('shortName', ticker)),
        }
    except Exception:
        return {}

@st.cache_data(ttl=300)
def get_historical_data_robust(tickers):
    """Descarga histórico y normaliza timezones."""
    if not tickers:
        return pd.DataFrame()

    unique_tickers = list(set([str(t).strip().upper() for t in tickers] + ['SPY']))
    try:
        data = yf.download(unique_tickers, period="2y", interval="1d", progress=False, auto_adjust=True)['Close']

        if isinstance(data, pd.Series):
            data = data.to_frame()
            if len(unique_tickers) == 1 or (data.columns[0] == 0 or 'Close' in str(data.columns[0])):
                data.columns = unique_tickers[:len(data.columns)]

        if not data.empty:
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            data = data.ffill().bfill()

        return data
    except Exception as e:
        st.warning(f"Error al descargar datos históricos: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_global_news(tickers, time_filter='d'):
    """Obtiene noticias financieras relevantes."""
    results = []
    queries = []
    if tickers:
        queries.append(f"{tickers[0]} bolsa inversión")
        if len(tickers) > 1:
            queries.append(f"{tickers[1]} mercado")
    queries.append("Mercado financiero bolsa")

    for query in queries[:2]:
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.news(query, region="es-es", safesearch="off", timelimit=time_filter, max_results=6))
                for n in raw:
                    if n.get('title') and n.get('url'):
                        results.append({
                            'title': n.get('title'), 'source': n.get('source', 'Web'),
                            'date': n.get('date', ''), 'url': n.get('url'),
                            'image': n.get('image', None)
                        })
        except Exception:
            pass

    # Deduplicar por título
    seen = set()
    unique = []
    for r in results:
        if r['title'] not in seen:
            seen.add(r['title'])
            unique.append(r)
    return unique[:10]

def clear_cache():
    st.cache_data.clear()

# --- DB FUNCTIONS ---
def add_asset_db(t, n, s, p, pl):
    try:
        supabase.table('assets').insert({
            "user_id": user.id, "ticker": t, "nombre": n,
            "shares": s, "avg_price": p, "platform": pl
        }).execute()
        log_transaction("COMPRA", t, n, s, p, pl)
        clear_cache()
    except Exception as e:
        st.error(f"Error al guardar activo: {e}")

def update_asset_db(asset_id, shares, avg_price):
    try:
        supabase.table('assets').update({"shares": shares, "avg_price": avg_price}).eq('id', asset_id).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error al actualizar activo: {e}")

def delete_asset_db(id_del):
    try:
        supabase.table('assets').delete().eq('id', id_del).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error al eliminar activo: {e}")

def update_liquidity_balance(liq_id, new_amount):
    try:
        supabase.table('liquidity').update({"amount": new_amount}).eq('id', liq_id).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error al actualizar liquidez: {e}")

def log_transaction(tipo, ticker, nombre, shares, price, platform=""):
    """Registra transacción en el log local."""
    st.session_state['tx_log'].append({
        'fecha': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'tipo': tipo, 'ticker': ticker, 'nombre': nombre,
        'shares': round(shares, 6), 'precio': round(price, 4),
        'importe': round(shares * price, 2), 'platform': platform
    })

# --- PROCESO DE CARGA ---
df_assets, total_liquidez, cash_id = get_user_data(user.id)
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series(dtype=float)
my_tickers = []

if not df_assets.empty:
    my_tickers = df_assets['ticker'].unique().tolist()
    current_prices = get_real_time_prices(my_tickers)
    history_raw = get_historical_data_robust(my_tickers)

    df_assets['Precio Actual'] = df_assets['ticker'].map(current_prices).fillna(0.0)
    df_assets['Valor Acciones'] = df_assets['shares'] * df_assets['Precio Actual']
    df_assets['Dinero Invertido'] = df_assets['shares'] * df_assets['avg_price']
    df_assets['Ganancia'] = df_assets['Valor Acciones'] - df_assets['Dinero Invertido']
    df_assets['Rentabilidad %'] = df_assets.apply(
        lambda r: (r['Ganancia'] / r['Dinero Invertido'] * 100) if r['Dinero Invertido'] > 0 else 0, axis=1
    )
    total_inv_val = df_assets['Valor Acciones'].sum()
    df_assets['Peso %'] = df_assets.apply(
        lambda r: (r['Valor Acciones'] / total_inv_val * 100) if total_inv_val > 0 else 0, axis=1
    )
    df_final = df_assets.rename(columns={'nombre': 'Nombre'})

    if not history_raw.empty:
        if 'SPY' in history_raw.columns:
            benchmark_data = history_raw['SPY']
            history_data = history_raw.drop(columns=['SPY'], errors='ignore')
        else:
            history_data = history_raw

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0.0
patrimonio_total = total_inversiones + total_liquidez

# --- SIDEBAR ---
with st.sidebar:
    avatar = user.user_metadata.get('avatar_url', '') if user.user_metadata else ''
    nombre = user.user_metadata.get('full_name', 'Inversor') if user.user_metadata else 'Inversor'
    if avatar:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; padding:10px; background:#21262D; border-radius:8px; border:1px solid #30363D;'>
            <img src='{avatar}' style='width:35px; border-radius:50%; border:2px solid #00CC96;'>
            <div style='line-height:1.2'><b style='color:white'>{nombre}</b><br><span style='font-size:0.7em; color:#00CC96'>● Online</span></div>
        </div><br>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding:10px; background:#21262D; border-radius:8px; border:1px solid #30363D;'>
            <b style='color:white'>👤 {nombre}</b><br><span style='font-size:0.7em; color:#00CC96'>● Online</span>
        </div><br>""", unsafe_allow_html=True)

    if st.button("🔄 Actualizar Datos", use_container_width=True, type="primary"):
        clear_cache()
        st.rerun()

    st.divider()
    sc1, sc2 = st.columns([1, 4])
    with sc1:
        st.write("📰")
    with sc2:
        st.session_state['show_news'] = st.toggle("Noticias", value=st.session_state['show_news'])

    pagina = st.radio("MENÚ", [
        "📊 Dashboard",
        "💰 Liquidez",
        "➕ Inversiones",
        "📋 Historial",
        "🔍 Watchlist",
        "🔬 Radiografía",
        "💬 Asesor AI",
        "🔮 Monte Carlo",
        "⚖️ Rebalanceo"
    ])

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Patrimonio: **{patrimonio_total:,.2f} €**")
    if st.button("Cerrar Sesión", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.rerun()

# --- MAIN ---
if st.session_state['show_news']:
    col_main, col_news = st.columns([3.5, 1.3])
else:
    col_main = st.container()
    col_news = st.container()

with col_main:

    # ==================================================================
    # DASHBOARD
    # ==================================================================
    if pagina == "📊 Dashboard":
        st.title("📊 Control de Mando Integral")

        col_kpi, col_date = st.columns([3, 1])
        with col_date:
            periodo_rapido = st.selectbox("Periodo:", ["1M", "3M", "6M", "1A", "2A", "Personalizado"], index=3)
            if periodo_rapido == "Personalizado":
                start_date = st.date_input("Desde:", value=datetime.now() - timedelta(days=365))
            else:
                dias = {"1M": 30, "3M": 90, "6M": 180, "1A": 365, "2A": 730}
                start_date = (datetime.now() - timedelta(days=dias[periodo_rapido])).date()

        # Calcular métricas avanzadas
        vol_anual = 0; sharpe_ratio = 0; max_drawdown = 0; beta_portfolio = 1.0
        sortino = 0; calmar = 0; var_95 = 0; alpha_jensen = 0
        daily_returns = pd.Series(dtype=float)

        if not history_data.empty:
            dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
            hist_filt = history_data[history_data.index >= dt_start].copy()

            if not hist_filt.empty and len(hist_filt) > 5:
                daily_returns = hist_filt.pct_change().mean(axis=1).dropna()
                if not daily_returns.empty and len(daily_returns) > 5:
                    port_prices = (1 + daily_returns).cumprod()
                    total_ret_period, vol_dec, max_dd_dec, sharpe_ratio = safe_metric_calc(port_prices)
                    vol_anual = vol_dec * 100
                    max_drawdown = max_dd_dec * 100
                    sortino = calc_sortino(daily_returns)
                    calmar = calc_calmar(daily_returns, max_dd_dec)
                    var_95 = calc_var(daily_returns, 0.05) * 100

                    if not benchmark_data.empty:
                        bench_filt = benchmark_data[benchmark_data.index >= dt_start]
                        bench_ret = bench_filt.pct_change().dropna()
                        common = daily_returns.index.intersection(bench_ret.index)
                        if len(common) > 10:
                            cov_val = daily_returns.loc[common].cov(bench_ret.loc[common])
                            var_val = bench_ret.loc[common].var()
                            beta_portfolio = cov_val / var_val if var_val != 0 else 1.0
                            # Alpha de Jensen
                            rf_daily = 0.03 / 252
                            alpha_jensen = (daily_returns.loc[common].mean() - rf_daily) - beta_portfolio * (bench_ret.loc[common].mean() - rf_daily)
                            alpha_jensen *= 252  # Anualizar

        with col_kpi:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("💰 Patrimonio Neto", f"{patrimonio_total:,.2f} €")

            if total_inversiones > 0:
                pnl_total = df_final['Ganancia'].sum()
                dinero_inv_total = df_final['Dinero Invertido'].sum()
                rent_total_pct = (pnl_total / dinero_inv_total) * 100 if dinero_inv_total > 0 else 0
                delta_color = "normal" if pnl_total >= 0 else "inverse"
                k2.metric("📈 P&L Total", f"{pnl_total:+,.2f} €", f"{rent_total_pct:+.2f}%", delta_color=delta_color)
            else:
                k2.metric("📈 P&L Total", "0.00 €")

            k3.metric("💧 Liquidez", f"{total_liquidez:,.2f} €",
                       f"{(total_liquidez / patrimonio_total * 100 if patrimonio_total > 0 else 0):.1f}% del total")
            k4.metric("⚖️ Sharpe", f"{sharpe_ratio:.2f}")

        st.divider()

        # Fila 2: Métricas avanzadas
        r1, r2, r3, r4, r5, r6 = st.columns(6)
        r1.metric("⚡ Volatilidad", f"{vol_anual:.2f}%")
        r2.metric("📉 Max Drawdown", f"{max_drawdown:.2f}%")
        r3.metric("🌊 Beta", f"{beta_portfolio:.2f}")
        r4.metric("🎯 Sortino", f"{sortino:.2f}")
        r5.metric("🔻 VaR 95%", f"{var_95:.2f}%")
        r6.metric("⭐ Alpha", f"{alpha_jensen * 100:.2f}%")

        st.divider()

        # GRÁFICOS PRINCIPALES
        c_chart, c_donut = st.columns([2, 1.2])
        with c_chart:
            st.subheader("🏁 Rendimiento (Base 100)")
            if not history_data.empty:
                dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
                hist_filt = history_data[history_data.index >= dt_start].copy()

                if not hist_filt.empty and len(hist_filt) > 2:
                    port_ret = hist_filt.pct_change().mean(axis=1).fillna(0)
                    port_cum = (1 + port_ret).cumprod() * 100

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Tu Cartera",
                                             line=dict(color='#00CC96', width=2.5),
                                             fill='tozeroy', fillcolor='rgba(0,204,150,0.05)'))

                    if not benchmark_data.empty:
                        bench_filt = benchmark_data[benchmark_data.index >= dt_start].copy()
                        if not bench_filt.empty:
                            bench_ret = bench_filt.pct_change().fillna(0)
                            bench_cum = (1 + bench_ret).cumprod() * 100
                            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500",
                                                      line=dict(color='#636EFA', dash='dot', width=1.5)))

                    # Mostrar activos individuales si son pocos
                    valid_cols = [c for c in hist_filt.columns if c in my_tickers]
                    if len(valid_cols) <= 5:
                        colors = ['#EF553B', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
                        for idx, col in enumerate(valid_cols):
                            col_ret = hist_filt[col].pct_change().fillna(0)
                            col_cum = (1 + col_ret).cumprod() * 100
                            fig.add_trace(go.Scatter(x=col_cum.index, y=col_cum, name=col,
                                                      line=dict(color=colors[idx % len(colors)], width=1), opacity=0.6))

                    fig.add_hline(y=100, line_dash="dash", line_color="gray", line_width=0.5)
                    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=20, b=0),
                                      paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified", yaxis_title="Base 100",
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Datos insuficientes para el periodo seleccionado.")
            else:
                st.info("Añade activos para ver el histórico.")

        with c_donut:
            st.subheader("🍰 Distribución")
            if patrimonio_total > 0:
                labels_pie = (['Liquidez'] + df_final['Nombre'].tolist()) if not df_final.empty else ['Liquidez']
                values_pie = ([total_liquidez] + df_final['Valor Acciones'].tolist()) if not df_final.empty else [total_liquidez]
                fig_pie = px.pie(names=labels_pie, values=values_pie, hole=0.65,
                                  color_discrete_sequence=['#636EFA'] + px.colors.qualitative.Set3)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label',
                                       textfont_size=10, hovertemplate='%{label}: %{value:,.2f}€<br>%{percent}')
                fig_pie.update_layout(template="plotly_dark", height=350, showlegend=False,
                                       margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)',
                                       annotations=[dict(text=f"{patrimonio_total:,.0f}€", x=0.5, y=0.5,
                                                          font_size=16, font_color='white', showarrow=False)])
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Cartera vacía.")

        st.divider()

        # FILA 2: Treemap + Barras + Drawdown
        c_tree, c_bar = st.columns([1.5, 1.5])
        with c_tree:
            st.subheader("🗺️ Mapa de Calor")
            if not df_final.empty:
                fig_tree = px.treemap(df_final, path=['Nombre'], values='Valor Acciones', color='Rentabilidad %',
                                      color_continuous_scale=['#EF553B', '#1e1e1e', '#00CC96'],
                                      color_continuous_midpoint=0,
                                      hover_data={'Valor Acciones': ':,.2f', 'Rentabilidad %': ':.2f', 'Peso %': ':.1f'})
                fig_tree.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=0, t=0, b=0),
                                        paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("Sin inversiones.")

        with c_bar:
            st.subheader("🏆 P&L por Activo")
            if not df_final.empty:
                df_sorted = df_final.sort_values('Ganancia', ascending=True)
                colors_bar = ['#EF553B' if x < 0 else '#00CC96' for x in df_sorted['Ganancia']]
                fig_bar = go.Figure(go.Bar(
                    x=df_sorted['Ganancia'], y=df_sorted['Nombre'], orientation='h',
                    marker_color=colors_bar,
                    text=df_sorted['Ganancia'].apply(lambda x: f"{x:+,.2f}€"),
                    textposition='outside', textfont_size=11
                ))
                fig_bar.update_layout(template="plotly_dark", height=350, margin=dict(l=0, r=10, t=0, b=0),
                                       paper_bgcolor='rgba(0,0,0,0)', xaxis_title="Ganancia (€)")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Sin inversiones.")

        # DRAWDOWN CHART
        if not daily_returns.empty and len(daily_returns) > 5:
            st.divider()
            c_dd, c_roll = st.columns(2)
            with c_dd:
                st.subheader("📉 Drawdown")
                cum_ret = (1 + daily_returns).cumprod()
                running_max = cum_ret.cummax()
                drawdown_series = (cum_ret - running_max) / running_max * 100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series, fill='tozeroy',
                                             fillcolor='rgba(239,85,59,0.2)', line=dict(color='#EF553B', width=1),
                                             name='Drawdown'))
                fig_dd.update_layout(template="plotly_dark", height=250, margin=dict(l=0, r=0, t=10, b=0),
                                      paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Drawdown %", hovermode="x unified")
                st.plotly_chart(fig_dd, use_container_width=True)

            with c_roll:
                st.subheader("📊 Retornos Rodantes (30d)")
                if len(daily_returns) > 30:
                    rolling_ret = daily_returns.rolling(30).mean() * 252 * 100
                    fig_roll = go.Figure()
                    fig_roll.add_trace(go.Scatter(x=rolling_ret.index, y=rolling_ret,
                                                   line=dict(color='#AB63FA', width=1.5), name='Retorno 30d'))
                    fig_roll.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
                    fig_roll.update_layout(template="plotly_dark", height=250, margin=dict(l=0, r=0, t=10, b=0),
                                            paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Ret. Anualizado %",
                                            hovermode="x unified")
                    st.plotly_chart(fig_roll, use_container_width=True)
                else:
                    st.info("Se necesitan al menos 30 días de datos.")

        # TABLA DE POSICIONES
        if not df_final.empty:
            st.divider()
            st.subheader("📋 Posiciones Detalladas")
            cols_show = ['Nombre', 'ticker', 'platform', 'shares', 'avg_price', 'Precio Actual',
                          'Dinero Invertido', 'Valor Acciones', 'Ganancia', 'Rentabilidad %', 'Peso %']
            cols_available = [c for c in cols_show if c in df_final.columns]
            df_show = df_final[cols_available].copy()
            df_show.columns = ['Nombre', 'Ticker', 'Broker', 'Acciones', 'P. Medio', 'P. Actual',
                                'Invertido', 'Valor', 'P&L', 'Rent. %', 'Peso %'][:len(cols_available)]

            st.dataframe(
                df_show.style.format({
                    'Acciones': '{:.4f}', 'P. Medio': '{:.4f}', 'P. Actual': '{:.4f}',
                    'Invertido': '{:,.2f}€', 'Valor': '{:,.2f}€', 'P&L': '{:+,.2f}€',
                    'Rent. %': '{:+.2f}%', 'Peso %': '{:.1f}%'
                }).applymap(lambda x: 'color: #00CC96' if isinstance(x, (int, float)) and x > 0
                             else ('color: #EF553B' if isinstance(x, (int, float)) and x < 0 else ''),
                             subset=['P&L', 'Rent. %']),
                use_container_width=True, height=min(400, 50 + len(df_show) * 35)
            )

            # Export CSV
            csv_data = df_show.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exportar CSV", csv_data, "cartera.csv", "text/csv", use_container_width=False)

    # ==================================================================
    # LIQUIDEZ
    # ==================================================================
    elif pagina == "💰 Liquidez":
        st.title("💰 Gestión de Liquidez")

        # Indicador principal
        pct_liq = (total_liquidez / patrimonio_total * 100) if patrimonio_total > 0 else 100
        color_liq = "#00CC96" if pct_liq >= 10 else ("#FFA15A" if pct_liq >= 5 else "#EF553B")
        st.markdown(f"""
        <div style="text-align:center; padding: 40px; background-color: #21262D; border-radius: 15px;
                     margin-bottom: 30px; border: 1px solid #30363D;">
            <h1 style="font-size: 4.5rem; color:{color_liq} !important; margin: 0;">{total_liquidez:,.2f} €</h1>
            <p style="color:#8B949E !important;">SALDO DISPONIBLE — {pct_liq:.1f}% del patrimonio</p>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("### 📥 Ingresar")
                a = st.number_input("Importe a ingresar (€)", 0.0, step=50.0, key="in")
                nota_in = st.text_input("Concepto", key="nota_in", placeholder="Ej: Nómina, transferencia...")
                if st.button("Confirmar Ingreso", type="primary", use_container_width=True) and a > 0:
                    update_liquidity_balance(int(cash_id), total_liquidez + a)
                    log_transaction("INGRESO", "CASH", nota_in or "Ingreso", 1, a)
                    st.toast(f"Ingresados {a:,.2f}€")
                    time.sleep(0.5)
                    st.rerun()
        with c2:
            with st.container(border=True):
                st.markdown("### 📤 Retirar")
                b = st.number_input("Importe a retirar (€)", 0.0, step=50.0, key="out")
                nota_out = st.text_input("Concepto", key="nota_out", placeholder="Ej: Gastos, transferencia...")
                if st.button("Confirmar Retirada", use_container_width=True) and b > 0:
                    if b > total_liquidez:
                        st.error(f"Saldo insuficiente. Disponible: {total_liquidez:,.2f}€")
                    else:
                        update_liquidity_balance(int(cash_id), total_liquidez - b)
                        log_transaction("RETIRO", "CASH", nota_out or "Retiro", 1, b)
                        st.toast(f"Retirados {b:,.2f}€")
                        time.sleep(0.5)
                        st.rerun()

        # Info sobre fondo de emergencia
        if patrimonio_total > 0:
            st.divider()
            st.subheader("💡 Análisis de Liquidez")
            c3, c4, c5 = st.columns(3)
            with c3:
                colchon = total_liquidez / patrimonio_total * 100 if patrimonio_total > 0 else 0
                status = "Óptimo" if colchon >= 15 else ("Aceptable" if colchon >= 5 else "Bajo")
                c_status = "#00CC96" if colchon >= 15 else ("#FFA15A" if colchon >= 5 else "#EF553B")
                st.metric("Colchón de Seguridad", f"{colchon:.1f}%", status)
            with c4:
                ratio_inv = total_inversiones / patrimonio_total * 100 if patrimonio_total > 0 else 0
                st.metric("Ratio Inversión", f"{ratio_inv:.1f}%")
            with c5:
                st.metric("Patrimonio Total", f"{patrimonio_total:,.2f} €")

    # ==================================================================
    # INVERSIONES
    # ==================================================================
    elif pagina == "➕ Inversiones":
        st.title("➕ Gestión de Activos")
        t1, t2, t3 = st.tabs(["🆕 Añadir Nuevo", "💰 Operar", "✏️ Editar / Eliminar"])

        with t1:
            c1, c2 = st.columns([1, 1])
            with c1:
                q = st.text_input("🔍 Buscar (Nombre / ISIN / Ticker):", placeholder="Ej: AAPL, Amundi, MSFT...")
                if st.button("Buscar", type="primary") and q:
                    try:
                        res = search(sanitize_input(q))
                        if 'quotes' in res and res['quotes']:
                            st.session_state['s'] = res['quotes']
                        else:
                            st.warning("No se encontraron resultados.")
                            st.session_state['s'] = []
                    except Exception:
                        st.error("Error en la búsqueda. Inténtalo de nuevo.")
                        st.session_state['s'] = []

                if 's' in st.session_state and st.session_state['s']:
                    opts = {
                        f"{x['symbol']} | {x.get('shortname', x.get('longname', 'N/A'))} ({x.get('exchDisp', 'Unknown')})": x
                        for x in st.session_state['s']
                    }
                    if opts:
                        sel = st.selectbox("Selecciona un activo:", list(opts.keys()))
                        if sel in opts:
                            st.session_state['sel_add'] = opts[sel]

            with c2:
                if 'sel_add' in st.session_state:
                    tk = st.session_state['sel_add']['symbol']
                    try:
                        inf = yf.Ticker(tk).fast_info
                        p = inf['last_price']
                        mon = inf.get('currency', 'N/A')
                        if p:
                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                st.metric("Precio Actual", f"{p:.2f} {mon}")
                            with col_p2:
                                st.metric("Ticker", tk)
                            if mon and mon != 'EUR':
                                st.warning(f"Activo cotizado en **{mon}**. Introduce los importes en EUR.")

                            with st.form("new_asset_form"):
                                st.markdown("**Datos de la inversión:**")
                                inv_amount = st.number_input("Dinero Invertido (EUR)", min_value=0.0, step=10.0,
                                                              help="Cuánto dinero pusiste en total")
                                val_actual = st.number_input("Valor Actual (EUR)", min_value=0.0, step=10.0,
                                                              help="Cuánto vale hoy tu posición")
                                pl = st.selectbox("Broker / Plataforma",
                                                   ["MyInvestor", "XTB", "Trade Republic", "Degiro", "Interactive Brokers",
                                                    "eToro", "Revolut", "Otro"])
                                if st.form_submit_button("💾 Guardar Activo", use_container_width=True) and val_actual > 0:
                                    sh = val_actual / p
                                    av = inv_amount / sh if sh > 0 else 0
                                    nombre_activo = st.session_state['sel_add'].get('longname',
                                                     st.session_state['sel_add'].get('shortname', tk))
                                    add_asset_db(tk, nombre_activo, sh, av, pl)
                                    st.success(f"Añadido: {nombre_activo} ({sh:.4f} acciones)")
                                    time.sleep(1)
                                    st.rerun()
                    except Exception:
                        st.error("Error al obtener precio. Inténtalo de nuevo.")

        with t2:
            if df_final.empty:
                st.warning("No tienes activos. Añade uno primero.")
            else:
                c1, c2 = st.columns([1, 1])
                with c1:
                    nom = st.selectbox("Selecciona Activo:", df_final['Nombre'].unique())
                    row = df_final[df_final['Nombre'] == nom].iloc[0]
                    st.markdown(f"""
                    <div class='stat-card'>
                        <p>Posición actual</p>
                        <h2 style='color:#00CC96 !important;'>{row['Valor Acciones']:,.2f} €</h2>
                        <p>{row['shares']:.4f} accs. | P. medio: {row['avg_price']:.4f}</p>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    tipo = st.radio("Tipo de Operación:", ["🟢 Comprar", "🔴 Vender"], horizontal=True)
                    m = st.number_input("Importe (€)", min_value=0.0, step=10.0)
                    if m > 0:
                        precio = row['Precio Actual']
                        if precio <= 0:
                            st.error("Precio no disponible.")
                        else:
                            sh_op = m / precio
                            st.caption(f"≈ {sh_op:.4f} acciones a {precio:.4f}")
                            if "Comprar" in tipo:
                                if m > total_liquidez:
                                    st.error(f"Liquidez insuficiente. Disponible: {total_liquidez:,.2f}€")
                                elif st.button("Confirmar Compra", type="primary", use_container_width=True):
                                    navg = ((row['shares'] * row['avg_price']) + m) / (row['shares'] + sh_op)
                                    update_asset_db(int(row['id']), row['shares'] + sh_op, navg)
                                    update_liquidity_balance(int(cash_id), total_liquidez - m)
                                    log_transaction("COMPRA", row['ticker'], nom, sh_op, precio, row.get('platform', ''))
                                    st.toast(f"Compradas {sh_op:.4f} accs. de {nom}")
                                    time.sleep(0.5)
                                    st.rerun()
                            else:
                                max_venta = row['shares'] * precio
                                if m > max_venta * 1.01:
                                    st.error(f"No puedes vender más de lo que tienes ({max_venta:,.2f}€)")
                                elif st.button("Confirmar Venta", use_container_width=True):
                                    nsh = row['shares'] - sh_op
                                    if nsh < 0.001:
                                        delete_asset_db(int(row['id']))
                                    else:
                                        update_asset_db(int(row['id']), nsh, row['avg_price'])
                                    update_liquidity_balance(int(cash_id), total_liquidez + m)
                                    log_transaction("VENTA", row['ticker'], nom, sh_op, precio, row.get('platform', ''))
                                    st.toast(f"Vendidas {sh_op:.4f} accs. de {nom}")
                                    time.sleep(0.5)
                                    st.rerun()

        with t3:
            if not df_final.empty:
                e = st.selectbox("Selecciona activo a editar:", df_final['Nombre'], key='edd')
                er = df_final[df_final['Nombre'] == e].iloc[0]

                with st.form("edit_asset_form"):
                    st.markdown(f"**Editando: {e}** (`{er['ticker']}`)")
                    col_e1, col_e2 = st.columns(2)
                    with col_e1:
                        nsh = st.number_input("Nº Acciones", value=float(er['shares']), min_value=0.0, step=0.0001, format="%.4f")
                    with col_e2:
                        nav = st.number_input("Precio Medio", value=float(er['avg_price']), min_value=0.0, step=0.01, format="%.4f")

                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.form_submit_button("💾 Guardar Cambios", use_container_width=True):
                            update_asset_db(int(er['id']), nsh, nav)
                            st.toast("Activo actualizado")
                            time.sleep(0.5)
                            st.rerun()

                st.markdown("---")
                if st.button(f"🗑️ Eliminar {e}", type="secondary", use_container_width=True):
                    delete_asset_db(int(er['id']))
                    st.toast(f"Eliminado: {e}")
                    time.sleep(0.5)
                    st.rerun()
            else:
                st.info("No hay activos para editar.")

    # ==================================================================
    # HISTORIAL DE TRANSACCIONES
    # ==================================================================
    elif pagina == "📋 Historial":
        st.title("📋 Historial de Operaciones")

        if st.session_state['tx_log']:
            df_tx = pd.DataFrame(st.session_state['tx_log'])
            df_tx = df_tx.sort_values('fecha', ascending=False)

            # Filtro por tipo
            tipos_disp = ["Todos"] + list(df_tx['tipo'].unique())
            filtro_tipo = st.selectbox("Filtrar por tipo:", tipos_disp)
            if filtro_tipo != "Todos":
                df_tx = df_tx[df_tx['tipo'] == filtro_tipo]

            # Colores por tipo
            def color_tipo(val):
                colors_map = {
                    'COMPRA': 'color: #00CC96', 'VENTA': 'color: #EF553B',
                    'INGRESO': 'color: #636EFA', 'RETIRO': 'color: #FFA15A'
                }
                return colors_map.get(val, '')

            st.dataframe(
                df_tx.style.applymap(color_tipo, subset=['tipo']),
                use_container_width=True, height=min(600, 50 + len(df_tx) * 35)
            )

            # Resumen
            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Operaciones", len(df_tx))
            compras = df_tx[df_tx['tipo'] == 'COMPRA']['importe'].sum()
            ventas = df_tx[df_tx['tipo'] == 'VENTA']['importe'].sum()
            c2.metric("Total Compras", f"{compras:,.2f}€")
            c3.metric("Total Ventas", f"{ventas:,.2f}€")
            c4.metric("Flujo Neto", f"{ventas - compras:+,.2f}€")

            csv_tx = df_tx.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Exportar Historial", csv_tx, "historial.csv", "text/csv")
        else:
            st.info("No hay operaciones registradas en esta sesión. Las operaciones se registran automáticamente al comprar, vender, ingresar o retirar.")

    # ==================================================================
    # WATCHLIST
    # ==================================================================
    elif pagina == "🔍 Watchlist":
        st.title("🔍 Watchlist — Activos en Seguimiento")

        with st.expander("➕ Añadir a Watchlist", expanded=not bool(st.session_state['watchlist'])):
            wq = st.text_input("Buscar ticker:", placeholder="Ej: AAPL, MSFT, AMZN...", key="wl_search")
            if st.button("Buscar", key="wl_btn") and wq:
                try:
                    res = search(sanitize_input(wq))
                    if 'quotes' in res and res['quotes']:
                        for q_item in res['quotes'][:5]:
                            sym = q_item['symbol']
                            name = q_item.get('shortname', q_item.get('longname', sym))
                            if st.button(f"➕ {sym} — {name}", key=f"wl_add_{sym}"):
                                if sym not in [w['ticker'] for w in st.session_state['watchlist']]:
                                    st.session_state['watchlist'].append({
                                        'ticker': sym, 'name': name,
                                        'added': datetime.now().strftime("%Y-%m-%d")
                                    })
                                    st.toast(f"Añadido {sym} a watchlist")
                                    st.rerun()
                                else:
                                    st.warning(f"{sym} ya está en tu watchlist.")
                    else:
                        st.warning("Sin resultados.")
                except Exception:
                    st.error("Error en búsqueda.")

        if st.session_state['watchlist']:
            st.divider()
            wl_tickers = [w['ticker'] for w in st.session_state['watchlist']]
            wl_prices = get_real_time_prices(wl_tickers)

            cols_per_row = 3
            for i in range(0, len(st.session_state['watchlist']), cols_per_row):
                cols_wl = st.columns(cols_per_row)
                for j, col_wl in enumerate(cols_wl):
                    idx = i + j
                    if idx < len(st.session_state['watchlist']):
                        w = st.session_state['watchlist'][idx]
                        price = wl_prices.get(w['ticker'], 0)
                        with col_wl:
                            with st.container(border=True):
                                st.markdown(f"**{w['name']}**")
                                st.markdown(f"`{w['ticker']}` — {price:,.2f}")
                                c_a, c_b = st.columns(2)
                                with c_a:
                                    if st.button("📊 Info", key=f"wl_info_{idx}", use_container_width=True):
                                        st.session_state[f'wl_detail_{idx}'] = True
                                with c_b:
                                    if st.button("🗑️", key=f"wl_del_{idx}", use_container_width=True):
                                        st.session_state['watchlist'].pop(idx)
                                        st.rerun()

                                if st.session_state.get(f'wl_detail_{idx}', False):
                                    info = get_ticker_info(w['ticker'])
                                    if info:
                                        st.caption(f"Sector: {info.get('sector', 'N/A')}")
                                        st.caption(f"P/E: {info.get('pe_ratio', 0):.2f} | Beta: {info.get('beta', 0):.2f}")
                                        st.caption(f"Div Yield: {info.get('dividend_yield', 0) * 100:.2f}%")
                                        hi = info.get('fifty_two_week_high', 0)
                                        lo = info.get('fifty_two_week_low', 0)
                                        if hi and lo:
                                            pct_from_high = ((price - hi) / hi * 100) if hi else 0
                                            st.caption(f"52w: {lo:.2f} — {hi:.2f} ({pct_from_high:+.1f}%)")
        else:
            st.info("Tu watchlist está vacía. Busca activos arriba para añadirlos.")

    # ==================================================================
    # RADIOGRAFÍA (PORTFOLIO X-RAY)
    # ==================================================================
    elif pagina == "🔬 Radiografía":
        st.title("🔬 Radiografía de Cartera")

        if df_final.empty:
            st.warning("Añade activos para analizar tu cartera.")
        else:
            # Puntuación de diversificación
            weights = (df_final['Peso %'] / 100).tolist()
            div_score = diversification_score(weights)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Diversificación", f"{div_score:.0f}/100")
            c2.metric("📊 Nº Activos", len(df_final))

            top_weight = df_final['Peso %'].max()
            c3.metric("⚠️ Mayor Concentración", f"{top_weight:.1f}%",
                       "Alto" if top_weight > 30 else "OK")

            inv_total = df_final['Dinero Invertido'].sum()
            c4.metric("💰 Total Invertido", f"{inv_total:,.2f} €")

            st.divider()

            # Correlación
            tab_corr, tab_sector, tab_broker, tab_detail = st.tabs(
                ["📈 Correlación", "🏢 Sectores", "🏦 Brokers", "🔎 Detalle Activo"])

            with tab_corr:
                st.subheader("Matriz de Correlación")
                if not history_data.empty and len(history_data.columns) > 1:
                    valid_cols = [c for c in history_data.columns if c in my_tickers]
                    if len(valid_cols) > 1:
                        corr_data = history_data[valid_cols].pct_change().dropna()
                        if len(corr_data) > 10:
                            corr_matrix = corr_data.corr()
                            fig_corr = px.imshow(corr_matrix, text_auto='.2f',
                                                  color_continuous_scale=['#EF553B', '#1e1e1e', '#00CC96'],
                                                  aspect='equal', labels=dict(color="Correlación"))
                            fig_corr.update_layout(template="plotly_dark", height=450,
                                                    paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
                            st.plotly_chart(fig_corr, use_container_width=True)

                            # Alertas de correlación alta
                            high_corr = []
                            for ic in range(len(corr_matrix.columns)):
                                for jc in range(ic + 1, len(corr_matrix.columns)):
                                    val = corr_matrix.iloc[ic, jc]
                                    if abs(val) > 0.8:
                                        high_corr.append((corr_matrix.columns[ic], corr_matrix.columns[jc], val))
                            if high_corr:
                                st.warning("**Pares altamente correlacionados (>0.8):**")
                                for a, b, v in high_corr:
                                    st.write(f"- {a} ↔ {b}: **{v:.2f}**")
                        else:
                            st.info("Datos insuficientes para correlación.")
                    else:
                        st.info("Se necesitan al menos 2 activos para la correlación.")
                else:
                    st.info("Sin datos históricos suficientes.")

            with tab_sector:
                st.subheader("Distribución por Sector")
                with st.spinner("Obteniendo información de sectores..."):
                    sectors = {}
                    for _, r in df_final.iterrows():
                        info = get_ticker_info(r['ticker'])
                        sec = info.get('sector', 'Desconocido') if info else 'Desconocido'
                        sectors[sec] = sectors.get(sec, 0) + r['Valor Acciones']

                if sectors:
                    df_sectors = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Valor'])
                    df_sectors = df_sectors.sort_values('Valor', ascending=False)
                    df_sectors['Porcentaje'] = df_sectors['Valor'] / df_sectors['Valor'].sum() * 100

                    c_s1, c_s2 = st.columns([1, 1])
                    with c_s1:
                        fig_sec = px.bar(df_sectors, x='Sector', y='Valor', color='Sector',
                                          text=df_sectors['Porcentaje'].apply(lambda x: f"{x:.1f}%"),
                                          color_discrete_sequence=px.colors.qualitative.Set3)
                        fig_sec.update_layout(template="plotly_dark", height=350, showlegend=False,
                                               paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_sec, use_container_width=True)
                    with c_s2:
                        fig_sec_pie = px.pie(df_sectors, names='Sector', values='Valor', hole=0.5,
                                              color_discrete_sequence=px.colors.qualitative.Set3)
                        fig_sec_pie.update_layout(template="plotly_dark", height=350,
                                                   paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_sec_pie, use_container_width=True)

            with tab_broker:
                st.subheader("Distribución por Broker")
                if 'platform' in df_final.columns:
                    broker_data = df_final.groupby('platform').agg(
                        Valor=('Valor Acciones', 'sum'),
                        Activos=('ticker', 'count'),
                        PnL=('Ganancia', 'sum')
                    ).reset_index()
                    broker_data.columns = ['Broker', 'Valor', 'Nº Activos', 'P&L']

                    c_b1, c_b2 = st.columns([1, 1])
                    with c_b1:
                        fig_bk = px.pie(broker_data, names='Broker', values='Valor', hole=0.5,
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig_bk.update_layout(template="plotly_dark", height=350,
                                              paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig_bk, use_container_width=True)
                    with c_b2:
                        st.dataframe(broker_data.style.format({
                            'Valor': '{:,.2f}€', 'P&L': '{:+,.2f}€'
                        }), use_container_width=True)

            with tab_detail:
                st.subheader("Análisis Individual")
                sel_asset = st.selectbox("Selecciona activo:", df_final['Nombre'].tolist(), key="xray_asset")
                if sel_asset:
                    row_sel = df_final[df_final['Nombre'] == sel_asset].iloc[0]
                    ticker_sel = row_sel['ticker']

                    info_sel = get_ticker_info(ticker_sel)
                    c_d1, c_d2 = st.columns([1, 1])
                    with c_d1:
                        st.markdown(f"### {sel_asset}")
                        st.write(f"**Ticker:** {ticker_sel}")
                        if info_sel:
                            st.write(f"**Sector:** {info_sel.get('sector', 'N/A')}")
                            st.write(f"**Industria:** {info_sel.get('industry', 'N/A')}")
                            st.write(f"**País:** {info_sel.get('country', 'N/A')}")
                            st.write(f"**P/E:** {info_sel.get('pe_ratio', 0):.2f}")
                            st.write(f"**Beta:** {info_sel.get('beta', 0):.2f}")
                            dy = info_sel.get('dividend_yield', 0) * 100
                            st.write(f"**Dividendo:** {dy:.2f}%")
                            mcap = info_sel.get('market_cap', 0)
                            if mcap > 1e12:
                                st.write(f"**Market Cap:** {mcap/1e12:.2f}T")
                            elif mcap > 1e9:
                                st.write(f"**Market Cap:** {mcap/1e9:.2f}B")
                            elif mcap > 1e6:
                                st.write(f"**Market Cap:** {mcap/1e6:.2f}M")

                    with c_d2:
                        st.metric("Valor Actual", f"{row_sel['Valor Acciones']:,.2f} €")
                        st.metric("P&L", f"{row_sel['Ganancia']:+,.2f} €", f"{row_sel['Rentabilidad %']:+.2f}%")
                        st.metric("Peso en Cartera", f"{row_sel['Peso %']:.1f}%")

                    # Gráfico histórico individual
                    if not history_data.empty and ticker_sel in history_data.columns:
                        st.divider()
                        hist_asset = history_data[ticker_sel].dropna()
                        if not hist_asset.empty:
                            fig_ind = go.Figure()
                            fig_ind.add_trace(go.Scatter(x=hist_asset.index, y=hist_asset,
                                                          line=dict(color='#00CC96', width=2),
                                                          fill='tozeroy', fillcolor='rgba(0,204,150,0.05)',
                                                          name=ticker_sel))
                            # Línea de precio medio
                            fig_ind.add_hline(y=row_sel['avg_price'], line_dash="dash",
                                               line_color="#FFA15A", annotation_text=f"P. Medio: {row_sel['avg_price']:.2f}")
                            fig_ind.update_layout(template="plotly_dark", height=300,
                                                   paper_bgcolor='rgba(0,0,0,0)',
                                                   margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Precio",
                                                   hovermode="x unified")
                            st.plotly_chart(fig_ind, use_container_width=True)

    # ==================================================================
    # ASESOR AI
    # ==================================================================
    elif pagina == "💬 Asesor AI":
        st.title("💬 Carterapro AI — Asesor Financiero")

        # Contexto enriquecido
        ctx_parts = [f"Patrimonio total: {patrimonio_total:,.2f}€.", f"Liquidez: {total_liquidez:,.2f}€."]
        if not df_final.empty:
            ctx_parts.append(f"Total inversiones: {total_inversiones:,.2f}€.")
            ctx_parts.append(f"Nº activos: {len(df_final)}.")
            pnl = df_final['Ganancia'].sum()
            ctx_parts.append(f"P&L total: {pnl:+,.2f}€.")
            for _, r in df_final.iterrows():
                ctx_parts.append(
                    f"[{r['Nombre']} ({r['ticker']}): Valor {r['Valor Acciones']:.2f}€, "
                    f"P&L {r['Ganancia']:+.2f}€ ({r['Rentabilidad %']:+.1f}%), Peso {r['Peso %']:.1f}%]"
                )
        ctx = " ".join(ctx_parts)

        # Quick actions
        st.markdown("**Preguntas rápidas:**")
        quick_cols = st.columns(4)
        quick_prompts = [
            "¿Cómo de diversificada está mi cartera?",
            "¿Qué activo debería vender?",
            "¿Debo aumentar la liquidez?",
            "Analiza mi riesgo actual"
        ]
        for i, col_q in enumerate(quick_cols):
            with col_q:
                if st.button(quick_prompts[i], key=f"quick_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": quick_prompts[i]})
                    st.rerun()

        st.divider()

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        if prompt := st.chat_input("Pregúntame sobre tu cartera..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if HAS_GROQ:
                with st.chat_message("assistant"):
                    try:
                        client = openai.OpenAI(base_url="https://api.groq.com/openai/v1",
                                                api_key=st.secrets["GROQ_API_KEY"])
                        system_prompt = (
                            "Eres un asesor financiero experto y profesional. "
                            "Analizas la cartera del usuario con datos reales. "
                            "Responde de forma clara, estructurada y con recomendaciones accionables. "
                            "Usa bullet points cuando sea útil. No inventes datos. "
                            "Si no tienes info suficiente, pide más contexto. "
                            f"Datos del usuario: {ctx}"
                        )
                        stream = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "system", "content": system_prompt}] + st.session_state.messages,
                            stream=True, temperature=0.4, max_tokens=1500
                        )
                        response = st.write_stream(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"Error al conectar con IA: {e}")
            else:
                st.warning("Configura GROQ_API_KEY en los secretos para usar el asesor AI.")

        if st.session_state.messages:
            if st.button("🗑️ Limpiar Chat", use_container_width=False):
                st.session_state.messages = []
                st.rerun()

    # ==================================================================
    # MONTE CARLO
    # ==================================================================
    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Simulación Monte Carlo — Proyección Patrimonial")

        st.markdown("""
        Simulación estocástica que proyecta el valor futuro de tu cartera basándose en la
        distribución histórica de retornos. Se muestran los percentiles P10, P50 y P90.
        """)

        c_mc1, c_mc2, c_mc3 = st.columns(3)
        with c_mc1:
            ys = st.slider("Horizonte (años)", 1, 30, 10)
        with c_mc2:
            n_sims = st.select_slider("Nº Simulaciones", options=[100, 500, 1000, 5000], value=1000)
        with c_mc3:
            aport_mensual = st.number_input("Aportación mensual (€)", 0.0, step=50.0, value=0.0)

        if st.button("🚀 Ejecutar Simulación", type="primary", use_container_width=True):
            mu, sigma = 0.07, 0.15
            if not history_data.empty:
                d = history_data.pct_change().dropna()
                if not d.empty and len(d) > 20:
                    mu = d.mean().mean() * 252
                    sigma = d.mean(axis=1).std() * np.sqrt(252)
                    mu = max(-0.3, min(0.5, mu))
                    sigma = max(0.05, min(0.8, sigma))

            st.info(f"Parámetros: μ={mu:.4f} (ret. anual), σ={sigma:.4f} (vol. anual)")

            n_steps = int(ys * 252)
            capital_inicial = max(total_inversiones, 1)

            with st.spinner("Simulando..."):
                all_paths = np.zeros((n_sims, n_steps + 1))
                all_paths[:, 0] = capital_inicial

                dt_sim = 1 / 252
                drift = (mu - 0.5 * sigma ** 2) * dt_sim
                diffusion = sigma * np.sqrt(dt_sim)
                aport_diario = aport_mensual * 12 / 252

                random_shocks = np.random.normal(0, 1, (n_sims, n_steps))
                for step in range(n_steps):
                    all_paths[:, step + 1] = (all_paths[:, step] * np.exp(drift + diffusion * random_shocks[:, step])
                                               + aport_diario)

            x = np.linspace(0, ys, n_steps + 1)
            p10 = np.percentile(all_paths, 10, axis=0)
            p25 = np.percentile(all_paths, 25, axis=0)
            p50 = np.percentile(all_paths, 50, axis=0)
            p75 = np.percentile(all_paths, 75, axis=0)
            p90 = np.percentile(all_paths, 90, axis=0)

            fig = go.Figure()

            # Banda P10-P90
            fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                      y=np.concatenate([p90, p10[::-1]]),
                                      fill='toself', fillcolor='rgba(0,204,150,0.08)',
                                      line=dict(color='rgba(0,0,0,0)'), name='P10-P90', showlegend=True))

            # Banda P25-P75
            fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                                      y=np.concatenate([p75, p25[::-1]]),
                                      fill='toself', fillcolor='rgba(0,204,150,0.15)',
                                      line=dict(color='rgba(0,0,0,0)'), name='P25-P75', showlegend=True))

            # Mediana
            fig.add_trace(go.Scatter(x=x, y=p50, mode='lines',
                                      line=dict(color='#00CC96', width=3), name='Mediana (P50)'))

            # Muestra de caminos
            sample_indices = np.random.choice(n_sims, min(20, n_sims), replace=False)
            for si in sample_indices:
                fig.add_trace(go.Scatter(x=x, y=all_paths[si], mode='lines',
                                          line=dict(color='rgba(255,255,255,0.03)', width=0.5), showlegend=False))

            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=450,
                               title=f"Proyección a {ys} años — {n_sims} simulaciones",
                               xaxis_title="Años", yaxis_title="Valor (€)", hovermode="x unified",
                               margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            # Resultados finales
            st.divider()
            st.subheader("📊 Distribución del Valor Final")
            c_r1, c_r2, c_r3, c_r4, c_r5 = st.columns(5)
            c_r1.metric("Pesimista (P10)", f"{p10[-1]:,.0f} €")
            c_r2.metric("P25", f"{p25[-1]:,.0f} €")
            c_r3.metric("Mediana (P50)", f"{p50[-1]:,.0f} €")
            c_r4.metric("P75", f"{p75[-1]:,.0f} €")
            c_r5.metric("Optimista (P90)", f"{p90[-1]:,.0f} €")

            total_aportado = capital_inicial + aport_mensual * 12 * ys
            ganancia_mediana = p50[-1] - total_aportado
            st.markdown(f"""
            ---
            - **Capital inicial:** {capital_inicial:,.2f}€
            - **Aportaciones totales:** {aport_mensual * 12 * ys:,.2f}€
            - **Total aportado:** {total_aportado:,.2f}€
            - **Ganancia mediana estimada:** {ganancia_mediana:+,.2f}€ ({(ganancia_mediana / total_aportado * 100):+.1f}%)
            - **Probabilidad de pérdida:** {(np.sum(all_paths[:, -1] < total_aportado) / n_sims * 100):.1f}%
            """)

    # ==================================================================
    # REBALANCEO
    # ==================================================================
    elif pagina == "⚖️ Rebalanceo":
        st.title("⚖️ Herramienta de Rebalanceo")

        if df_final.empty:
            st.warning("Añade activos para usar el rebalanceo.")
        else:
            tab_manual, tab_auto = st.tabs(["📐 Manual", "🤖 Estrategias"])

            with tab_manual:
                st.markdown("Establece los pesos objetivo para cada activo. El total debe ser 100%.")

                c_inputs, c_results = st.columns([1, 1.5])
                ws = {}
                tot_w = 0
                with c_inputs:
                    for idx_r, r_row in df_final.iterrows():
                        col_n, col_w, col_c = st.columns([2, 1, 1])
                        with col_n:
                            st.write(f"**{r_row['Nombre']}**")
                            st.caption(f"Actual: {r_row['Peso %']:.1f}%")
                        with col_w:
                            w = st.number_input("Objetivo %", 0, 100, int(round(r_row['Peso %'])),
                                                 key=f"reb_{idx_r}")
                            ws[r_row['Nombre']] = w
                            tot_w += w
                        with col_c:
                            diff = w - r_row['Peso %']
                            color_d = "#00CC96" if abs(diff) < 2 else "#FFA15A"
                            st.markdown(f"<span style='color:{color_d}'>{diff:+.1f}%</span>", unsafe_allow_html=True)

                    st.divider()
                    color_tot = "#00CC96" if tot_w == 100 else "#EF553B"
                    st.markdown(f"**Total: <span style='color:{color_tot}'>{tot_w}%</span>**", unsafe_allow_html=True)

                with c_results:
                    if tot_w == 100:
                        if st.button("📊 Calcular Rebalanceo", type="primary", use_container_width=True):
                            rebalance_data = []
                            for _, r_row in df_final.iterrows():
                                target_val = patrimonio_total * ws[r_row['Nombre']] / 100
                                diff_val = target_val - r_row['Valor Acciones']
                                rebalance_data.append({
                                    'Activo': r_row['Nombre'],
                                    'Actual €': r_row['Valor Acciones'],
                                    'Actual %': r_row['Peso %'],
                                    'Objetivo %': ws[r_row['Nombre']],
                                    'Objetivo €': target_val,
                                    'Operación €': diff_val,
                                    'Acción': 'COMPRAR' if diff_val > 0 else ('VENDER' if diff_val < 0 else 'OK')
                                })

                            df_reb = pd.DataFrame(rebalance_data)

                            st.dataframe(df_reb.style.format({
                                'Actual €': '{:,.2f}', 'Objetivo €': '{:,.2f}',
                                'Actual %': '{:.1f}%', 'Objetivo %': '{:.0f}%',
                                'Operación €': '{:+,.2f}'
                            }).applymap(lambda x: 'color: #00CC96' if x == 'COMPRAR'
                                         else ('color: #EF553B' if x == 'VENDER' else ''),
                                         subset=['Acción']), use_container_width=True)

                            # Gráfico comparación
                            fig_reb = go.Figure()
                            fig_reb.add_trace(go.Bar(name='Actual', x=df_reb['Activo'], y=df_reb['Actual %'],
                                                      marker_color='#636EFA'))
                            fig_reb.add_trace(go.Bar(name='Objetivo', x=df_reb['Activo'], y=df_reb['Objetivo %'],
                                                      marker_color='#00CC96'))
                            fig_reb.update_layout(template="plotly_dark", barmode='group', height=350,
                                                   paper_bgcolor='rgba(0,0,0,0)', yaxis_title="%")
                            st.plotly_chart(fig_reb, use_container_width=True)

                            # Resumen de operaciones necesarias
                            compras = df_reb[df_reb['Operación €'] > 0]['Operación €'].sum()
                            ventas = abs(df_reb[df_reb['Operación €'] < 0]['Operación €'].sum())
                            st.markdown(f"""
                            **Resumen:**
                            - Compras necesarias: **{compras:,.2f}€**
                            - Ventas necesarias: **{ventas:,.2f}€**
                            - Liquidez actual: **{total_liquidez:,.2f}€**
                            - {"✅ Liquidez suficiente" if total_liquidez >= compras - ventas else "⚠️ Necesitas más liquidez"}
                            """)
                    else:
                        st.warning(f"Los pesos deben sumar 100% (actual: {tot_w}%)")

            with tab_auto:
                st.subheader("Estrategias Predefinidas")
                estrategia = st.selectbox("Selecciona estrategia:", [
                    "Equiponderado (pesos iguales)",
                    "Por Momentum (mayor peso a ganadores)",
                    "Por Valor (mayor peso a perdedores — contrarian)",
                    "Mínima Volatilidad"
                ])

                if st.button("Aplicar Estrategia", type="primary"):
                    n_assets = len(df_final)
                    if estrategia == "Equiponderado (pesos iguales)":
                        target_weights = {r['Nombre']: 100 / n_assets for _, r in df_final.iterrows()}
                    elif estrategia == "Por Momentum (mayor peso a ganadores)":
                        rents = df_final['Rentabilidad %'].clip(lower=0)
                        total_r = rents.sum()
                        if total_r > 0:
                            target_weights = {r['Nombre']: (max(0, r['Rentabilidad %']) / total_r * 100)
                                               for _, r in df_final.iterrows()}
                        else:
                            target_weights = {r['Nombre']: 100 / n_assets for _, r in df_final.iterrows()}
                    elif estrategia == "Por Valor (mayor peso a perdedores — contrarian)":
                        inv_rents = df_final['Rentabilidad %'].max() - df_final['Rentabilidad %'] + 1
                        total_ir = inv_rents.sum()
                        target_weights = {}
                        for (_, r), ir in zip(df_final.iterrows(), inv_rents):
                            target_weights[r['Nombre']] = ir / total_ir * 100
                    else:  # Mínima Volatilidad
                        if not history_data.empty:
                            vols = {}
                            for _, r in df_final.iterrows():
                                if r['ticker'] in history_data.columns:
                                    vol = history_data[r['ticker']].pct_change().std()
                                    vols[r['Nombre']] = 1 / max(vol, 0.001)
                                else:
                                    vols[r['Nombre']] = 1
                            total_inv_vol = sum(vols.values())
                            target_weights = {k: v / total_inv_vol * 100 for k, v in vols.items()}
                        else:
                            target_weights = {r['Nombre']: 100 / n_assets for _, r in df_final.iterrows()}

                    # Mostrar resultado
                    strat_data = []
                    for _, r in df_final.iterrows():
                        tw = target_weights.get(r['Nombre'], 0)
                        target_val = patrimonio_total * tw / 100
                        diff_val = target_val - r['Valor Acciones']
                        strat_data.append({
                            'Activo': r['Nombre'], 'Actual %': r['Peso %'],
                            'Objetivo %': tw, 'Diff %': tw - r['Peso %'],
                            'Operación €': diff_val
                        })

                    df_strat = pd.DataFrame(strat_data)
                    st.dataframe(df_strat.style.format({
                        'Actual %': '{:.1f}', 'Objetivo %': '{:.1f}',
                        'Diff %': '{:+.1f}', 'Operación €': '{:+,.2f}'
                    }), use_container_width=True)

# ==================================================================
# NOTICIAS (SIDEBAR)
# ==================================================================
if st.session_state['show_news']:
    with col_news:
        c_n1, c_n2 = st.columns([3, 1])
        with c_n1:
            st.markdown("### 📰 Noticias")
        with c_n2:
            if st.button("🔄", key="ref"):
                clear_cache()
                st.rerun()

        tm = {'Hoy': 'd', 'Semana': 'w'}
        sel_time = st.pills("Filtro", list(tm.keys()), default="Hoy", label_visibility="collapsed")
        news = get_global_news(my_tickers, tm.get(sel_time, 'd'))

        h = ""
        if news:
            for n in news:
                im = f"<img src='{n['image']}' class='news-img' onerror=\"this.style.display='none'\"/>" if n.get('image') else ""
                h += f"""<div class="news-card">{im}
                    <div class="news-source">{n.get('source', 'Web')} • {n.get('date', '')[:10]}</div>
                    <div class="news-title"><a href="{n['url']}" target="_blank">{n['title']}</a></div>
                </div>"""
        else:
            h = "<div style='text-align:center;color:#666;padding:20px'>Sin noticias disponibles</div>"

        st.markdown(f"<div class='news-scroll-area'>{h}</div>", unsafe_allow_html=True)
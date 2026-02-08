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

# --- CONFIGURACION GLOBAL ---
st.set_page_config(page_title="Carterapro", layout="wide", page_icon="", initial_sidebar_state="expanded")

# ==============================================================================
# DISENO
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --bg-primary: #0a0e14;
        --bg-secondary: #141920;
        --bg-card: #1a1f29;
        --bg-card-hover: #212733;
        --border: #2a3040;
        --border-hover: #00CC96;
        --text-primary: #EAEAEA;
        --text-secondary: #8b949e;
        --accent: #00CC96;
        --accent-dark: #007d5c;
        --red: #EF553B;
        --blue: #636EFA;
        --orange: #FFA15A;
        --purple: #AB63FA;
    }

    .stApp { background-color: var(--bg-primary); color: var(--text-primary); font-family: 'Inter', sans-serif; }
    h1, h2, h3, p, div, span, label { color: var(--text-primary) !important; }
    section[data-testid="stSidebar"] { background-color: var(--bg-secondary); border-right: 1px solid var(--border); }

    /* KPIs */
    div[data-testid="stMetric"] {
        background-color: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
        padding: 14px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem !important; color: white !important; font-weight: 700; }
    div[data-testid="stMetricDelta"] svg { display: none; }

    /* Botones */
    .stButton>button {
        border-radius: 8px; font-weight: 600; border: 1px solid var(--border);
        background-color: var(--bg-card); color: white; transition: all 0.2s;
    }
    .stButton>button:hover { border-color: var(--accent); color: var(--accent); transform: translateY(-1px); }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
        border: none; color: #000 !important; font-weight: 700;
    }
    .stButton>button[kind="primary"]:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,204,150,0.3); }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background-color: var(--bg-secondary); border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 8px; border: none; color: var(--text-secondary); font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: var(--bg-card) !important; color: white !important; }

    /* Selectbox / Inputs */
    .stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input {
        background-color: var(--bg-card) !important; border-color: var(--border) !important;
        color: white !important; border-radius: 8px !important;
    }

    /* Expanders */
    .streamlit-expanderHeader { background-color: var(--bg-card) !important; border-radius: 8px !important; }

    /* Graficos */
    .js-plotly-plot .plotly .main-svg { background-color: rgba(0,0,0,0) !important; }

    /* Stat Cards */
    .stat-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-secondary) 100%);
        border: 1px solid var(--border); border-radius: 12px; padding: 20px;
        text-align: center; transition: all 0.2s;
    }
    .stat-card:hover { transform: translateY(-2px); border-color: var(--accent); }
    .stat-card h2 { margin: 0; font-size: 2rem; }
    .stat-card p { margin: 5px 0 0 0; color: var(--text-secondary) !important; font-size: 0.85rem; }

    /* Market Ticker */
    .market-ticker {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: 10px;
        padding: 12px 16px; margin-bottom: 8px; transition: all 0.2s; cursor: pointer;
    }
    .market-ticker:hover { border-color: var(--accent); transform: translateX(2px); }

    /* News */
    .news-card-full {
        background-color: var(--bg-card); border-radius: 10px; padding: 16px; margin-bottom: 12px;
        border: 1px solid var(--border); transition: all 0.2s;
    }
    .news-card-full:hover { transform: translateY(-2px); border-color: var(--accent); }
    .news-card-full img { width: 100%; height: 140px; object-fit: cover; border-radius: 8px; margin-bottom: 10px; }
    .news-card-full .news-title a { color: #FFF !important; text-decoration: none; font-weight: 600; font-size: 0.95rem; line-height: 1.4; }
    .news-card-full .news-title a:hover { color: var(--accent) !important; }
    .news-card-full .news-meta { font-size: 0.72rem; color: var(--text-secondary); margin-top: 6px; }
    .news-card-full .news-source-badge {
        display: inline-block; background: var(--bg-secondary); border: 1px solid var(--border);
        border-radius: 4px; padding: 2px 6px; font-size: 0.65rem; color: var(--accent);
        text-transform: uppercase; font-weight: 700; margin-bottom: 6px;
    }

    /* Section headers */
    .section-header {
        display: flex; align-items: center; gap: 8px; margin-bottom: 12px;
        padding-bottom: 8px; border-bottom: 1px solid var(--border);
    }
    .section-header h3 { margin: 0 !important; font-size: 1.1rem; }

    /* Recommendation cards */
    .reco-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, #1a2430 100%);
        border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 10px;
        border-left: 3px solid var(--accent);
    }
    .reco-card .reco-ticker { font-size: 1rem; font-weight: 700; color: var(--accent) !important; }
    .reco-card .reco-name { font-size: 0.82rem; color: var(--text-secondary) !important; }
    .reco-card .reco-reason { font-size: 0.78rem; color: var(--text-primary) !important; margin-top: 6px; line-height: 1.3; }
</style>
""", unsafe_allow_html=True)

# --- CONEXIONES ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except Exception:
    st.error("Error Critico: Configura SUPABASE_URL y SUPABASE_KEY en los secretos.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- SESION ---
for k, v in {'user': None, 'tx_log': []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# LOGIN - MULTIUSUARIO
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
    st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
    _c1, _c2, _c3 = st.columns([1.2, 1.6, 1.2])
    with _c2:
        st.markdown("""
        <div style='text-align:center; padding: 40px 30px; background: linear-gradient(135deg, #141920, #1a1f29);
                     border-radius: 16px; border: 1px solid #2a3040;'>
            <h1 style='color: #00CC96 !important; font-size: 2.8rem; margin-bottom: 4px;'>Carterapro</h1>
            <p style='color: #8b949e !important; font-size: 1rem;'>Tu cartera de inversiones bajo control</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Iniciar Sesion", "Crear Cuenta"])
        with tab_login:
            if st.button("Entrar con Google", type="primary", use_container_width=True):
                try:
                    data = supabase.auth.sign_in_with_oauth({"provider": "google", "options": {"redirect_to": "https://carterapro.streamlit.app"}})
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("login_form"):
                em = st.text_input("Email")
                pa = st.text_input("Contrasena", type="password")
                if st.form_submit_button("Entrar", use_container_width=True, type="primary"):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": em, "password": pa})
                        st.session_state['user'] = res.user
                        st.rerun()
                    except Exception:
                        st.error("Credenciales incorrectas.")
        with tab_signup:
            with st.form("signup_form"):
                em2 = st.text_input("Email")
                pa2 = st.text_input("Contrasena", type="password")
                pa3 = st.text_input("Confirmar contrasena", type="password")
                if st.form_submit_button("Registrarse", use_container_width=True, type="primary"):
                    if pa2 != pa3:
                        st.error("Las contrasenas no coinciden.")
                    elif len(pa2) < 6:
                        st.error("Minimo 6 caracteres.")
                    else:
                        try:
                            supabase.auth.sign_up({"email": em2, "password": pa2})
                            st.success("Cuenta creada. Revisa tu email.")
                        except Exception as e:
                            st.error(f"Error: {e}")
    st.stop()

user = st.session_state['user']

# ==============================================================================
# MOTOR DE DATOS
# ==============================================================================

def sanitize_input(text):
    return re.sub(r'[^\w\s\-\.]', '', str(text)).strip().upper()

def safe_metric_calc(price_series):
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
    downside = returns[returns < 0]
    down_std = downside.std() * np.sqrt(252)
    if down_std < 0.001:
        return 0
    return (returns.mean() * 252 - rf) / down_std

def calc_var(returns, confidence=0.05):
    if len(returns) < 10:
        return 0
    return np.percentile(returns, confidence * 100)

def diversification_score(weights):
    if len(weights) == 0:
        return 0
    hhi = sum(w ** 2 for w in weights)
    n = len(weights)
    min_hhi = 1.0 / n if n > 0 else 1.0
    if 1.0 == min_hhi:
        return 100
    return max(0, min(100, (1 - (hhi - min_hhi) / (1.0 - min_hhi)) * 100))

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
        return assets, 0.0, 0
    return assets, liq_res[0]['amount'], liq_res[0]['id']

def get_real_time_prices(tickers):
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

@st.cache_data(ttl=600)
def get_ticker_info(ticker):
    try:
        info = yf.Ticker(ticker).info
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
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_market_news(query_terms, time_filter='d', max_results=12):
    results = []
    queries = []
    if query_terms:
        batch = " ".join(query_terms[:3])
        queries.append(f"{batch} stock market news")
        queries.append(f"{batch} bolsa cotizacion")
    queries.append("mercados financieros bolsa Europa USA")
    queries.append("S&P 500 NASDAQ market today")

    seen_titles = set()
    for q in queries:
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.news(q, region="wt-wt", safesearch="off", timelimit=time_filter, max_results=max_results))
                for n in raw:
                    title = n.get('title', '')
                    if title and n.get('url') and title not in seen_titles:
                        seen_titles.add(title)
                        results.append({
                            'title': title, 'source': n.get('source', 'Web'),
                            'date': n.get('date', ''), 'url': n.get('url'),
                            'image': n.get('image', None),
                            'body': n.get('body', '')[:200]
                        })
        except Exception:
            pass
    return results[:20]

@st.cache_data(ttl=1800)
def get_market_indices():
    indices = {
        '^GSPC': 'S&P 500', '^IXIC': 'NASDAQ', '^DJI': 'Dow Jones',
        '^STOXX50E': 'Euro Stoxx 50', '^IBEX': 'IBEX 35', '^GDAXI': 'DAX'
    }
    data = []
    for ticker, name in indices.items():
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period='5d')
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                data.append({'ticker': ticker, 'name': name, 'price': current, 'change': change})
        except Exception:
            pass
    return data

@st.cache_data(ttl=1800)
def get_trending_assets():
    popular = [
        ('VOO', 'Vanguard S&P 500 ETF', 'ETF indexado al S&P 500, la base de cualquier cartera'),
        ('VT', 'Vanguard Total World Stock', 'Exposicion global a todo el mercado mundial'),
        ('QQQ', 'Invesco QQQ Trust', 'ETF del NASDAQ-100, foco en tecnologia'),
        ('SCHD', 'Schwab US Dividend', 'ETF de dividendos de calidad USA'),
        ('VWO', 'Vanguard FTSE Emerging', 'Mercados emergentes para diversificar'),
        ('BND', 'Vanguard Total Bond Market', 'Renta fija USA para reducir volatilidad'),
        ('GLD', 'SPDR Gold Shares', 'Oro como cobertura contra inflacion'),
        ('AAPL', 'Apple Inc.', 'Mega-cap tecnologica con crecimiento estable'),
        ('MSFT', 'Microsoft Corp.', 'Lider en cloud y AI enterprise'),
        ('AMZN', 'Amazon.com Inc.', 'E-commerce + AWS cloud dominante'),
    ]
    results = []
    for ticker, name, reason in popular:
        try:
            fi = yf.Ticker(ticker).fast_info
            price = fi.get('last_price', 0)
            if price:
                results.append({
                    'ticker': ticker, 'name': name, 'reason': reason,
                    'price': price, 'currency': fi.get('currency', 'USD')
                })
        except Exception:
            results.append({'ticker': ticker, 'name': name, 'reason': reason, 'price': 0, 'currency': 'USD'})
    return results

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
        st.error(f"Error al guardar: {e}")

def update_asset_db(asset_id, shares, avg_price):
    try:
        supabase.table('assets').update({"shares": shares, "avg_price": avg_price}).eq('id', asset_id).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error al actualizar: {e}")

def delete_asset_db(id_del):
    try:
        supabase.table('assets').delete().eq('id', id_del).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error al eliminar: {e}")

def update_liquidity_balance(liq_id, new_amount):
    try:
        supabase.table('liquidity').update({"amount": new_amount}).eq('id', liq_id).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error liquidez: {e}")

def log_transaction(tipo, ticker, nombre, shares, price, platform=""):
    st.session_state['tx_log'].append({
        'fecha': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'tipo': tipo, 'ticker': ticker, 'nombre': nombre,
        'shares': round(shares, 6), 'precio': round(price, 4),
        'importe': round(shares * price, 2), 'platform': platform
    })

# --- CARGA DE DATOS DEL USUARIO ---
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
        lambda r: (r['Ganancia'] / r['Dinero Invertido'] * 100) if r['Dinero Invertido'] > 0 else 0, axis=1)
    total_inv_val = df_assets['Valor Acciones'].sum()
    df_assets['Peso %'] = df_assets.apply(
        lambda r: (r['Valor Acciones'] / total_inv_val * 100) if total_inv_val > 0 else 0, axis=1)
    df_final = df_assets.rename(columns={'nombre': 'Nombre'})

    if not history_raw.empty:
        if 'SPY' in history_raw.columns:
            benchmark_data = history_raw['SPY']
            history_data = history_raw.drop(columns=['SPY'], errors='ignore')
        else:
            history_data = history_raw

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0.0
patrimonio_total = total_inversiones + total_liquidez

# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    avatar = user.user_metadata.get('avatar_url', '') if user.user_metadata else ''
    nombre_user = user.user_metadata.get('full_name', '') if user.user_metadata else ''
    if not nombre_user:
        nombre_user = (user.email or 'Inversor').split('@')[0].capitalize()

    if avatar:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; padding:12px; background:var(--bg-card);
                     border-radius:10px; border:1px solid var(--border); margin-bottom:16px;'>
            <img src='{avatar}' style='width:36px; height:36px; border-radius:50%; border:2px solid var(--accent);'>
            <div style='line-height:1.2; overflow:hidden;'>
                <b style='color:white; font-size:0.9rem;'>{nombre_user}</b><br>
                <span style='font-size:0.7em; color:var(--accent);'>Online</span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding:12px; background:var(--bg-card); border-radius:10px;
                     border:1px solid var(--border); margin-bottom:16px;'>
            <b style='color:white;'>{nombre_user}</b><br>
            <span style='font-size:0.7em; color:var(--accent);'>Online</span>
        </div>""", unsafe_allow_html=True)

    # Patrimonio rapido
    st.markdown(f"""
    <div style='text-align:center; padding:10px; background:var(--bg-card); border-radius:10px;
                 border:1px solid var(--border); margin-bottom:16px;'>
        <span style='color:var(--text-secondary) !important; font-size:0.75rem;'>PATRIMONIO</span><br>
        <span style='color:var(--accent) !important; font-size:1.4rem; font-weight:800;'>{patrimonio_total:,.2f} EUR</span>
    </div>""", unsafe_allow_html=True)

    if st.button("Actualizar", use_container_width=True, type="primary"):
        clear_cache()
        st.rerun()

    st.divider()

    pagina = st.radio("", [
        "Dashboard",
        "Mercado & Noticias",
        "Inversiones",
        "Liquidez",
        "Rebalanceo",
        "Radiografia",
        "Simulador",
        "Historial",
    ], label_visibility="collapsed")

    st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)
    if st.button("Cerrar Sesion", use_container_width=True):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.rerun()

# ==============================================================================
# PAGINAS
# ==============================================================================

# ======================================================================
# DASHBOARD
# ======================================================================
if pagina == "Dashboard":
    st.markdown("## Dashboard")

    # Periodo
    periodos = {"1M": 30, "3M": 90, "6M": 180, "1A": 365, "2A": 730, "MAX": 9999}
    if 'periodo_sel' not in st.session_state:
        st.session_state['periodo_sel'] = '1A'
    col_period = st.columns(len(periodos) + 1)
    for i, (label, dias) in enumerate(periodos.items()):
        with col_period[i]:
            if st.button(label, key=f"per_{label}", use_container_width=True,
                         type="primary" if st.session_state['periodo_sel'] == label else "secondary"):
                st.session_state['periodo_sel'] = label
                st.rerun()

    periodo = st.session_state.get('periodo_sel', '1A')
    dias_periodo = periodos.get(periodo, 365)
    start_date = (datetime.now() - timedelta(days=dias_periodo)).date()

    # KPIs principales
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Patrimonio", f"{patrimonio_total:,.2f} EUR")
    if total_inversiones > 0:
        pnl_total = df_final['Ganancia'].sum()
        dinero_inv = df_final['Dinero Invertido'].sum()
        rent_pct = (pnl_total / dinero_inv * 100) if dinero_inv > 0 else 0
        k2.metric("P&L", f"{pnl_total:+,.2f} EUR", f"{rent_pct:+.2f}%",
                   delta_color="normal" if pnl_total >= 0 else "inverse")
    else:
        k2.metric("P&L", "0.00 EUR")
    k3.metric("Liquidez", f"{total_liquidez:,.2f} EUR")
    k4.metric("Activos", f"{len(df_final)}")

    # Metricas avanzadas
    vol_anual = 0; sharpe_ratio = 0; max_drawdown = 0; beta_portfolio = 1.0
    sortino = 0; var_95 = 0; alpha_jensen = 0
    daily_returns = pd.Series(dtype=float)

    if not history_data.empty:
        dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
        hist_filt = history_data[history_data.index >= dt_start].copy()
        if not hist_filt.empty and len(hist_filt) > 5:
            daily_returns = hist_filt.pct_change().mean(axis=1).dropna()
            if len(daily_returns) > 5:
                port_prices = (1 + daily_returns).cumprod()
                _, vol_dec, max_dd_dec, sharpe_ratio = safe_metric_calc(port_prices)
                vol_anual = vol_dec * 100
                max_drawdown = max_dd_dec * 100
                sortino = calc_sortino(daily_returns)
                var_95 = calc_var(daily_returns, 0.05) * 100
                if not benchmark_data.empty:
                    bench_filt = benchmark_data[benchmark_data.index >= dt_start]
                    bench_ret = bench_filt.pct_change().dropna()
                    common = daily_returns.index.intersection(bench_ret.index)
                    if len(common) > 10:
                        cov_val = daily_returns.loc[common].cov(bench_ret.loc[common])
                        var_val = bench_ret.loc[common].var()
                        beta_portfolio = cov_val / var_val if var_val != 0 else 1.0
                        rf_d = 0.03 / 252
                        alpha_jensen = ((daily_returns.loc[common].mean() - rf_d)
                                        - beta_portfolio * (bench_ret.loc[common].mean() - rf_d)) * 252

    k5.metric("Sharpe", f"{sharpe_ratio:.2f}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volatilidad", f"{vol_anual:.1f}%")
    m2.metric("Max DD", f"{max_drawdown:.1f}%")
    m3.metric("Beta", f"{beta_portfolio:.2f}")
    m4.metric("Sortino", f"{sortino:.2f}")

    st.markdown("---")

    # Charts
    col_chart, col_alloc = st.columns([2.2, 1])
    with col_chart:
        st.markdown("##### Rendimiento vs S&P 500")
        if not history_data.empty:
            dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
            hist_filt = history_data[history_data.index >= dt_start].copy()
            if not hist_filt.empty and len(hist_filt) > 2:
                port_ret = hist_filt.pct_change().mean(axis=1).fillna(0)
                port_cum = (1 + port_ret).cumprod() * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Tu Cartera",
                                         line=dict(color='#00CC96', width=2.5),
                                         fill='tozeroy', fillcolor='rgba(0,204,150,0.06)'))
                if not benchmark_data.empty:
                    bench_filt = benchmark_data[benchmark_data.index >= dt_start].copy()
                    if not bench_filt.empty:
                        bench_cum = (1 + bench_filt.pct_change().fillna(0)).cumprod() * 100
                        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500",
                                                  line=dict(color='#636EFA', dash='dot', width=1.5)))
                fig.add_hline(y=100, line_dash="dash", line_color="#333", line_width=0.5)
                fig.update_layout(template="plotly_dark", height=340, margin=dict(l=0, r=0, t=10, b=0),
                                  paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified",
                                  legend=dict(orientation="h", y=1.08))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Datos insuficientes para el periodo.")
        else:
            st.info("Anade activos para ver el rendimiento.")

    with col_alloc:
        st.markdown("##### Distribucion")
        if patrimonio_total > 0:
            labels_p = (['Cash'] + df_final['Nombre'].tolist()) if not df_final.empty else ['Cash']
            values_p = ([total_liquidez] + df_final['Valor Acciones'].tolist()) if not df_final.empty else [total_liquidez]
            fig_pie = px.pie(names=labels_p, values=values_p, hole=0.7,
                              color_discrete_sequence=['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A',
                                                       '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])
            fig_pie.update_traces(textposition='inside', textinfo='percent', textfont_size=9)
            fig_pie.update_layout(template="plotly_dark", height=340, showlegend=True,
                                   margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)',
                                   legend=dict(font=dict(size=9)),
                                   annotations=[dict(text=f"<b>{patrimonio_total:,.0f}EUR</b>", x=0.5, y=0.5,
                                                      font_size=14, font_color='white', showarrow=False)])
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Cartera vacia.")

    # Treemap + Bar
    if not df_final.empty:
        st.markdown("---")
        col_tree, col_bar = st.columns(2)
        with col_tree:
            st.markdown("##### Mapa de Calor")
            fig_tree = px.treemap(df_final, path=['Nombre'], values='Valor Acciones', color='Rentabilidad %',
                                  color_continuous_scale=['#EF553B', '#1e1e1e', '#00CC96'], color_continuous_midpoint=0)
            fig_tree.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=0, b=0),
                                    paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tree, use_container_width=True)

        with col_bar:
            st.markdown("##### P&L por Activo")
            df_sorted = df_final.sort_values('Ganancia', ascending=True)
            colors_b = ['#EF553B' if x < 0 else '#00CC96' for x in df_sorted['Ganancia']]
            fig_bar = go.Figure(go.Bar(
                x=df_sorted['Ganancia'], y=df_sorted['Nombre'], orientation='h',
                marker_color=colors_b, text=df_sorted['Ganancia'].apply(lambda x: f"{x:+,.0f}EUR"), textposition='outside'))
            fig_bar.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=10, t=0, b=0),
                                   paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)

    # Drawdown
    if not daily_returns.empty and len(daily_returns) > 5:
        st.markdown("---")
        st.markdown("##### Drawdown")
        cum_ret = (1 + daily_returns).cumprod()
        dd_s = (cum_ret - cum_ret.cummax()) / cum_ret.cummax() * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s, fill='tozeroy',
                                     fillcolor='rgba(239,85,59,0.15)', line=dict(color='#EF553B', width=1)))
        fig_dd.update_layout(template="plotly_dark", height=180, margin=dict(l=0, r=0, t=0, b=0),
                              paper_bgcolor='rgba(0,0,0,0)', yaxis_title="DD %", hovermode="x unified")
        st.plotly_chart(fig_dd, use_container_width=True)

    # Tabla
    if not df_final.empty:
        st.markdown("---")
        st.markdown("##### Posiciones")
        cols_show = ['Nombre', 'ticker', 'platform', 'shares', 'avg_price', 'Precio Actual',
                      'Dinero Invertido', 'Valor Acciones', 'Ganancia', 'Rentabilidad %', 'Peso %']
        cols_avail = [c for c in cols_show if c in df_final.columns]
        df_show = df_final[cols_avail].copy()
        col_names = ['Nombre', 'Ticker', 'Broker', 'Acciones', 'P.Medio', 'P.Actual',
                      'Invertido', 'Valor', 'P&L', 'Rent.%', 'Peso%'][:len(cols_avail)]
        df_show.columns = col_names
        st.dataframe(df_show.style.format({
            'Acciones': '{:.4f}', 'P.Medio': '{:.4f}', 'P.Actual': '{:.4f}',
            'Invertido': '{:,.2f}EUR', 'Valor': '{:,.2f}EUR', 'P&L': '{:+,.2f}EUR',
            'Rent.%': '{:+.2f}%', 'Peso%': '{:.1f}%'
        }), use_container_width=True, height=min(400, 50 + len(df_show) * 35))

        csv_data = df_show.to_csv(index=False).encode('utf-8')
        st.download_button("Exportar CSV", csv_data, "cartera.csv", "text/csv")

# ======================================================================
# MERCADO & NOTICIAS
# ======================================================================
elif pagina == "Mercado & Noticias":
    st.markdown("## Mercado & Noticias")

    # --- Indices en tiempo real ---
    st.markdown("##### Indices Principales")
    indices_data = get_market_indices()
    if indices_data:
        idx_cols = st.columns(len(indices_data))
        for i, idx in enumerate(indices_data):
            with idx_cols[i]:
                color = "#00CC96" if idx['change'] >= 0 else "#EF553B"
                arrow = "^" if idx['change'] >= 0 else "v"
                st.markdown(f"""
                <div class='market-ticker'>
                    <span style='font-size:0.7rem; color:var(--text-secondary);'>{idx['name']}</span><br>
                    <span style='font-size:1.1rem; font-weight:700; color:white;'>{idx['price']:,.0f}</span>
                    <span style='font-size:0.85rem; color:{color}; font-weight:600;'> {arrow} {idx['change']:+.2f}%</span>
                </div>""", unsafe_allow_html=True)
    st.markdown("---")

    # --- Noticias + Recomendaciones ---
    col_news, col_reco = st.columns([2.2, 1])

    with col_news:
        st.markdown("##### Noticias del Mercado")
        time_filter = st.selectbox("Periodo", ["Hoy", "Esta semana", "Este mes"], label_visibility="collapsed")
        tf_map = {"Hoy": "d", "Esta semana": "w", "Este mes": "m"}
        news = get_market_news(my_tickers, tf_map.get(time_filter, 'd'))

        if news:
            # Top story
            top = news[0]
            img_html = f"<img src='{top['image']}' onerror=\"this.style.display='none'\" />" if top.get('image') else ""
            st.markdown(f"""
            <div class='news-card-full' style='border-left: 3px solid var(--accent);'>
                {img_html}
                <span class='news-source-badge'>{top['source']}</span>
                <div class='news-title'><a href="{top['url']}" target="_blank">{top['title']}</a></div>
                <p style='color:var(--text-secondary); font-size:0.8rem; margin-top:6px;'>{top.get('body','')}</p>
                <div class='news-meta'>{top['date'][:16] if top['date'] else ''}</div>
            </div>""", unsafe_allow_html=True)

            # Grid de noticias
            rest = news[1:]
            for i in range(0, len(rest), 2):
                c1, c2 = st.columns(2)
                for j, col in enumerate([c1, c2]):
                    idx2 = i + j
                    if idx2 < len(rest):
                        n = rest[idx2]
                        img_h = f"<img src='{n['image']}' onerror=\"this.style.display='none'\" style='height:100px;'/>" if n.get('image') else ""
                        with col:
                            st.markdown(f"""
                            <div class='news-card-full'>
                                {img_h}
                                <span class='news-source-badge'>{n['source']}</span>
                                <div class='news-title'><a href="{n['url']}" target="_blank">{n['title']}</a></div>
                                <div class='news-meta'>{n['date'][:16] if n['date'] else ''}</div>
                            </div>""", unsafe_allow_html=True)
        else:
            st.info("No se encontraron noticias. Pulsa Actualizar.")

    with col_reco:
        st.markdown("##### Activos Populares")
        st.caption("Activos relevantes para diversificar")
        trending = get_trending_assets()

        # Filtrar los que el usuario ya tiene
        owned = set(my_tickers)
        for asset in trending:
            in_portfolio = asset['ticker'] in owned
            badge = " En cartera" if in_portfolio else ""
            border_color = "var(--accent)" if not in_portfolio else "#333"
            st.markdown(f"""
            <div class='reco-card' style='border-left-color:{border_color};'>
                <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <span class='reco-ticker'>{asset['ticker']}</span>
                    <span style='font-size:0.85rem; color:white; font-weight:600;'>{asset['price']:,.2f} {asset['currency']}</span>
                </div>
                <div class='reco-name'>{asset['name']} {badge}</div>
                <div class='reco-reason'>{asset['reason']}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Esto no es asesoramiento financiero. Investiga antes de invertir.")

# ======================================================================
# INVERSIONES
# ======================================================================
elif pagina == "Inversiones":
    st.markdown("## Gestion de Activos")
    t1, t2, t3 = st.tabs(["Anadir", "Comprar mas", "Editar / Eliminar"])

    with t1:
        c1, c2 = st.columns([1, 1])
        with c1:
            q = st.text_input("Buscar activo:", placeholder="Ej: AAPL, Amundi, MSFT, VOO...")
            if st.button("Buscar", type="primary", key="search_add") and q:
                try:
                    res = search(sanitize_input(q))
                    st.session_state['s'] = res.get('quotes', []) if 'quotes' in res else []
                    if not st.session_state['s']:
                        st.warning("Sin resultados.")
                except Exception:
                    st.error("Error en busqueda.")
                    st.session_state['s'] = []

            if 's' in st.session_state and st.session_state['s']:
                opts = {f"{x['symbol']} -- {x.get('shortname', x.get('longname', 'N/A'))} ({x.get('exchDisp', '')})": x
                        for x in st.session_state['s']}
                sel = st.selectbox("Resultado:", list(opts.keys()))
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
                        c_p1, c_p2 = st.columns(2)
                        c_p1.metric("Precio", f"{p:.2f} {mon}")
                        c_p2.metric("Ticker", tk)
                        if mon and mon != 'EUR':
                            st.info(f"Cotiza en **{mon}**. Introduce importes en EUR.")

                        with st.form("new_asset"):
                            inv_amount = st.number_input("Dinero invertido (EUR)", 0.0, step=10.0)
                            val_actual = st.number_input("Valor actual (EUR)", 0.0, step=10.0)
                            pl = st.selectbox("Broker", ["MyInvestor", "XTB", "Trade Republic", "Degiro",
                                                          "Interactive Brokers", "eToro", "Revolut", "Otro"])
                            if st.form_submit_button("Guardar", use_container_width=True, type="primary") and val_actual > 0:
                                sh = val_actual / p
                                av = inv_amount / sh if sh > 0 else 0
                                name = st.session_state['sel_add'].get('longname',
                                        st.session_state['sel_add'].get('shortname', tk))
                                add_asset_db(tk, name, sh, av, pl)
                                st.success(f"Anadido: {name}")
                                time.sleep(0.8)
                                st.rerun()
                except Exception:
                    st.error("Error al obtener precio.")

    with t2:
        if df_final.empty:
            st.info("Anade activos primero.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                nom = st.selectbox("Activo:", df_final['Nombre'].unique(), key="buy_sel")
                row = df_final[df_final['Nombre'] == nom].iloc[0]
                st.markdown(f"""
                <div class='stat-card'>
                    <p>Posicion actual</p>
                    <h2 style='color:var(--accent) !important;'>{row['Valor Acciones']:,.2f} EUR</h2>
                    <p>{row['shares']:.4f} accs - P.medio: {row['avg_price']:.4f}</p>
                </div>""", unsafe_allow_html=True)
            with c2:
                m = st.number_input("Importe a comprar (EUR)", 0.0, step=10.0)
                if m > 0:
                    precio = row['Precio Actual']
                    if precio <= 0:
                        st.error("Precio no disponible.")
                    else:
                        sh_op = m / precio
                        st.caption(f"Approx {sh_op:.4f} acciones a {precio:.4f}")
                        if m > total_liquidez:
                            st.error(f"Liquidez insuficiente ({total_liquidez:,.2f}EUR disponible)")
                        elif st.button("Confirmar Compra", type="primary", use_container_width=True):
                            navg = ((row['shares'] * row['avg_price']) + m) / (row['shares'] + sh_op)
                            update_asset_db(int(row['id']), row['shares'] + sh_op, navg)
                            update_liquidity_balance(int(cash_id), total_liquidez - m)
                            log_transaction("COMPRA", row['ticker'], nom, sh_op, precio, row.get('platform', ''))
                            st.toast(f"Compradas {sh_op:.4f} accs de {nom}")
                            time.sleep(0.5)
                            st.rerun()

    with t3:
        if not df_final.empty:
            e = st.selectbox("Activo:", df_final['Nombre'], key='edit_sel')
            er = df_final[df_final['Nombre'] == e].iloc[0]
            with st.form("edit_form"):
                st.markdown(f"**{e}** - `{er['ticker']}`")
                ce1, ce2 = st.columns(2)
                with ce1:
                    nsh = st.number_input("Acciones", value=float(er['shares']), min_value=0.0, format="%.4f")
                with ce2:
                    nav = st.number_input("P. Medio", value=float(er['avg_price']), min_value=0.0, format="%.4f")
                if st.form_submit_button("Guardar", use_container_width=True):
                    update_asset_db(int(er['id']), nsh, nav)
                    st.toast("Actualizado")
                    time.sleep(0.5)
                    st.rerun()
            st.markdown("")
            if st.button(f"Eliminar {e}", use_container_width=True):
                delete_asset_db(int(er['id']))
                st.toast(f"Eliminado: {e}")
                time.sleep(0.5)
                st.rerun()
        else:
            st.info("No hay activos.")

# ======================================================================
# LIQUIDEZ
# ======================================================================
elif pagina == "Liquidez":
    st.markdown("## Liquidez")

    pct_liq = (total_liquidez / patrimonio_total * 100) if patrimonio_total > 0 else 100
    color_liq = "#00CC96" if pct_liq >= 10 else ("#FFA15A" if pct_liq >= 5 else "#EF553B")
    st.markdown(f"""
    <div style="text-align:center; padding: 30px; background: var(--bg-card); border-radius: 14px;
                 border: 1px solid var(--border); margin-bottom: 20px;">
        <span style='color:var(--text-secondary) !important; font-size:0.85rem;'>SALDO DISPONIBLE</span>
        <h1 style="font-size: 3.5rem; color:{color_liq} !important; margin: 8px 0;">{total_liquidez:,.2f} EUR</h1>
        <span style='color:var(--text-secondary) !important;'>{pct_liq:.1f}% del patrimonio</span>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("**Ingresar**")
            a = st.number_input("Importe (EUR)", 0.0, step=50.0, key="liq_in")
            nota_in = st.text_input("Concepto", key="nota_in", placeholder="Ej: Nomina...")
            if st.button("Confirmar Ingreso", type="primary", use_container_width=True) and a > 0:
                update_liquidity_balance(int(cash_id), total_liquidez + a)
                log_transaction("INGRESO", "CASH", nota_in or "Ingreso", 1, a)
                st.toast(f"+{a:,.2f}EUR")
                time.sleep(0.5)
                st.rerun()
    with c2:
        with st.container(border=True):
            st.markdown("**Retirar**")
            b = st.number_input("Importe (EUR)", 0.0, step=50.0, key="liq_out")
            nota_out = st.text_input("Concepto", key="nota_out", placeholder="Ej: Gastos...")
            if st.button("Confirmar Retirada", use_container_width=True) and b > 0:
                if b > total_liquidez:
                    st.error(f"Saldo insuficiente ({total_liquidez:,.2f}EUR)")
                else:
                    update_liquidity_balance(int(cash_id), total_liquidez - b)
                    log_transaction("RETIRO", "CASH", nota_out or "Retiro", 1, b)
                    st.toast(f"-{b:,.2f}EUR")
                    time.sleep(0.5)
                    st.rerun()

    if patrimonio_total > 0:
        st.markdown("---")
        c3, c4, c5 = st.columns(3)
        colchon = total_liquidez / patrimonio_total * 100
        c3.metric("Colchon", f"{colchon:.1f}%", "Optimo" if colchon >= 15 else ("OK" if colchon >= 5 else "Bajo"))
        c4.metric("Inversion", f"{(total_inversiones / patrimonio_total * 100):.1f}%")
        c5.metric("Total", f"{patrimonio_total:,.2f} EUR")

# ======================================================================
# REBALANCEO -- SOLO COMPRANDO
# ======================================================================
elif pagina == "Rebalanceo":
    st.markdown("## Rebalanceo Inteligente")
    st.caption("El rebalanceo se realiza **solo comprando mas** -- nunca vendiendo. Se calcula cuanto aportar a cada activo para alcanzar los pesos objetivo.")

    if df_final.empty:
        st.warning("Anade activos primero.")
    else:
        tab_manual, tab_strat, tab_suggest = st.tabs(["Pesos Personalizados", "Estrategias", "Sugerencias"])

        # -- MANUAL --
        with tab_manual:
            col_in, col_out = st.columns([1, 1.5])
            ws = {}
            tot_w = 0
            with col_in:
                st.markdown("**Establece pesos objetivo (%)**")
                for idx_r, r_row in df_final.iterrows():
                    c_n, c_w = st.columns([2, 1])
                    with c_n:
                        st.markdown(f"**{r_row['Nombre']}**")
                        st.caption(f"Actual: {r_row['Peso %']:.1f}% - {r_row['Valor Acciones']:,.0f}EUR")
                    with c_w:
                        w = st.number_input("%", 0, 100, int(round(r_row['Peso %'])), key=f"reb_{idx_r}",
                                             label_visibility="collapsed")
                        ws[r_row['Nombre']] = w
                        tot_w += w

                aporte_extra = st.number_input("Aporte adicional (EUR)", 0.0, step=50.0,
                                                help="Dinero nuevo que vas a invertir para el rebalanceo")
                color_t = "var(--accent)" if tot_w == 100 else "var(--red)"
                st.markdown(f"**Total: <span style='color:{color_t}'>{tot_w}%</span>**", unsafe_allow_html=True)

            with col_out:
                if tot_w == 100:
                    if st.button("Calcular Plan de Compras", type="primary", use_container_width=True):
                        capital_target = total_inversiones + aporte_extra
                        reb_data = []
                        total_comprar = 0
                        for _, r_row in df_final.iterrows():
                            target_val = capital_target * ws[r_row['Nombre']] / 100
                            diff = target_val - r_row['Valor Acciones']
                            comprar = max(0, diff)  # SOLO COMPRAR
                            total_comprar += comprar
                            reb_data.append({
                                'Activo': r_row['Nombre'],
                                'Ticker': r_row['ticker'],
                                'Actual EUR': r_row['Valor Acciones'],
                                'Actual %': r_row['Peso %'],
                                'Objetivo %': ws[r_row['Nombre']],
                                'Comprar EUR': comprar,
                            })

                        df_reb = pd.DataFrame(reb_data)
                        df_reb['Nuevo Valor EUR'] = df_reb['Actual EUR'] + df_reb['Comprar EUR']
                        new_total = df_reb['Nuevo Valor EUR'].sum()
                        df_reb['Nuevo %'] = df_reb['Nuevo Valor EUR'] / new_total * 100 if new_total > 0 else 0

                        st.dataframe(df_reb[['Activo', 'Actual EUR', 'Actual %', 'Objetivo %', 'Comprar EUR', 'Nuevo %']].style.format({
                            'Actual EUR': '{:,.2f}', 'Actual %': '{:.1f}', 'Objetivo %': '{:.0f}',
                            'Comprar EUR': '{:,.2f}', 'Nuevo %': '{:.1f}'
                        }), use_container_width=True)

                        # Grafico
                        fig_reb = go.Figure()
                        fig_reb.add_trace(go.Bar(name='Actual', x=df_reb['Activo'], y=df_reb['Actual %'],
                                                  marker_color='#636EFA'))
                        fig_reb.add_trace(go.Bar(name='Tras Rebalanceo', x=df_reb['Activo'], y=df_reb['Nuevo %'],
                                                  marker_color='#00CC96'))
                        fig_reb.update_layout(template="plotly_dark", barmode='group', height=300,
                                               paper_bgcolor='rgba(0,0,0,0)', yaxis_title="%")
                        st.plotly_chart(fig_reb, use_container_width=True)

                        # Resumen
                        disponible = total_liquidez + aporte_extra
                        if disponible >= total_comprar:
                            msg_suf = "Capital suficiente"
                        else:
                            msg_suf = f"Faltan {total_comprar - disponible:,.2f}EUR"
                        st.markdown(f"""
                        ---
                        **Plan de Compras:**
                        - Total a invertir: **{total_comprar:,.2f}EUR**
                        - Liquidez disponible: **{total_liquidez:,.2f}EUR** + Aporte: **{aporte_extra:,.2f}EUR** = **{disponible:,.2f}EUR**
                        - {msg_suf}
                        """)
                else:
                    st.warning(f"Los pesos deben sumar 100% (actual: {tot_w}%)")

        # -- ESTRATEGIAS --
        with tab_strat:
            estrategia = st.selectbox("Estrategia:", [
                "Equiponderado",
                "Momentum (mas a ganadores)",
                "Minima Volatilidad",
            ])
            aporte_strat = st.number_input("Aporte adicional (EUR)", 0.0, step=50.0, key="aporte_strat")

            if st.button("Calcular", type="primary", key="calc_strat"):
                n_assets = len(df_final)
                if estrategia == "Equiponderado":
                    tw = {r['Nombre']: 100 / n_assets for _, r in df_final.iterrows()}
                elif estrategia == "Momentum (mas a ganadores)":
                    rents = df_final['Rentabilidad %'].clip(lower=0)
                    total_r = rents.sum()
                    if total_r > 0:
                        tw = {r['Nombre']: (max(0, r['Rentabilidad %']) / total_r * 100)
                              for _, r in df_final.iterrows()}
                    else:
                        tw = {r['Nombre']: 100 / n_assets for _, r in df_final.iterrows()}
                else:  # Min Vol
                    if not history_data.empty:
                        vols = {}
                        for _, r in df_final.iterrows():
                            if r['ticker'] in history_data.columns:
                                v = history_data[r['ticker']].pct_change().std()
                                vols[r['Nombre']] = 1 / max(v, 0.001)
                            else:
                                vols[r['Nombre']] = 1
                        tot_v = sum(vols.values())
                        tw = {k: v / tot_v * 100 for k, v in vols.items()}
                    else:
                        tw = {r['Nombre']: 100 / n_assets for _, r in df_final.iterrows()}

                capital_target = total_inversiones + aporte_strat
                strat_data = []
                total_buy = 0
                for _, r in df_final.iterrows():
                    target_val = capital_target * tw[r['Nombre']] / 100
                    comprar = max(0, target_val - r['Valor Acciones'])
                    total_buy += comprar
                    strat_data.append({
                        'Activo': r['Nombre'], 'Actual %': r['Peso %'],
                        'Objetivo %': tw[r['Nombre']], 'Comprar EUR': comprar
                    })

                df_strat = pd.DataFrame(strat_data)
                st.dataframe(df_strat.style.format({
                    'Actual %': '{:.1f}', 'Objetivo %': '{:.1f}', 'Comprar EUR': '{:,.2f}'
                }), use_container_width=True)

                disponible_s = total_liquidez + aporte_strat
                ok_msg = "OK" if disponible_s >= total_buy else "Falta liquidez"
                st.info(f"Total a comprar: **{total_buy:,.2f}EUR** | Disponible: **{disponible_s:,.2f}EUR** | {ok_msg}")

        # -- SUGERENCIAS --
        with tab_suggest:
            st.markdown("##### Que falta en tu cartera?")
            st.caption("Analisis automatico basado en tu composicion actual.")

            if not df_final.empty:
                # Obtener sectores actuales
                sectors_owned = {}
                for _, r in df_final.iterrows():
                    info = get_ticker_info(r['ticker'])
                    sec = info.get('sector', 'Desconocido') if info else 'Desconocido'
                    sectors_owned[sec] = sectors_owned.get(sec, 0) + r['Peso %']

                # Sugerencias por gap
                all_sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
                               'Industrials', 'Energy', 'Real Estate', 'Utilities', 'Communication Services',
                               'Consumer Defensive', 'Basic Materials']
                sector_etfs = {
                    'Technology': ('XLK', 'Technology Select Sector SPDR'),
                    'Healthcare': ('XLV', 'Health Care Select Sector SPDR'),
                    'Financial Services': ('XLF', 'Financial Select Sector SPDR'),
                    'Consumer Cyclical': ('XLY', 'Consumer Discretionary SPDR'),
                    'Industrials': ('XLI', 'Industrial Select Sector SPDR'),
                    'Energy': ('XLE', 'Energy Select Sector SPDR'),
                    'Real Estate': ('XLRE', 'Real Estate Select Sector SPDR'),
                    'Utilities': ('XLU', 'Utilities Select Sector SPDR'),
                    'Communication Services': ('XLC', 'Communication Services SPDR'),
                    'Consumer Defensive': ('XLP', 'Consumer Staples SPDR'),
                    'Basic Materials': ('XLB', 'Materials Select Sector SPDR'),
                }

                missing_sectors = [s for s in all_sectors if s not in sectors_owned or sectors_owned.get(s, 0) < 3]

                # Diversification check
                weights = (df_final['Peso %'] / 100).tolist()
                div = diversification_score(weights)

                st.metric("Diversificacion actual", f"{div:.0f}/100")

                if div < 60:
                    st.warning("Tu cartera esta concentrada. Considera diversificar mas.")
                elif div < 80:
                    st.info("Diversificacion aceptable. Hay margen de mejora.")
                else:
                    st.success("Buena diversificacion.")

                if missing_sectors:
                    st.markdown("**Sectores sin cobertura o infrarrepresentados:**")
                    for sec in missing_sectors[:6]:
                        etf = sector_etfs.get(sec)
                        if etf:
                            ticker_s, name_s = etf
                            in_cart = ticker_s in my_tickers
                            badge_s = "Ya tienes" if in_cart else ""
                            st.markdown(f"""
                            <div class='reco-card'>
                                <div style='display:flex; justify-content:space-between;'>
                                    <span class='reco-ticker'>{ticker_s}</span>
                                    <span style='font-size:0.75rem; color:var(--text-secondary);'>{sec}</span>
                                </div>
                                <div class='reco-name'>{name_s} {badge_s}</div>
                                <div class='reco-reason'>ETF sectorial para cubrir la exposicion a {sec}</div>
                            </div>""", unsafe_allow_html=True)

                # Renta fija / Oro
                has_bonds = any('BND' in t or 'AGG' in t or 'TLT' in t for t in my_tickers)
                has_gold = any('GLD' in t or 'IAU' in t for t in my_tickers)
                if not has_bonds:
                    st.markdown("""
                    <div class='reco-card'>
                        <span class='reco-ticker'>BND / AGG</span>
                        <div class='reco-name'>Renta Fija USA</div>
                        <div class='reco-reason'>No tienes bonos en cartera. Anadir renta fija reduce la volatilidad y protege en caidas.</div>
                    </div>""", unsafe_allow_html=True)
                if not has_gold:
                    st.markdown("""
                    <div class='reco-card'>
                        <span class='reco-ticker'>GLD / IAU</span>
                        <div class='reco-name'>Oro</div>
                        <div class='reco-reason'>El oro actua como cobertura contra inflacion y descorrelaciona tu cartera.</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("Anade activos para recibir sugerencias.")

# ======================================================================
# RADIOGRAFIA
# ======================================================================
elif pagina == "Radiografia":
    st.markdown("## Radiografia de Cartera")

    if df_final.empty:
        st.warning("Anade activos.")
    else:
        weights = (df_final['Peso %'] / 100).tolist()
        div = diversification_score(weights)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Diversificacion", f"{div:.0f}/100")
        c2.metric("Activos", len(df_final))
        c3.metric("Mayor peso", f"{df_final['Peso %'].max():.1f}%")
        c4.metric("Invertido", f"{df_final['Dinero Invertido'].sum():,.2f}EUR")

        st.markdown("---")
        tab_corr, tab_sec, tab_brok, tab_det = st.tabs(["Correlacion", "Sectores", "Brokers", "Detalle"])

        with tab_corr:
            if not history_data.empty and len(history_data.columns) > 1:
                valid_c = [c for c in history_data.columns if c in my_tickers]
                if len(valid_c) > 1:
                    corr = history_data[valid_c].pct_change().dropna().corr()
                    fig_c = px.imshow(corr, text_auto='.2f',
                                      color_continuous_scale=['#EF553B', '#1e1e1e', '#00CC96'], aspect='equal')
                    fig_c.update_layout(template="plotly_dark", height=420, paper_bgcolor='rgba(0,0,0,0)',
                                         margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_c, use_container_width=True)

                    high_c = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                              for i in range(len(corr)) for j in range(i + 1, len(corr))
                              if abs(corr.iloc[i, j]) > 0.8]
                    if high_c:
                        st.warning("Pares muy correlacionados (>0.8):")
                        for a_t, b_t, v in high_c:
                            st.write(f"- {a_t} <-> {b_t}: **{v:.2f}**")
                else:
                    st.info("Necesitas 2+ activos.")
            else:
                st.info("Sin datos historicos.")

        with tab_sec:
            with st.spinner("Cargando sectores..."):
                sectors = {}
                for _, r in df_final.iterrows():
                    info = get_ticker_info(r['ticker'])
                    sec = info.get('sector', 'Desconocido') if info else 'Desconocido'
                    sectors[sec] = sectors.get(sec, 0) + r['Valor Acciones']
            if sectors:
                df_sec = pd.DataFrame(list(sectors.items()), columns=['Sector', 'Valor']).sort_values('Valor', ascending=False)
                df_sec['%'] = df_sec['Valor'] / df_sec['Valor'].sum() * 100
                cs1, cs2 = st.columns(2)
                with cs1:
                    fig_s = px.bar(df_sec, x='Sector', y='Valor', color='Sector',
                                    text=df_sec['%'].apply(lambda x: f"{x:.1f}%"),
                                    color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_s.update_layout(template="plotly_dark", height=320, showlegend=False,
                                         paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_s, use_container_width=True)
                with cs2:
                    fig_sp = px.pie(df_sec, names='Sector', values='Valor', hole=0.5,
                                     color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_sp.update_layout(template="plotly_dark", height=320, paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_sp, use_container_width=True)

        with tab_brok:
            if 'platform' in df_final.columns:
                bk = df_final.groupby('platform').agg(Valor=('Valor Acciones', 'sum'), N=('ticker', 'count'),
                                                       PnL=('Ganancia', 'sum')).reset_index()
                bk.columns = ['Broker', 'Valor', 'Activos', 'P&L']
                cb1, cb2 = st.columns(2)
                with cb1:
                    fig_bk = px.pie(bk, names='Broker', values='Valor', hole=0.5)
                    fig_bk.update_layout(template="plotly_dark", height=320, paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_bk, use_container_width=True)
                with cb2:
                    st.dataframe(bk.style.format({'Valor': '{:,.2f}EUR', 'P&L': '{:+,.2f}EUR'}), use_container_width=True)

        with tab_det:
            sel_a = st.selectbox("Activo:", df_final['Nombre'].tolist(), key="xray_det")
            if sel_a:
                row_s = df_final[df_final['Nombre'] == sel_a].iloc[0]
                tk_s = row_s['ticker']
                info_s = get_ticker_info(tk_s)
                cd1, cd2 = st.columns(2)
                with cd1:
                    st.markdown(f"### {sel_a}")
                    if info_s:
                        for label_i, key_i in [("Sector", "sector"), ("Industria", "industry"), ("Pais", "country")]:
                            st.write(f"**{label_i}:** {info_s.get(key_i, 'N/A')}")
                        st.write(f"**P/E:** {info_s.get('pe_ratio', 0):.2f} - **Beta:** {info_s.get('beta', 0):.2f}")
                        st.write(f"**Dividendo:** {info_s.get('dividend_yield', 0) * 100:.2f}%")
                        mc = info_s.get('market_cap', 0)
                        if mc > 1e12: st.write(f"**Market Cap:** {mc/1e12:.2f}T")
                        elif mc > 1e9: st.write(f"**Market Cap:** {mc/1e9:.2f}B")
                with cd2:
                    st.metric("Valor", f"{row_s['Valor Acciones']:,.2f}EUR")
                    st.metric("P&L", f"{row_s['Ganancia']:+,.2f}EUR", f"{row_s['Rentabilidad %']:+.2f}%")
                    st.metric("Peso", f"{row_s['Peso %']:.1f}%")
                if not history_data.empty and tk_s in history_data.columns:
                    h_a = history_data[tk_s].dropna()
                    if not h_a.empty:
                        fig_i = go.Figure()
                        fig_i.add_trace(go.Scatter(x=h_a.index, y=h_a, line=dict(color='#00CC96', width=2),
                                                    fill='tozeroy', fillcolor='rgba(0,204,150,0.05)', name=tk_s))
                        fig_i.add_hline(y=row_s['avg_price'], line_dash="dash", line_color="#FFA15A",
                                         annotation_text=f"P.Medio: {row_s['avg_price']:.2f}")
                        fig_i.update_layout(template="plotly_dark", height=280, paper_bgcolor='rgba(0,0,0,0)',
                                             margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
                        st.plotly_chart(fig_i, use_container_width=True)

# ======================================================================
# SIMULADOR (MONTE CARLO)
# ======================================================================
elif pagina == "Simulador":
    st.markdown("## Simulador Monte Carlo")
    st.caption("Proyeccion estocastica del valor futuro con percentiles P10/P50/P90.")

    c1, c2, c3 = st.columns(3)
    with c1:
        ys = st.slider("Horizonte (anos)", 1, 30, 10)
    with c2:
        n_sims = st.select_slider("Simulaciones", [100, 500, 1000, 5000], 1000)
    with c3:
        aport = st.number_input("Aportacion mensual (EUR)", 0.0, step=50.0, value=0.0)

    if st.button("Simular", type="primary", use_container_width=True):
        mu, sigma = 0.07, 0.15
        if not history_data.empty:
            d = history_data.pct_change().dropna()
            if len(d) > 20:
                mu = np.clip(d.mean().mean() * 252, -0.3, 0.5)
                sigma = np.clip(d.mean(axis=1).std() * np.sqrt(252), 0.05, 0.8)

        n_steps = int(ys * 252)
        cap = max(total_inversiones, 1)

        with st.spinner("Simulando..."):
            paths = np.zeros((n_sims, n_steps + 1))
            paths[:, 0] = cap
            dt = 1 / 252
            drift = (mu - 0.5 * sigma ** 2) * dt
            diff = sigma * np.sqrt(dt)
            aport_d = aport * 12 / 252
            shocks = np.random.normal(0, 1, (n_sims, n_steps))
            for s in range(n_steps):
                paths[:, s + 1] = paths[:, s] * np.exp(drift + diff * shocks[:, s]) + aport_d

        x = np.linspace(0, ys, n_steps + 1)
        p10 = np.percentile(paths, 10, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p90 = np.percentile(paths, 90, axis=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([p90, p10[::-1]]),
                                  fill='toself', fillcolor='rgba(0,204,150,0.08)',
                                  line=dict(color='rgba(0,0,0,0)'), name='P10-P90'))
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([p75, p25[::-1]]),
                                  fill='toself', fillcolor='rgba(0,204,150,0.15)',
                                  line=dict(color='rgba(0,0,0,0)'), name='P25-P75'))
        fig.add_trace(go.Scatter(x=x, y=p50, line=dict(color='#00CC96', width=3), name='Mediana'))
        for si in np.random.choice(n_sims, min(15, n_sims), replace=False):
            fig.add_trace(go.Scatter(x=x, y=paths[si], line=dict(color='rgba(255,255,255,0.03)', width=0.5),
                                      showlegend=False))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=400,
                           xaxis_title="Anos", yaxis_title="EUR", hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

        cr1, cr2, cr3, cr4, cr5 = st.columns(5)
        cr1.metric("P10", f"{p10[-1]:,.0f}EUR")
        cr2.metric("P25", f"{p25[-1]:,.0f}EUR")
        cr3.metric("P50", f"{p50[-1]:,.0f}EUR")
        cr4.metric("P75", f"{p75[-1]:,.0f}EUR")
        cr5.metric("P90", f"{p90[-1]:,.0f}EUR")

        total_aportado = cap + aport * 12 * ys
        prob_loss = np.sum(paths[:, -1] < total_aportado) / n_sims * 100
        st.markdown(f"""
        - **Capital inicial:** {cap:,.2f}EUR | **Aportaciones:** {aport * 12 * ys:,.2f}EUR
        - **Total aportado:** {total_aportado:,.2f}EUR
        - **Ganancia mediana:** {p50[-1] - total_aportado:+,.2f}EUR
        - **Prob. perdida:** {prob_loss:.1f}%
        """)

# ======================================================================
# HISTORIAL
# ======================================================================
elif pagina == "Historial":
    st.markdown("## Historial de Operaciones")

    if st.session_state['tx_log']:
        df_tx = pd.DataFrame(st.session_state['tx_log']).sort_values('fecha', ascending=False)
        tipos = ["Todos"] + list(df_tx['tipo'].unique())
        filtro = st.selectbox("Filtrar:", tipos)
        if filtro != "Todos":
            df_tx = df_tx[df_tx['tipo'] == filtro]

        def color_tipo(val):
            return {'COMPRA': 'color:#00CC96', 'VENTA': 'color:#EF553B',
                    'INGRESO': 'color:#636EFA', 'RETIRO': 'color:#FFA15A'}.get(val, '')

        st.dataframe(df_tx.style.applymap(color_tipo, subset=['tipo']),
                      use_container_width=True, height=min(500, 50 + len(df_tx) * 35))

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Operaciones", len(df_tx))
        c2.metric("Compras", f"{df_tx[df_tx['tipo'] == 'COMPRA']['importe'].sum():,.2f}EUR")
        c3.metric("Ingresos", f"{df_tx[df_tx['tipo'] == 'INGRESO']['importe'].sum():,.2f}EUR")

        st.download_button("Exportar", df_tx.to_csv(index=False).encode('utf-8'), "historial.csv", "text/csv")
    else:
        st.info("Las operaciones se registran automaticamente al operar.")
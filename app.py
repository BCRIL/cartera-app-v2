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
import html as html_module
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

    /* Info tips - explicaciones accesibles */
    .info-tip {
        background: linear-gradient(135deg, #111822 0%, #141d28 100%);
        border: 1px solid #1e3a50; border-left: 3px solid #3b82f6;
        border-radius: 8px; padding: 12px 16px; margin: 8px 0 14px 0;
        font-size: 0.82rem; line-height: 1.55; color: #b0bec5 !important;
    }
    .info-tip b { color: #e0e0e0 !important; }
    .info-tip .tip-icon { font-size: 1rem; margin-right: 6px; }
    .chart-explain {
        background: var(--bg-card); border: 1px solid var(--border); border-radius: 8px;
        padding: 10px 14px; margin-top: 6px; font-size: 0.78rem; line-height: 1.5;
        color: var(--text-secondary) !important;
    }
    .chart-explain b { color: var(--text-primary) !important; }
    .metric-explain {
        font-size: 0.7rem; color: var(--text-secondary) !important;
        margin-top: 2px; padding: 0 4px; line-height: 1.3;
    }
    .health-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-left: 6px;
    }
    .health-good { background: rgba(0,204,150,0.15); color: #00CC96; border: 1px solid rgba(0,204,150,0.3); }
    .health-warn { background: rgba(255,161,90,0.15); color: #FFA15A; border: 1px solid rgba(255,161,90,0.3); }
    .health-bad { background: rgba(239,85,59,0.15); color: #EF553B; border: 1px solid rgba(239,85,59,0.3); }
</style>
""", unsafe_allow_html=True)

# --- SEGURIDAD: htmlescape helper ---
def esc(text):
    """Escape HTML entities to prevent XSS in user-rendered content."""
    return html_module.escape(str(text)) if text else ""

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip()))

def validate_password(pw):
    if len(pw) < 8:
        return False, "Minimo 8 caracteres."
    if not re.search(r'[A-Za-z]', pw):
        return False, "Debe contener al menos una letra."
    if not re.search(r'[0-9]', pw):
        return False, "Debe contener al menos un numero."
    return True, ""

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
for k, v in {'user': None, 'tx_log': [], 'login_attempts': 0, 'last_attempt': 0}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==============================================================================
# LOGIN - MULTIUSUARIO
# ==============================================================================
# --- GOOGLE OAUTH: Capturar callback ---
query_params = st.query_params
if "code" in query_params and not st.session_state['user']:
    try:
        auth_code = str(query_params["code"])[:256]  # limitar longitud
        session = supabase.auth.exchange_code_for_session({"auth_code": auth_code})
        st.session_state['user'] = session.user
        st.session_state['login_attempts'] = 0
        st.query_params.clear()
        st.rerun()
    except Exception:
        st.query_params.clear()
        st.error("No se pudo completar el inicio de sesion con Google. Intentalo de nuevo.")

# TOKEN-based session (Supabase access_token in URL fragment)
if "access_token" in query_params and not st.session_state['user']:
    try:
        access_token = str(query_params["access_token"])[:2048]
        refresh_token = str(query_params.get("refresh_token", ""))[:2048]
        if access_token and refresh_token:
            session = supabase.auth.set_session(access_token, refresh_token)
            st.session_state['user'] = session.user
            st.session_state['login_attempts'] = 0
        st.query_params.clear()
        st.rerun()
    except Exception:
        st.query_params.clear()

if not st.session_state['user']:
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
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
            # --- Google OAuth ---
            try:
                redirect_url = st.secrets.get("REDIRECT_URL", "https://carterapro.streamlit.app")
                oauth_response = supabase.auth.sign_in_with_oauth({
                    "provider": "google",
                    "options": {
                        "redirect_to": redirect_url,
                        "scopes": "openid email profile",
                    }
                })
                google_url = oauth_response.url if oauth_response else None
            except Exception:
                google_url = None

            if google_url:
                st.link_button("🔑 Iniciar sesion con Google", google_url, use_container_width=True, type="primary")
            else:
                st.warning("Google Sign-In no disponible. Usa email y contrasena.")

            st.markdown("<div style='text-align:center; padding:10px 0; color:var(--text-secondary);'>──── o ────</div>", unsafe_allow_html=True)

            with st.form("login_form"):
                em = st.text_input("Email", autocomplete="email")
                pa = st.text_input("Contrasena", type="password", autocomplete="current-password")
                if st.form_submit_button("Entrar", use_container_width=True, type="primary"):
                    # Rate limiting
                    now = time.time()
                    if st.session_state['login_attempts'] >= 5 and (now - st.session_state['last_attempt']) < 60:
                        st.error("Demasiados intentos. Espera 1 minuto.")
                    elif not validate_email(em):
                        st.error("Email no valido.")
                    elif not pa:
                        st.error("Introduce tu contrasena.")
                    else:
                        try:
                            res = supabase.auth.sign_in_with_password({"email": em.strip(), "password": pa})
                            st.session_state['user'] = res.user
                            st.session_state['login_attempts'] = 0
                            st.rerun()
                        except Exception:
                            st.session_state['login_attempts'] += 1
                            st.session_state['last_attempt'] = now
                            remaining = 5 - st.session_state['login_attempts']
                            if remaining > 0:
                                st.error(f"Credenciales incorrectas. {remaining} intentos restantes.")
                            else:
                                st.error("Demasiados intentos fallidos. Espera 1 minuto.")
        with tab_signup:
            with st.form("signup_form"):
                em2 = st.text_input("Email", autocomplete="email")
                pa2 = st.text_input("Contrasena", type="password", autocomplete="new-password")
                pa3 = st.text_input("Confirmar contrasena", type="password")
                if st.form_submit_button("Registrarse", use_container_width=True, type="primary"):
                    if not validate_email(em2):
                        st.error("Email no valido.")
                    elif pa2 != pa3:
                        st.error("Las contrasenas no coinciden.")
                    else:
                        pw_ok, pw_msg = validate_password(pa2)
                        if not pw_ok:
                            st.error(pw_msg)
                        else:
                            try:
                                supabase.auth.sign_up({"email": em2.strip(), "password": pa2})
                                st.success("Cuenta creada. Revisa tu email para confirmar.")
                            except Exception as e:
                                st.error(f"Error al crear cuenta. Intentalo de nuevo.")
    st.stop()

user = st.session_state['user']

# ==============================================================================
# MOTOR DE DATOS
# ==============================================================================

def sanitize_input(text):
    """Sanitize user input: allow only safe characters for ticker/asset search."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = re.sub(r'[^\w\s\-\.]', '', text).strip().upper()
    return cleaned[:50]  # max length limit

def sanitize_ticker(ticker):
    """Strict ticker sanitization: only alphanumeric, dots, hyphens."""
    if not ticker or not isinstance(ticker, str):
        return ""
    return re.sub(r'[^A-Za-z0-9\.\-\^]', '', ticker.strip().upper())[:20]

def sanitize_number(val, min_val=0.0, max_val=1e12):
    """Validate numeric inputs are within bounds."""
    try:
        n = float(val)
        return max(min_val, min(max_val, n))
    except (ValueError, TypeError):
        return min_val

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
    t = sanitize_ticker(t)
    n = esc(str(n)[:100])
    s = sanitize_number(s, 0.0)
    p = sanitize_number(p, 0.0)
    pl = esc(str(pl)[:50])
    if not t or s <= 0:
        st.error("Datos de activo no validos.")
        return
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
    shares = sanitize_number(shares, 0.0)
    avg_price = sanitize_number(avg_price, 0.0)
    try:
        supabase.table('assets').update({"shares": shares, "avg_price": avg_price}).eq('id', asset_id).eq('user_id', user.id).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error al actualizar: {e}")

def delete_asset_db(id_del):
    try:
        supabase.table('assets').delete().eq('id', id_del).eq('user_id', user.id).execute()
        clear_cache()
    except Exception as e:
        st.error(f"Error al eliminar: {e}")

def update_liquidity_balance(liq_id, new_amount):
    new_amount = sanitize_number(new_amount, 0.0)
    try:
        supabase.table('liquidity').update({"amount": new_amount}).eq('id', liq_id).eq('user_id', user.id).execute()
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

    # XSS-safe display
    safe_name = esc(nombre_user)
    safe_avatar = esc(avatar) if avatar and avatar.startswith('https://') else ''

    if safe_avatar:
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; padding:12px; background:var(--bg-card);
                     border-radius:10px; border:1px solid var(--border); margin-bottom:16px;'>
            <img src='{safe_avatar}' style='width:36px; height:36px; border-radius:50%; border:2px solid var(--accent);'
                 referrerpolicy='no-referrer'>
            <div style='line-height:1.2; overflow:hidden;'>
                <b style='color:white; font-size:0.9rem;'>{safe_name}</b><br>
                <span style='font-size:0.7em; color:var(--accent);'>Online</span>
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='padding:12px; background:var(--bg-card); border-radius:10px;
                     border:1px solid var(--border); margin-bottom:16px;'>
            <b style='color:white;'>{safe_name}</b><br>
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
    # Dynamic portfolio-aware intro
    if df_final.empty:
        dash_intro = """<div class='info-tip'>
            <span class='tip-icon'>📖</span> <b>Tu panel de control financiero.</b>
            Aun no tienes inversiones. Empieza anadiendo activos en la seccion <b>Inversiones</b>
            y deposita liquidez en <b>Liquidez</b> para comenzar a construir tu cartera.
        </div>"""
    else:
        n_activos = len(df_final)
        best_asset = df_final.loc[df_final['Rentabilidad %'].idxmax()]
        worst_asset = df_final.loc[df_final['Rentabilidad %'].idxmin()]
        pct_liq_dash = (total_liquidez / patrimonio_total * 100) if patrimonio_total > 0 else 0
        dash_intro = f"""<div class='info-tip'>
            <span class='tip-icon'>📖</span> <b>Resumen de tu cartera en vivo.</b>
            Tienes <b>{n_activos} inversiones</b> con un patrimonio total de <b>{patrimonio_total:,.0f}EUR</b>.
            Tu mejor activo es <b>{esc(best_asset['Nombre'])}</b> ({best_asset['Rentabilidad %']:+.1f}%)
            y el que peor va es <b>{esc(worst_asset['Nombre'])}</b> ({worst_asset['Rentabilidad %']:+.1f}%).
            {'<b style="color:#FFA15A">Tu liquidez es solo el ' + f'{pct_liq_dash:.0f}%' + ' del patrimonio, considera mantener al menos un 10%.</b>' if 0 < pct_liq_dash < 10 else ''}
        </div>"""
    st.markdown(dash_intro, unsafe_allow_html=True)

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
    k1.metric("Patrimonio", f"{patrimonio_total:,.2f} EUR", help="Valor total de tus inversiones + dinero en efectivo. Es todo lo que tienes.")
    if total_inversiones > 0:
        pnl_total = df_final['Ganancia'].sum()
        dinero_inv = df_final['Dinero Invertido'].sum()
        rent_pct = (pnl_total / dinero_inv * 100) if dinero_inv > 0 else 0
        k2.metric("P&L", f"{pnl_total:+,.2f} EUR", f"{rent_pct:+.2f}%",
                   delta_color="normal" if pnl_total >= 0 else "inverse",
                   help="Profit & Loss (Ganancia/Perdida). Cuanto has ganado o perdido en total desde que invertiste.")
    else:
        k2.metric("P&L", "0.00 EUR", help="Profit & Loss. Aun no tienes inversiones.")
    k3.metric("Liquidez", f"{total_liquidez:,.2f} EUR", help="Dinero disponible en efectivo que no esta invertido. Tu colchon de seguridad.")
    k4.metric("Activos", f"{len(df_final)}", help="Numero de inversiones diferentes que tienes en cartera.")

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

    k5.metric("Sharpe", f"{sharpe_ratio:.2f}", help="Ratio de Sharpe: mide si el riesgo que asumes esta bien recompensado. Mayor de 1 = bueno, mayor de 2 = excelente.")

    # Interpretacion automatica del P&L
    if total_inversiones > 0:
        if rent_pct > 15:
            health = "health-good"
            msg_pnl = "Excelente rendimiento"
        elif rent_pct > 0:
            health = "health-good"
            msg_pnl = "Rendimiento positivo"
        elif rent_pct > -5:
            health = "health-warn"
            msg_pnl = "Pequena perdida, normal a corto plazo"
        else:
            health = "health-bad"
            msg_pnl = "En perdidas significativas"
        st.markdown(f"<span class='health-badge {health}'>{msg_pnl} ({rent_pct:+.1f}%)</span>", unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Volatilidad", f"{vol_anual:.1f}%", help="Cuanto sube y baja tu cartera. Menos del 15% = tranquila, mas del 25% = arriesgada.")
    m2.metric("Max DD", f"{max_drawdown:.1f}%", help="Maxima caida desde un pico. Es la peor racha que ha tenido tu cartera. Cuanto menor, mejor.")
    m3.metric("Beta", f"{beta_portfolio:.2f}", help="Sensibilidad al mercado. Beta=1 se mueve igual que el mercado, <1 es mas tranquila, >1 es mas agresiva.")
    m4.metric("Sortino", f"{sortino:.2f}", help="Como Sharpe pero solo mide el riesgo de perdida. Mayor valor = mejor relacion ganancia/riesgo de caida.")

    # Explicacion rapida de metricas - portfolio-aware
    with st.expander("ℹ️ ¿Que significan estos numeros?", expanded=False):
        # Dynamic thresholds based on actual data
        vol_status = "✅ Tranquila" if vol_anual < 15 else ("⚠️ Moderada" if vol_anual < 25 else "🚨 Alta")
        dd_status = "✅ Contenida" if max_drawdown > -10 else ("⚠️ Normal" if max_drawdown > -20 else "🚨 Profunda")
        beta_status = "✅ Defensiva" if beta_portfolio < 0.8 else ("✅ Neutral" if beta_portfolio < 1.2 else "⚠️ Agresiva")
        sharpe_status = "✅ Bueno" if sharpe_ratio > 1 else ("⚠️ Mejorable" if sharpe_ratio > 0.5 else "🚨 Bajo")
        sortino_status = "✅ Bueno" if sortino > 1.5 else ("⚠️ Mejorable" if sortino > 0.7 else "🚨 Bajo")
        st.markdown(f"""
        | Metrica | Tu valor | Estado | Que mide |
        |---------|----------|--------|----------|
        | **Volatilidad** | {vol_anual:.1f}% | {vol_status} | Cuanto fluctua tu cartera |
        | **Max Drawdown** | {max_drawdown:.1f}% | {dd_status} | La peor caida historica |
        | **Beta** | {beta_portfolio:.2f} | {beta_status} | Sensibilidad al mercado |
        | **Sharpe** | {sharpe_ratio:.2f} | {sharpe_status} | Rentabilidad ajustada al riesgo |
        | **Sortino** | {sortino:.2f} | {sortino_status} | Rentabilidad ajustada a caidas |
        """)

    st.markdown("---")

    # Charts
    col_chart, col_alloc = st.columns([2.2, 1])
    with col_chart:
        st.markdown("##### Rendimiento vs S&P 500")
        st.markdown("""<div class='chart-explain'>
            <b>¿Como leer este grafico?</b> La <b style='color:#00CC96'>linea verde</b> es tu cartera y la
            <b style='color:#636EFA'>linea azul punteada</b> es el S&P 500 (las 500 mayores empresas de EE.UU.).
            Si tu linea verde esta por encima de la azul, lo estas haciendo mejor que el mercado.
            La linea horizontal en 100 es tu punto de partida.
        </div>""", unsafe_allow_html=True)
        if not history_data.empty:
            dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
            hist_filt = history_data[history_data.index >= dt_start].copy()
            if not hist_filt.empty and len(hist_filt) > 2:
                port_ret = hist_filt.pct_change().mean(axis=1).fillna(0)
                port_cum = (1 + port_ret).cumprod() * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Tu Cartera",
                                         line=dict(color='#00CC96', width=2.5, shape='spline'),
                                         fill='tozeroy', fillcolor='rgba(0,204,150,0.06)',
                                         hovertemplate='<b>Tu cartera</b><br>Fecha: %{x|%d %b %Y}<br>Valor: %{y:.1f}<extra></extra>'))
                if not benchmark_data.empty:
                    bench_filt = benchmark_data[benchmark_data.index >= dt_start].copy()
                    if not bench_filt.empty:
                        bench_cum = (1 + bench_filt.pct_change().fillna(0)).cumprod() * 100
                        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500",
                                                  line=dict(color='#636EFA', dash='dot', width=1.8, shape='spline'),
                                                  hovertemplate='<b>S&P 500</b><br>Fecha: %{x|%d %b %Y}<br>Valor: %{y:.1f}<extra></extra>'))
                fig.add_hline(y=100, line_dash="dash", line_color="#555", line_width=1,
                              annotation_text="Punto de partida", annotation_font_color="#888", annotation_font_size=10)
                fig.update_layout(template="plotly_dark", height=360, margin=dict(l=0, r=0, t=10, b=0),
                                  paper_bgcolor='rgba(0,0,0,0)', hovermode="x unified",
                                  legend=dict(orientation="h", y=1.08),
                                  yaxis_title="Rendimiento (base 100)",
                                  xaxis_title="Fecha")
                fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
                fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
                st.plotly_chart(fig, use_container_width=True)
                # Interpretacion automatica
                final_port = port_cum.iloc[-1]
                if final_port > 100:
                    st.markdown(f"""<div class='chart-explain'>📈 Tu cartera ha subido un <b style='color:#00CC96'>{final_port - 100:.1f}%</b> en este periodo.</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class='chart-explain'>📉 Tu cartera ha bajado un <b style='color:#EF553B'>{100 - final_port:.1f}%</b> en este periodo.</div>""", unsafe_allow_html=True)
            else:
                st.info("Datos insuficientes para el periodo.")
        else:
            st.info("Anade activos para ver el rendimiento.")

    with col_alloc:
        st.markdown("##### Distribucion")
        # Dynamic allocation advice
        if not df_final.empty and patrimonio_total > 0:
            max_peso = df_final['Peso %'].max()
            max_name = df_final.loc[df_final['Peso %'].idxmax(), 'Nombre']
            if max_peso > 40:
                alloc_msg = f"<b style='color:#EF553B'>⚠️ {esc(max_name)}</b> representa el <b>{max_peso:.0f}%</b> de tu cartera. Concentracion alta."
            elif max_peso > 25:
                alloc_msg = f"<b>{esc(max_name)}</b> es tu mayor posicion ({max_peso:.0f}%). Aceptable pero vigila."
            else:
                alloc_msg = f"Buena distribucion. Tu mayor posicion (<b>{esc(max_name)}</b>) es solo el {max_peso:.0f}%."
        else:
            alloc_msg = "Anade activos para ver la distribucion de tu cartera."
        st.markdown(f"""<div class='chart-explain'>{alloc_msg}</div>""", unsafe_allow_html=True)
        if patrimonio_total > 0:
            labels_p = (['Cash'] + df_final['Nombre'].tolist()) if not df_final.empty else ['Cash']
            values_p = ([total_liquidez] + df_final['Valor Acciones'].tolist()) if not df_final.empty else [total_liquidez]
            fig_pie = px.pie(names=labels_p, values=values_p, hole=0.72,
                              color_discrete_sequence=['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A',
                                                       '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52'])
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=9,
                                   pull=[0.02] * len(labels_p),
                                   hovertemplate='<b>%{label}</b><br>Valor: %{value:,.0f} EUR<br>Peso: %{percent}<extra></extra>',
                                   marker=dict(line=dict(color='#0a0e14', width=2)))
            fig_pie.update_layout(template="plotly_dark", height=360, showlegend=True,
                                   margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)',
                                   legend=dict(font=dict(size=9), orientation="h", y=-0.1),
                                   annotations=[dict(text=f"<b>{patrimonio_total:,.0f}<br>EUR</b>", x=0.5, y=0.5,
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
            st.markdown("""<div class='chart-explain'>
                Cada bloque es un activo. El <b>tamano</b> indica cuanto dinero tienes en el.
                <b style='color:#00CC96'>Verde</b> = ganando dinero, <b style='color:#EF553B'>rojo</b> = perdiendo.
            </div>""", unsafe_allow_html=True)
            fig_tree = px.treemap(df_final, path=['Nombre'], values='Valor Acciones', color='Rentabilidad %',
                                  color_continuous_scale=['#EF553B', '#2d1f1f', '#1e1e1e', '#1f2d1f', '#00CC96'],
                                  color_continuous_midpoint=0,
                                  hover_data={'Valor Acciones': ':,.0f', 'Rentabilidad %': ':+.1f'})
            fig_tree.update_traces(texttemplate='<b>%{label}</b><br>%{customdata[1]:+.1f}%',
                                    textfont_size=11,
                                    marker=dict(cornerradius=5))
            fig_tree.update_layout(template="plotly_dark", height=320, margin=dict(l=5, r=5, t=5, b=5),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    coloraxis_colorbar=dict(title="Rent.%", thickness=12, len=0.6))
            st.plotly_chart(fig_tree, use_container_width=True)

        with col_bar:
            st.markdown("##### P&L por Activo")
            st.markdown("""<div class='chart-explain'>
                Ganancia o perdida en euros de cada inversion. Barras <b style='color:#00CC96'>verdes</b> = ganas,
                <b style='color:#EF553B'>rojas</b> = pierdes.
            </div>""", unsafe_allow_html=True)
            df_sorted = df_final.sort_values('Ganancia', ascending=True)
            colors_b = ['#EF553B' if x < 0 else '#00CC96' for x in df_sorted['Ganancia']]
            fig_bar = go.Figure(go.Bar(
                x=df_sorted['Ganancia'], y=df_sorted['Nombre'], orientation='h',
                marker_color=colors_b,
                marker_line=dict(color='rgba(255,255,255,0.08)', width=1),
                text=df_sorted['Ganancia'].apply(lambda x: f"{x:+,.0f}EUR"), textposition='outside',
                hovertemplate='<b>%{y}</b><br>P&L: %{x:+,.2f} EUR<extra></extra>'))
            fig_bar.update_layout(template="plotly_dark", height=320, margin=dict(l=0, r=40, t=0, b=0),
                                   paper_bgcolor='rgba(0,0,0,0)',
                                   xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)', zeroline=True, zerolinecolor='rgba(255,255,255,0.15)'))
            fig_bar.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
            st.plotly_chart(fig_bar, use_container_width=True)

    # Drawdown
    if not daily_returns.empty and len(daily_returns) > 5:
        st.markdown("---")
        st.markdown("##### Drawdown (Peores caidas)")
        st.markdown("""<div class='chart-explain'>
            El <b>drawdown</b> muestra cuanto ha caido tu cartera desde su mejor momento. Es como medir
            "la peor racha". Un drawdown de <b>-10%</b> significa que en algun momento perdiste un 10% desde el punto mas alto.
            Es <b>normal</b> tener drawdowns; lo importante es que se recuperen.
        </div>""", unsafe_allow_html=True)
        cum_ret = (1 + daily_returns).cumprod()
        dd_s = (cum_ret - cum_ret.cummax()) / cum_ret.cummax() * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=dd_s.index, y=dd_s, fill='tozeroy',
                                     fillcolor='rgba(239,85,59,0.12)', line=dict(color='#EF553B', width=1.2, shape='spline'),
                                     hovertemplate='Fecha: %{x|%d %b %Y}<br>Caida: <b>%{y:.1f}%</b><extra></extra>',
                                     name='Drawdown'))
        fig_dd.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Caida %", hovermode="x unified",
                              xaxis_title="Fecha")
        fig_dd.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
        st.plotly_chart(fig_dd, use_container_width=True)
        worst_dd = dd_s.min()
        if worst_dd < -20:
            st.markdown(f"""<div class='chart-explain'>⚠️ Tu peor caida fue del <b style='color:#EF553B'>{worst_dd:.1f}%</b>. Esto puede ser normal en carteras agresivas, pero asegurate de estar comodo con este nivel de riesgo.</div>""", unsafe_allow_html=True)
        elif worst_dd < -10:
            st.markdown(f"""<div class='chart-explain'>📊 Tu peor caida fue del <b>{worst_dd:.1f}%</b>. Esta dentro de lo esperado para la mayoria de carteras.</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='chart-explain'>✅ Tu peor caida fue solo del <b style='color:#00CC96'>{worst_dd:.1f}%</b>. Tu cartera es bastante estable.</div>""", unsafe_allow_html=True)

    # Tabla
    if not df_final.empty:
        st.markdown("---")
        st.markdown("##### Posiciones")
        st.markdown("""<div class='chart-explain'>
            Tu lista completa de inversiones. <b>P.Medio</b> = precio al que compraste de media.
            <b>P.Actual</b> = precio actual en el mercado. <b>P&L</b> = ganancia o perdida total.
            <b>Peso%</b> = que porcentaje de tu cartera representa.
        </div>""", unsafe_allow_html=True)
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
    # Portfolio-aware market intro
    if my_tickers:
        mkt_intro = f"""<div class='info-tip'>
            <span class='tip-icon'>🌍</span> <b>Pulso del mercado.</b>
            Noticias filtradas para tus {len(my_tickers)} activos: <b>{', '.join(my_tickers[:5])}</b>{'...' if len(my_tickers) > 5 else ''}.
            Los indices en <b style='color:#00CC96'>verde</b> suben y en <b style='color:#EF553B'>rojo</b> bajan.
        </div>"""
    else:
        mkt_intro = """<div class='info-tip'>
            <span class='tip-icon'>🌍</span> <b>Pulso del mercado.</b>
            Aqui ves los principales indices bursatiles, noticias y activos populares.
            Anade activos en <b>Inversiones</b> para ver noticias personalizadas.
        </div>"""
    st.markdown(mkt_intro, unsafe_allow_html=True)

    # --- Indices en tiempo real ---
    st.markdown("##### Indices Principales")
    st.markdown("""<div class='chart-explain'>
        Los <b>indices</b> son como termometros del mercado. Agrupan las mayores empresas de cada region.
        Por ejemplo, el <b>S&P 500</b> incluye las 500 mayores empresas de EE.UU.
    </div>""", unsafe_allow_html=True)
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
            top_title = esc(top['title'])
            top_source = esc(top['source'])
            top_body = esc(top.get('body', ''))
            top_url = esc(top['url']) if top.get('url', '').startswith('http') else '#'
            top_date = esc(top['date'][:16]) if top.get('date') else ''
            img_html = f"<img src='{esc(top['image'])}' onerror=\"this.style.display='none'\" />" if top.get('image', '').startswith('http') else ""
            st.markdown(f"""
            <div class='news-card-full' style='border-left: 3px solid var(--accent);'>
                {img_html}
                <span class='news-source-badge'>{top_source}</span>
                <div class='news-title'><a href="{top_url}" target="_blank" rel="noopener noreferrer">{top_title}</a></div>
                <p style='color:var(--text-secondary); font-size:0.8rem; margin-top:6px;'>{top_body}</p>
                <div class='news-meta'>{top_date}</div>
            </div>""", unsafe_allow_html=True)

            # Grid de noticias
            rest = news[1:]
            for i in range(0, len(rest), 2):
                c1, c2 = st.columns(2)
                for j, col in enumerate([c1, c2]):
                    idx2 = i + j
                    if idx2 < len(rest):
                        n = rest[idx2]
                        n_title = esc(n['title'])
                        n_source = esc(n['source'])
                        n_url = esc(n['url']) if n.get('url', '').startswith('http') else '#'
                        n_date = esc(n['date'][:16]) if n.get('date') else ''
                        img_h = f"<img src='{esc(n['image'])}' onerror=\"this.style.display='none'\" style='height:100px;'/>" if n.get('image', '').startswith('http') else ""
                        with col:
                            st.markdown(f"""
                            <div class='news-card-full'>
                                {img_h}
                                <span class='news-source-badge'>{n_source}</span>
                                <div class='news-title'><a href="{n_url}" target="_blank" rel="noopener noreferrer">{n_title}</a></div>
                                <div class='news-meta'>{n_date}</div>
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
    # Portfolio-aware liquidez advice
    if pct_liq >= 20:
        liq_advice = f"Tienes <b>{pct_liq:.0f}%</b> en efectivo. Es un colchon generoso. Podrias considerar invertir parte si tienes buenas oportunidades."
    elif pct_liq >= 10:
        liq_advice = f"Tienes <b>{pct_liq:.0f}%</b> en efectivo. Nivel saludable dentro del rango recomendado (10-20%)."
    elif pct_liq >= 5:
        liq_advice = f"Tu liquidez es del <b style='color:#FFA15A'>{pct_liq:.0f}%</b>. Por debajo del 10% recomendado. Considera reforzar tu colchon."
    else:
        liq_advice = f"<b style='color:#EF553B'>Atencion</b>: solo tienes un <b>{pct_liq:.0f}%</b> en efectivo. Muy por debajo del minimo recomendado."
    st.markdown(f"""<div class='info-tip'>
        <span class='tip-icon'>💰</span> <b>Tu efectivo disponible.</b> {liq_advice}
    </div>""", unsafe_allow_html=True)
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
# REBALANCEO -- SOLO COMPRANDO (CALCULO AUTOMATICO)
# ======================================================================
elif pagina == "Rebalanceo":
    st.markdown("## Rebalanceo Inteligente")

    if df_final.empty:
        st.markdown("""<div class='info-tip'>
            <span class='tip-icon'>⚖️</span> <b>Rebalanceo.</b>
            Anade activos primero para poder rebalancear tu cartera.
        </div>""", unsafe_allow_html=True)
        st.warning("Anade activos primero.")
    else:
        # ---- Real-time portfolio analysis ----
        wts = df_final['Peso %'].tolist()
        names = df_final['Nombre'].tolist()
        max_w, min_w = max(wts), min(wts)
        over_asset = names[wts.index(max_w)]
        under_asset = names[wts.index(min_w)]
        weights_arr = (df_final['Peso %'] / 100).tolist()
        div_score = diversification_score(weights_arr)

        # Generate real-time advice
        consejos = []
        if max_w > 40:
            consejos.append(f"🚨 <b>{esc(over_asset)}</b> representa el {max_w:.1f}% de tu cartera. Una concentracion superior al 40% en un solo activo es arriesgada.")
        if min_w < 5 and len(wts) > 2:
            consejos.append(f"⚠️ <b>{esc(under_asset)}</b> solo pesa un {min_w:.1f}%. Considera aumentar su posicion para que tenga impacto real.")
        if div_score < 50:
            consejos.append("⚠️ Diversificacion baja. Repartir mejor el capital reduce el riesgo de perdidas grandes.")
        elif div_score < 70:
            consejos.append("💡 Diversificacion aceptable pero mejorable. Equilibrar pesos te dara mas estabilidad.")
        else:
            consejos.append("✅ Buena diversificacion. Mantener un rebalanceo periodico te ayudara a conservarla.")
        if len(wts) <= 2:
            consejos.append(f"💡 Con solo {len(wts)} activo(s), tu cartera es muy concentrada. Considera anadir mas posiciones.")

        equal_w = 100 / len(wts)
        drifts = [(names[i], wts[i] - equal_w) for i in range(len(wts))]
        big_drifts = [(n, d) for n, d in drifts if abs(d) > 10]
        if big_drifts:
            drift_msgs = [f"<b>{esc(n)}</b> ({d:+.1f}pp)" for n, d in sorted(big_drifts, key=lambda x: -abs(x[1]))]
            consejos.append(f"📊 Activos mas desviados del peso equitativo: {', '.join(drift_msgs[:3])}")

        reb_intro = f"""<div class='info-tip'>
            <span class='tip-icon'>⚖️</span> <b>Rebalanceo solo comprando.</b>
            Tu cartera tiene <b>{len(wts)} activos</b> con un patrimonio de <b>{total_inversiones:,.0f}EUR</b>.
            Selecciona los pesos objetivo y el plazo: el sistema calcula <b>matematicamente</b>
            cuanto necesitas invertir cada mes, sin vender ningun activo.
        </div>"""
        st.markdown(reb_intro, unsafe_allow_html=True)

        st.markdown("### 📋 Analisis de tu cartera actual")
        for consejo in consejos:
            st.markdown(f"<div style='padding:8px 14px; margin:4px 0; background:var(--bg-card); border-radius:8px; border-left:3px solid var(--accent); font-size:0.9rem;'>{consejo}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---- Helper: buy-only rebalance math ----
        def calcular_rebalanceo_compras(valores_actuales, pesos_objetivo):
            """
            Calcula el capital minimo B necesario para alcanzar los pesos objetivo
            solo comprando (nunca vendiendo).
            Matematica: B_i = w_i * (T + B) - V_i >= 0
            => B >= V_i/w_i - T para todo i
            => B = max(V_i/w_i) - T
            Retorna dict {nombre: euros_a_comprar}
            """
            T = sum(valores_actuales.values())
            # Find minimum new capital needed
            ratios = []
            for nombre, w in pesos_objetivo.items():
                if w > 0:
                    ratios.append(valores_actuales.get(nombre, 0) / (w / 100))
                elif valores_actuales.get(nombre, 0) > 0:
                    ratios.append(float('inf'))
            if not ratios:
                return {}, 0
            B = max(max(ratios) - T, 0)
            new_total = T + B
            compras = {}
            for nombre, w in pesos_objetivo.items():
                target_val = (w / 100) * new_total
                actual_val = valores_actuales.get(nombre, 0)
                compras[nombre] = max(0, round(target_val - actual_val, 2))
            return compras, B

        tab_manual, tab_strat, tab_suggest = st.tabs(["Plan Personalizado", "Estrategias", "Sugerencias"])

        # ============================================================
        # -- MANUAL --
        # ============================================================
        with tab_manual:
            st.markdown("#### 1. Selecciona los pesos objetivo")
            ws = {}
            tot_w = 0
            cols_per_row = min(len(df_final), 4)
            asset_list = list(df_final.iterrows())

            for batch_start in range(0, len(asset_list), cols_per_row):
                batch = asset_list[batch_start:batch_start + cols_per_row]
                cols = st.columns(len(batch))
                for col, (idx_r, r_row) in zip(cols, batch):
                    with col:
                        st.markdown(f"**{r_row['Nombre']}**")
                        st.caption(f"Actual: {r_row['Peso %']:.1f}%  ({r_row['Valor Acciones']:,.0f}€)")
                        w = st.number_input("Objetivo %", 0, 100, int(round(r_row['Peso %'])),
                                             key=f"reb_{idx_r}", label_visibility="collapsed")
                        ws[r_row['Nombre']] = w
                        tot_w += w

            color_t = "var(--accent)" if tot_w == 100 else "var(--red)"
            st.markdown(f"**Total pesos: <span style='color:{color_t}'>{tot_w}%</span>**", unsafe_allow_html=True)

            st.markdown("#### 2. Elige el plazo")
            meses_reb = st.slider("Meses para alcanzar los pesos objetivo", 1, 36, 6,
                                   help="El sistema divide el total a invertir entre estos meses.")

            if tot_w != 100:
                st.warning(f"Los pesos deben sumar 100% (actual: {tot_w}%). Ajusta antes de continuar.")
            else:
                if st.button("📅 Calcular Plan de Compras", type="primary", use_container_width=True):
                    # --- Core math ---
                    valores_actuales = {r['Nombre']: r['Valor Acciones'] for _, r in df_final.iterrows()}
                    compras_total, capital_necesario = calcular_rebalanceo_compras(valores_actuales, ws)
                    total_a_comprar = sum(compras_total.values())
                    aporte_mensual = total_a_comprar / meses_reb if meses_reb > 0 else total_a_comprar
                    capital_final = total_inversiones + total_a_comprar

                    # ---- Summary metrics ----
                    st.markdown("### 📊 Tu Plan de Rebalanceo")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Inversion necesaria", f"{total_a_comprar:,.0f}€",
                              help="Capital total que necesitas aportar para alcanzar los pesos objetivo, calculado matematicamente.")
                    k2.metric("Aporte mensual", f"{aporte_mensual:,.0f}€",
                              help="Divide el total entre los meses seleccionados.")
                    k3.metric("Duracion", f"{meses_reb} meses")
                    k4.metric("Cartera final", f"{capital_final:,.0f}€")

                    # Liquidity check
                    if total_liquidez >= total_a_comprar:
                        st.success(f"✅ Tu liquidez actual ({total_liquidez:,.0f}€) cubre el rebalanceo completo.")
                    elif total_liquidez > 0:
                        falta = total_a_comprar - total_liquidez
                        st.warning(f"💰 Tu liquidez ({total_liquidez:,.0f}€) cubre parte. Necesitas aportar {falta:,.0f}€ adicionales en {meses_reb} meses ({falta/meses_reb:,.0f}€/mes).")
                    else:
                        st.info(f"💰 Necesitas aportar {total_a_comprar:,.0f}€ en {meses_reb} meses ({aporte_mensual:,.0f}€/mes).")

                    # ---- Per-asset summary table ----
                    resumen_data = []
                    for _, r_row in df_final.iterrows():
                        nombre = r_row['Nombre']
                        compra = compras_total.get(nombre, 0)
                        nuevo_val = r_row['Valor Acciones'] + compra
                        nuevo_pct = nuevo_val / capital_final * 100 if capital_final > 0 else 0
                        resumen_data.append({
                            'Activo': nombre,
                            'Valor Actual': r_row['Valor Acciones'],
                            'Peso Actual': r_row['Peso %'],
                            'Peso Objetivo': ws[nombre],
                            'Comprar Total': compra,
                            'Comprar/Mes': compra / meses_reb if meses_reb > 0 else compra,
                            'Peso Final': nuevo_pct,
                        })
                    df_resumen = pd.DataFrame(resumen_data)
                    st.dataframe(df_resumen.style.format({
                        'Valor Actual': '{:,.0f}€', 'Peso Actual': '{:.1f}%', 'Peso Objetivo': '{:.0f}%',
                        'Comprar Total': '{:,.0f}€', 'Comprar/Mes': '{:,.0f}€', 'Peso Final': '{:.1f}%'
                    }).background_gradient(subset=['Comprar/Mes'], cmap='Greens', vmin=0),
                        use_container_width=True)

                    # ---- Monthly plan table ----
                    st.markdown("### 📅 Plan Mensual Detallado")
                    st.markdown("""<div class='chart-explain'>
                        Cuanto invertir en cada activo cada mes (en EUR). Siguiendo este plan
                        alcanzaras los pesos objetivo en {m} meses sin vender nada.
                    </div>""".format(m=meses_reb), unsafe_allow_html=True)

                    plan_rows = []
                    acumulado = {k: 0.0 for k in compras_total}
                    for mes in range(1, meses_reb + 1):
                        restante = {k: compras_total[k] - acumulado[k] for k in compras_total}
                        restante = {k: max(0, v) for k, v in restante.items()}
                        total_rest = sum(restante.values())
                        row = {'Mes': f"Mes {mes}"}
                        for nombre in compras_total:
                            if total_rest > 0:
                                prop = restante[nombre] / total_rest
                                compra = min(aporte_mensual * prop, restante[nombre])
                            else:
                                compra = 0
                            acumulado[nombre] += compra
                            row[nombre] = round(compra, 2)
                        row['Total'] = round(sum(v for k, v in row.items() if k != 'Mes'), 2)
                        plan_rows.append(row)

                    df_plan = pd.DataFrame(plan_rows)
                    fmt = {col: '{:,.2f}' for col in df_plan.columns if col != 'Mes'}
                    st.dataframe(df_plan.style.format(fmt).background_gradient(
                        subset=[c for c in df_plan.columns if c not in ['Mes', 'Total']],
                        cmap='Greens', vmin=0
                    ), use_container_width=True, height=min(400, 40 + 35 * len(df_plan)))

                    # ---- Bar chart: actual vs objective vs final ----
                    st.markdown("### Peso Actual vs Objetivo")
                    fig_reb = go.Figure()
                    fig_reb.add_trace(go.Bar(name='Actual', x=df_resumen['Activo'], y=df_resumen['Peso Actual'],
                                              marker_color='#636EFA',
                                              marker_line=dict(color='rgba(255,255,255,0.15)', width=1),
                                              hovertemplate='<b>%{x}</b><br>Actual: %{y:.1f}%<extra></extra>'))
                    fig_reb.add_trace(go.Bar(name='Objetivo', x=df_resumen['Activo'], y=df_resumen['Peso Objetivo'],
                                              marker_color='#FFA15A', opacity=0.6,
                                              marker_line=dict(color='rgba(255,255,255,0.15)', width=1),
                                              hovertemplate='<b>%{x}</b><br>Objetivo: %{y:.1f}%<extra></extra>'))
                    fig_reb.add_trace(go.Bar(name='Tras Rebalanceo', x=df_resumen['Activo'], y=df_resumen['Peso Final'],
                                              marker_color='#00CC96',
                                              marker_line=dict(color='rgba(255,255,255,0.15)', width=1),
                                              hovertemplate='<b>%{x}</b><br>Final: %{y:.1f}%<extra></extra>'))
                    fig_reb.update_layout(template="plotly_dark", barmode='group', height=350,
                                           paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Peso (%)",
                                           legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                                           margin=dict(t=40))
                    fig_reb.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
                    st.plotly_chart(fig_reb, use_container_width=True)

                    # ---- Area chart: monthly progression ----
                    st.markdown("### 📈 Evolucion mensual de la cartera")
                    st.markdown("""<div class='chart-explain'>
                        Como crece cada posicion mes a mes siguiendo el plan de compras.
                    </div>""", unsafe_allow_html=True)

                    colors_prog = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                                   '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
                    prog_data = []
                    valor_prog = {r['Nombre']: r['Valor Acciones'] for _, r in df_final.iterrows()}
                    for nombre in compras_total:
                        prog_data.append({'Mes': 0, 'Activo': nombre, 'Valor': valor_prog[nombre]})

                    acum_prog = {k: 0.0 for k in compras_total}
                    for mes in range(1, meses_reb + 1):
                        restante = {k: compras_total[k] - acum_prog[k] for k in compras_total}
                        restante = {k: max(0, v) for k, v in restante.items()}
                        total_rest = sum(restante.values())
                        for nombre in compras_total:
                            if total_rest > 0:
                                prop = restante[nombre] / total_rest
                                compra = min(aporte_mensual * prop, restante[nombre])
                            else:
                                compra = 0
                            acum_prog[nombre] += compra
                            valor_prog[nombre] = df_final.loc[df_final['Nombre'] == nombre, 'Valor Acciones'].iloc[0] + acum_prog[nombre]
                        for nombre in compras_total:
                            prog_data.append({'Mes': mes, 'Activo': nombre, 'Valor': valor_prog[nombre]})

                    df_prog = pd.DataFrame(prog_data)
                    fig_prog = go.Figure()
                    for i, nombre in enumerate(compras_total.keys()):
                        df_act = df_prog[df_prog['Activo'] == nombre]
                        fig_prog.add_trace(go.Scatter(
                            x=df_act['Mes'], y=df_act['Valor'],
                            name=nombre, stackgroup='one', mode='lines',
                            line=dict(width=0.5, shape='spline', color=colors_prog[i % len(colors_prog)]),
                            hovertemplate=f'<b>{nombre}</b><br>Mes %{{x}}<br>Valor: %{{y:,.0f}}€<extra></extra>'
                        ))
                    fig_prog.update_layout(
                        template="plotly_dark", height=350, paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Mes", yaxis_title="Valor (EUR)",
                        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
                        margin=dict(t=40), hovermode="x unified"
                    )
                    fig_prog.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
                    fig_prog.update_xaxes(gridcolor='rgba(255,255,255,0.04)', dtick=max(1, meses_reb // 12))
                    st.plotly_chart(fig_prog, use_container_width=True)

                    # ---- Per-asset detail ----
                    st.markdown("### 🔍 Detalle por activo")
                    for _, r_row in df_final.iterrows():
                        nombre = r_row['Nombre']
                        compra = compras_total.get(nombre, 0)
                        compra_mensual_act = compra / meses_reb if meses_reb > 0 else compra
                        nuevo_val = r_row['Valor Acciones'] + compra
                        nuevo_pct = nuevo_val / capital_final * 100 if capital_final > 0 else 0

                        if compra < 1:
                            estado = "✅ Ya en objetivo — no necesitas comprar"
                            color_estado = "var(--accent)"
                        else:
                            estado = f"📈 De {r_row['Peso %']:.1f}% a {nuevo_pct:.1f}% (+{compra:,.0f}€)"
                            color_estado = "#00CC96"

                        with st.expander(f"{nombre} — {compra_mensual_act:,.0f}€/mes × {meses_reb} meses = {compra:,.0f}€"):
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Valor actual", f"{r_row['Valor Acciones']:,.0f}€")
                            c2.metric("Comprar total", f"{compra:,.0f}€")
                            c3.metric("Valor final", f"{nuevo_val:,.0f}€")
                            st.markdown(f"<span style='color:{color_estado}; font-weight:600;'>{estado}</span>", unsafe_allow_html=True)
                            if compra_mensual_act >= 1:
                                meses_list = list(range(1, meses_reb + 1))
                                compras_mensuales = [compra_mensual_act] * meses_reb
                                total_prev = compra_mensual_act * (meses_reb - 1)
                                compras_mensuales[-1] = compra - total_prev
                                fig_mini = go.Figure(go.Bar(
                                    x=meses_list, y=compras_mensuales,
                                    marker_color=colors_prog[list(compras_total.keys()).index(nombre) % len(colors_prog)],
                                    marker_line=dict(color='rgba(255,255,255,0.1)', width=1),
                                    hovertemplate='Mes %{x}<br>Comprar: %{y:,.0f}€<extra></extra>'
                                ))
                                fig_mini.update_layout(
                                    template="plotly_dark", height=150, paper_bgcolor='rgba(0,0,0,0)',
                                    margin=dict(l=0, r=0, t=5, b=25), xaxis_title="Mes", yaxis_title="€",
                                    xaxis=dict(dtick=max(1, meses_reb // 12))
                                )
                                fig_mini.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
                                st.plotly_chart(fig_mini, use_container_width=True)

                    # ---- Consejos finales ----
                    st.markdown("### 💬 Consejos sobre tu rebalanceo")
                    advice_cards = []

                    # How close does it get?
                    diff_actual = sum(abs(r['Peso %'] - ws[r['Nombre']]) for _, r in df_final.iterrows())
                    if diff_actual < 3:
                        advice_cards.append(("✅", "Tu cartera ya esta muy cerca de los pesos objetivo. No necesitas un gran esfuerzo."))
                    elif total_a_comprar < 100:
                        advice_cards.append(("✅", f"Solo necesitas {total_a_comprar:,.0f}€ para alcanzar el equilibrio perfecto."))
                    else:
                        advice_cards.append(("📊", f"Necesitas aportar <b>{total_a_comprar:,.0f}€</b> ({aporte_mensual:,.0f}€/mes durante {meses_reb} meses) para alcanzar exactamente los pesos que quieres."))

                    if meses_reb <= 2:
                        advice_cards.append(("⚡", f"Plan agresivo de {meses_reb} mes(es). Requiere {aporte_mensual:,.0f}€/mes. Asegurate de tener esa capacidad de ahorro."))
                    elif meses_reb <= 6:
                        advice_cards.append(("👍", f"Plazo razonable. {aporte_mensual:,.0f}€/mes es un ritmo solido."))
                    elif meses_reb >= 18:
                        advice_cards.append(("🐢", f"Plan a {meses_reb} meses. Revisa trimestralmente: los precios cambian y podrias necesitar ajustar los importes."))

                    over_target = [n for n, c in compras_total.items() if c < 1]
                    if over_target:
                        advice_cards.append(("📌", f"<b>{', '.join(esc(n) for n in over_target)}</b> ya {'supera' if len(over_target)==1 else 'superan'} el peso objetivo. El calculo redirige el capital a los activos infraponderados."))

                    under_buy = [(n, c) for n, c in compras_total.items() if c > 0]
                    if under_buy:
                        biggest = max(under_buy, key=lambda x: x[1])
                        advice_cards.append(("🎯", f"La mayor compra es <b>{esc(biggest[0])}</b> con {biggest[1]:,.0f}€ ({biggest[1]/meses_reb:,.0f}€/mes). Es el activo mas infraponderado respecto a tu objetivo."))

                    for icon, text in advice_cards:
                        st.markdown(f"""<div style='padding:10px 16px; margin:6px 0; background:var(--bg-card);
                            border-radius:8px; border-left:3px solid var(--accent); font-size:0.9rem;'>
                            {icon} {text}
                        </div>""", unsafe_allow_html=True)

        # ============================================================
        # -- ESTRATEGIAS PREDEFINIDAS --
        # ============================================================
        with tab_strat:
            st.markdown("""<div class='chart-explain'>
                Elige una estrategia predefinida. El sistema calcula automaticamente los pesos
                y el capital exacto que necesitas aportar cada mes.
            </div>""", unsafe_allow_html=True)
            estrategia = st.selectbox("Estrategia:", [
                "Equiponderado",
                "Momentum (mas a ganadores)",
                "Minima Volatilidad",
            ], help="Equiponderado = misma cantidad en cada activo. Momentum = mas en los que mejor van. Min. Volatilidad = mas en los mas estables.")

            meses_strat = st.slider("Meses para rebalancear", 1, 36, 6, key="meses_strat",
                                     help="Numero de meses para alcanzar los pesos objetivo.")

            if st.button("Calcular Plan", type="primary", key="calc_strat"):
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

                # Calculate using exact math
                valores_s = {r['Nombre']: r['Valor Acciones'] for _, r in df_final.iterrows()}
                compras_strat, cap_needed = calcular_rebalanceo_compras(valores_s, tw)
                total_buy = sum(compras_strat.values())
                aporte_m_s = total_buy / meses_strat if meses_strat > 0 else total_buy
                capital_final_s = total_inversiones + total_buy

                # Summary
                k1, k2, k3 = st.columns(3)
                k1.metric("Inversion necesaria", f"{total_buy:,.0f}€")
                k2.metric("Aporte mensual", f"{aporte_m_s:,.0f}€")
                k3.metric("Cartera final", f"{capital_final_s:,.0f}€")

                # Table
                wt_data = []
                for _, r in df_final.iterrows():
                    n = r['Nombre']
                    wt_data.append({
                        'Activo': n,
                        'Actual %': r['Peso %'],
                        'Objetivo %': tw[n],
                        'Comprar Total': compras_strat[n],
                        'Comprar/Mes': compras_strat[n] / meses_strat if meses_strat > 0 else compras_strat[n],
                    })
                df_strat = pd.DataFrame(wt_data)
                st.dataframe(df_strat.style.format({
                    'Actual %': '{:.1f}', 'Objetivo %': '{:.1f}', 'Comprar Total': '{:,.0f}€', 'Comprar/Mes': '{:,.0f}€'
                }).background_gradient(subset=['Comprar/Mes'], cmap='Greens', vmin=0), use_container_width=True)

                # Monthly plan
                st.markdown(f"**📅 Plan mensual ({meses_strat} meses):**")
                plan_rows_s = []
                acum_s = {k: 0.0 for k in compras_strat}
                for mes in range(1, meses_strat + 1):
                    restante = {k: compras_strat[k] - acum_s[k] for k in compras_strat}
                    restante = {k: max(0, v) for k, v in restante.items()}
                    total_rest = sum(restante.values())
                    row = {'Mes': f"Mes {mes}"}
                    for nombre in compras_strat:
                        if total_rest > 0:
                            prop = restante[nombre] / total_rest
                            compra = min(aporte_m_s * prop, restante[nombre])
                        else:
                            compra = 0
                        acum_s[nombre] += compra
                        row[nombre] = round(compra, 2)
                    row['Total'] = round(sum(v for k, v in row.items() if k != 'Mes'), 2)
                    plan_rows_s.append(row)

                df_plan_s = pd.DataFrame(plan_rows_s)
                fmt_s = {col: '{:,.2f}' for col in df_plan_s.columns if col != 'Mes'}
                st.dataframe(df_plan_s.style.format(fmt_s).background_gradient(
                    subset=[c for c in df_plan_s.columns if c not in ['Mes', 'Total']],
                    cmap='Greens', vmin=0
                ), use_container_width=True, height=min(400, 40 + 35 * len(df_plan_s)))

                # Strategy advice
                if estrategia == "Equiponderado":
                    st.info("💡 **Equiponderado:** Cada activo tendra el mismo peso. Ideal para maxima diversificacion sin preferencias.")
                elif estrategia == "Momentum (mas a ganadores)":
                    st.info("💡 **Momentum:** Mas peso a los activos con mejor rentabilidad reciente. Mayor potencial pero mas riesgo si cambia la tendencia.")
                else:
                    st.info("💡 **Minima Volatilidad:** Priorizas estabilidad. Los activos menos volatiles reciben mas peso.")

        # ============================================================
        # -- SUGERENCIAS --
        # ============================================================
        with tab_suggest:
            st.markdown("##### Que falta en tu cartera?")
            st.markdown("""<div class='info-tip'>
                <span class='tip-icon'>💡</span> <b>Sugerencias automaticas.</b>
                Analizamos los sectores de la economia donde no tienes exposicion y te sugerimos
                ETFs (fondos que replican indices) para cubrirlos. Una cartera diversificada en
                diferentes sectores reduce el riesgo de depender de una sola industria.
            </div>""", unsafe_allow_html=True)

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
        st.markdown("""<div class='info-tip'>
            <span class='tip-icon'>🔬</span> <b>Analisis profundo de tu cartera.</b>
            Anade activos en la seccion <b>Inversiones</b> para ver el analisis completo.
        </div>""", unsafe_allow_html=True)
        st.warning("Anade activos.")
    else:
        weights = (df_final['Peso %'] / 100).tolist()
        div = diversification_score(weights)
        n_a = len(df_final)
        max_w = df_final['Peso %'].max()
        max_w_name = df_final.loc[df_final['Peso %'].idxmax(), 'Nombre']

        # Portfolio-aware intro
        if div >= 80:
            radio_msg = f"Tu cartera tiene {n_a} activos bien repartidos. Diversificacion excelente."
        elif div >= 50:
            radio_msg = f"Tienes {n_a} activos pero <b>{esc(max_w_name)}</b> pesa un {max_w:.0f}%. Podrías distribuir mejor."
        else:
            radio_msg = f"Cartera concentrada: <b>{esc(max_w_name)}</b> representa el {max_w:.0f}% del total. Considera diversificar."

        st.markdown(f"""<div class='info-tip'>
            <span class='tip-icon'>🔬</span> <b>Analisis profundo de tu cartera.</b> {radio_msg}
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        # Gauge chart for diversification
        with c1:
            gauge_color = '#00CC96' if div >= 70 else ('#FFA15A' if div >= 40 else '#EF553B')
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=div,
                title={'text': "Diversificacion", 'font': {'size': 13, 'color': '#EAEAEA'}},
                number={'suffix': '/100', 'font': {'size': 22, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'rgba(0,0,0,0)',
                             'dtick': 25, 'tickfont': {'size': 9, 'color': '#555'}},
                    'bar': {'color': gauge_color, 'thickness': 0.75},
                    'bgcolor': 'rgba(26,31,41,0.8)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(239,85,59,0.12)'},
                        {'range': [40, 70], 'color': 'rgba(255,161,90,0.12)'},
                        {'range': [70, 100], 'color': 'rgba(0,204,150,0.12)'},
                    ],
                }
            ))
            fig_gauge.update_layout(height=160, margin=dict(l=20, r=20, t=35, b=10),
                                     paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_gauge, use_container_width=True)

        c2.metric("Activos", len(df_final), help="Numero total de inversiones diferentes.")
        c3.metric("Mayor peso", f"{df_final['Peso %'].max():.1f}%", help="Tu activo mas grande. Idealmente < 30%.")
        c4.metric("Invertido", f"{df_final['Dinero Invertido'].sum():,.2f}EUR", help="Dinero total que has puesto en inversiones.")

        # Diagnostico automatico
        if div >= 80:
            st.markdown("<span class='health-badge health-good'>Buena diversificacion</span>", unsafe_allow_html=True)
        elif div >= 50:
            st.markdown("<span class='health-badge health-warn'>Diversificacion mejorable</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='health-badge health-bad'>Cartera muy concentrada - considera diversificar</span>", unsafe_allow_html=True)

        st.markdown("---")
        tab_corr, tab_sec, tab_brok, tab_det = st.tabs(["Correlacion", "Sectores", "Brokers", "Detalle"])

        with tab_corr:
            st.markdown("""<div class='info-tip'>
                <span class='tip-icon'>🔗</span> <b>Correlacion entre activos.</b>
                Mide si tus activos se mueven juntos. Un valor de <b>1.0</b> = se mueven exactamente igual
                (poca diversificacion), <b>0</b> = no tienen relacion, <b>-1.0</b> = se mueven en direccion opuesta
                (ideal para diversificar). Los colores <b style='color:#00CC96'>verdes</b> indican alta correlacion
                y <b style='color:#EF553B'>rojos</b> baja o negativa.
            </div>""", unsafe_allow_html=True)
            if not history_data.empty and len(history_data.columns) > 1:
                valid_c = [c for c in history_data.columns if c in my_tickers]
                if len(valid_c) > 1:
                    corr = history_data[valid_c].pct_change().dropna().corr()
                    fig_c = px.imshow(corr, text_auto='.2f',
                                      color_continuous_scale=['#EF553B', '#3d1f1f', '#1e1e1e', '#1f3d2f', '#00CC96'],
                                      aspect='equal', zmin=-1, zmax=1)
                    fig_c.update_layout(template="plotly_dark", height=max(350, len(valid_c) * 55),
                                         paper_bgcolor='rgba(0,0,0,0)',
                                         margin=dict(l=10, r=10, t=10, b=10),
                                         coloraxis_colorbar=dict(title="Corr", thickness=12, len=0.7))
                    fig_c.update_traces(textfont=dict(size=11, color='white'))
                    st.plotly_chart(fig_c, use_container_width=True)

                    # Dynamic portfolio-aware correlation analysis
                    high_c = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                              for i in range(len(corr)) for j in range(i + 1, len(corr))
                              if abs(corr.iloc[i, j]) > 0.8]
                    low_c = [(corr.columns[i], corr.columns[j], corr.iloc[i, j])
                              for i in range(len(corr)) for j in range(i + 1, len(corr))
                              if corr.iloc[i, j] < 0.2]
                    avg_corr = corr.values[np.triu_indices_from(corr.values, 1)].mean() if len(valid_c) > 1 else 0

                    corr_summary = f"""<div class='chart-explain'>
                        <b>Correlacion media de tu cartera: {avg_corr:.2f}</b>.
                        {'Tus activos estan bastante correlacionados (se mueven juntos). Anaadir activos de sectores diferentes ayudaria.' if avg_corr > 0.7 else
                         'Buena descorrelacion entre activos.' if avg_corr < 0.4 else
                         'Correlacion moderada. Hay margen para diversificar mas.'}
                    </div>"""
                    st.markdown(corr_summary, unsafe_allow_html=True)

                    if high_c:
                        st.warning(f"Pares muy correlacionados (>0.8): {len(high_c)} encontrados")
                        for a_t, b_t, v in high_c[:5]:
                            st.write(f"- {a_t} <-> {b_t}: **{v:.2f}**")
                    if low_c:
                        st.success(f"Buenas combinaciones (<0.2): {', '.join([f'{a}-{b}' for a, b, v in low_c[:3]])}")
                else:
                    st.info("Necesitas 2+ activos.")
            else:
                st.info("Sin datos historicos.")

        with tab_sec:
            st.markdown("""<div class='info-tip'>
                <span class='tip-icon'>🏭</span> <b>Distribucion por sectores.</b>
                Muestra en que industrias esta invertido tu dinero (tecnologia, salud, finanzas, etc.).
                Una buena cartera suele estar repartida entre <b>al menos 3-4 sectores diferentes</b>
                para no depender de uno solo.
            </div>""", unsafe_allow_html=True)
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
                                    color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_s.update_traces(textposition='outside', textfont_size=10,
                                         marker_line=dict(color='rgba(255,255,255,0.08)', width=1),
                                         hovertemplate='<b>%{x}</b><br>Valor: %{y:,.0f} EUR<extra></extra>')
                    fig_s.update_layout(template="plotly_dark", height=340, showlegend=False,
                                         paper_bgcolor='rgba(0,0,0,0)',
                                         xaxis=dict(tickangle=45),
                                         yaxis_title="Valor (EUR)")
                    fig_s.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
                    st.plotly_chart(fig_s, use_container_width=True)
                with cs2:
                    fig_sp = px.pie(df_sec, names='Sector', values='Valor', hole=0.55,
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig_sp.update_traces(textposition='inside', textinfo='percent+label', textfont_size=9,
                                          marker=dict(line=dict(color='#0a0e14', width=1.5)),
                                          hovertemplate='<b>%{label}</b><br>Valor: %{value:,.0f} EUR<br>%{percent}<extra></extra>')
                    fig_sp.update_layout(template="plotly_dark", height=340, paper_bgcolor='rgba(0,0,0,0)',
                                          showlegend=False)
                    st.plotly_chart(fig_sp, use_container_width=True)

        with tab_brok:
            st.markdown("""<div class='info-tip'>
                <span class='tip-icon'>🏦</span> <b>Distribucion por brokers.</b>
                Muestra como esta repartido tu dinero entre los diferentes brokers que usas.
                Tener mas de un broker puede ser bueno para diversificar riesgo de plataforma.
            </div>""", unsafe_allow_html=True)
            if 'platform' in df_final.columns:
                bk = df_final.groupby('platform').agg(Valor=('Valor Acciones', 'sum'), N=('ticker', 'count'),
                                                       PnL=('Ganancia', 'sum')).reset_index()
                bk.columns = ['Broker', 'Valor', 'Activos', 'P&L']
                cb1, cb2 = st.columns(2)
                with cb1:
                    fig_bk = px.pie(bk, names='Broker', values='Valor', hole=0.55,
                                     color_discrete_sequence=['#636EFA', '#00CC96', '#EF553B', '#AB63FA', '#FFA15A', '#19D3F3'])
                    fig_bk.update_traces(textposition='inside', textinfo='percent+label', textfont_size=10,
                                          marker=dict(line=dict(color='#0a0e14', width=1.5)),
                                          hovertemplate='<b>%{label}</b><br>Valor: %{value:,.0f} EUR<extra></extra>')
                    fig_bk.update_layout(template="plotly_dark", height=340, paper_bgcolor='rgba(0,0,0,0)',
                                          showlegend=False)
                    st.plotly_chart(fig_bk, use_container_width=True)
                with cb2:
                    st.dataframe(bk.style.format({'Valor': '{:,.2f}EUR', 'P&L': '{:+,.2f}EUR'}), use_container_width=True)

        with tab_det:
            st.markdown("""<div class='chart-explain'>
                Selecciona un activo para ver sus datos fundamentales, el grafico de precio historico
                y tu precio medio de compra (linea naranja punteada). Si el precio actual esta <b>por encima</b>
                de tu precio medio, estas ganando en esa inversion.
            </div>""", unsafe_allow_html=True)
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
                        pe = info_s.get('pe_ratio', 0)
                        beta_s = info_s.get('beta', 0)
                        div_y = info_s.get('dividend_yield', 0) * 100
                        st.write(f"**P/E:** {pe:.2f} - **Beta:** {beta_s:.2f}")
                        st.write(f"**Dividendo:** {div_y:.2f}%")
                        mc = info_s.get('market_cap', 0)
                        if mc > 1e12: st.write(f"**Market Cap:** {mc/1e12:.2f}T")
                        elif mc > 1e9: st.write(f"**Market Cap:** {mc/1e9:.2f}B")
                        # Explicaciones de metricas
                        with st.expander("ℹ️ ¿Que significan estas metricas?"):
                            st.markdown(f"""
                            - **P/E (Price-to-Earnings):** Cuantos anos de beneficios tardarias en recuperar lo invertido. P/E de {pe:.0f} significa que por cada EUR 1 de beneficio, pagas {pe:.0f} EUR. Un P/E < 15 es "barato", > 25 es "caro".
                            - **Beta:** Si el mercado sube un 1%, este activo sube aprox. un {beta_s:.1f}%. Beta > 1 = mas arriesgado que el mercado.
                            - **Dividendo:** Reparte un {div_y:.2f}% anual en pagos. Es como un "interes" que recibes por mantener la accion.
                            - **Market Cap:** Tamano de la empresa. Mega (>100B), Large (>10B), Mid (>2B), Small (<2B).
                            """)
                with cd2:
                    st.metric("Valor", f"{row_s['Valor Acciones']:,.2f}EUR")
                    st.metric("P&L", f"{row_s['Ganancia']:+,.2f}EUR", f"{row_s['Rentabilidad %']:+.2f}%")
                    st.metric("Peso", f"{row_s['Peso %']:.1f}%")
                if not history_data.empty and tk_s in history_data.columns:
                    h_a = history_data[tk_s].dropna()
                    if not h_a.empty:
                        fig_i = go.Figure()
                        fig_i.add_trace(go.Scatter(x=h_a.index, y=h_a, line=dict(color='#00CC96', width=2, shape='spline'),
                                                    fill='tozeroy', fillcolor='rgba(0,204,150,0.05)', name=tk_s,
                                                    hovertemplate='%{x|%d %b %Y}<br><b>%{y:,.2f}</b><extra></extra>'))
                        fig_i.add_hline(y=row_s['avg_price'], line_dash="dash", line_color="#FFA15A",
                                         annotation_text=f"Tu precio de compra: {row_s['avg_price']:.2f}",
                                         annotation_font_color="#FFA15A", annotation_font_size=11)
                        # Add 52-week high/low annotations
                        if info_s:
                            hi52 = info_s.get('fifty_two_week_high', 0)
                            lo52 = info_s.get('fifty_two_week_low', 0)
                            if hi52 > 0:
                                fig_i.add_hline(y=hi52, line_dash="dot", line_color="rgba(0,204,150,0.3)", line_width=0.8,
                                                 annotation_text=f"Max 52s: {hi52:.2f}", annotation_font_color="#555", annotation_font_size=9)
                            if lo52 > 0:
                                fig_i.add_hline(y=lo52, line_dash="dot", line_color="rgba(239,85,59,0.3)", line_width=0.8,
                                                 annotation_text=f"Min 52s: {lo52:.2f}", annotation_font_color="#555", annotation_font_size=9,
                                                 annotation_position="bottom right")
                        fig_i.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)',
                                             margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                                             yaxis_title="Precio", xaxis_title="Fecha")
                        fig_i.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
                        fig_i.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
                        st.plotly_chart(fig_i, use_container_width=True)

# ======================================================================
# SIMULADOR (MONTE CARLO)
# ======================================================================
elif pagina == "Simulador":
    st.markdown("## Simulador Monte Carlo")
    # Compute portfolio volatility for intro
    vol_anual = None
    if not history_data.empty:
        _d = history_data.pct_change().dropna()
        if len(_d) > 20:
            vol_anual = float(np.clip(_d.mean(axis=1).std() * np.sqrt(252), 0.05, 0.8) * 100)
    # Portfolio-aware intro
    if total_inversiones > 0:
        vol_text = f" ({vol_anual:.1f}% anual)" if vol_anual else ""
        sim_intro = f"""<div class='info-tip'>
            <span class='tip-icon'>🎲</span> <b>¿Cuanto podria valer tu cartera de {total_inversiones:,.0f}EUR en el futuro?</b>
            Basandonos en la volatilidad real de tu cartera{vol_text}, simulamos miles de escenarios.
            Cuanto mas tiempo y mas simulaciones, mas fiable es la proyeccion.
        </div>"""
    else:
        sim_intro = """<div class='info-tip'>
            <span class='tip-icon'>🎲</span> <b>Simulador de futuro.</b>
            Genera miles de escenarios posibles. Anade activos primero para usar datos reales de tu cartera.
        </div>"""
    st.markdown(sim_intro, unsafe_allow_html=True)

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
                                  line=dict(color='rgba(0,0,0,0)'), name='Rango amplio (80%)',
                                  hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]), y=np.concatenate([p75, p25[::-1]]),
                                  fill='toself', fillcolor='rgba(0,204,150,0.15)',
                                  line=dict(color='rgba(0,0,0,0)'), name='Rango probable (50%)',
                                  hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=x, y=p50, line=dict(color='#00CC96', width=3, shape='spline'), name='Resultado mas probable',
                                  hovertemplate='Ano %{x:.1f}<br><b>%{y:,.0f} EUR</b><extra></extra>'))
        # Capital aportado line
        capital_line = np.array([cap + aport * 12 * t for t in x])
        fig.add_trace(go.Scatter(x=x, y=capital_line, line=dict(color='#FFA15A', width=1.5, dash='dot'),
                                  name='Capital aportado', hovertemplate='Ano %{x:.1f}<br>Aportado: %{y:,.0f} EUR<extra></extra>'))
        for si in np.random.choice(n_sims, min(15, n_sims), replace=False):
            fig.add_trace(go.Scatter(x=x, y=paths[si], line=dict(color='rgba(255,255,255,0.03)', width=0.5),
                                      showlegend=False, hoverinfo='skip'))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', height=420,
                           xaxis_title="Anos en el futuro", yaxis_title="Valor en EUR", hovermode="x unified",
                           margin=dict(l=0, r=0, t=10, b=0))
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.04)')
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.04)')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""<div class='chart-explain'>
            <b>¿Como leer este grafico?</b> La <b style='color:#00CC96'>linea verde</b> es el resultado mas probable (mediana).
            La <b>banda oscura</b> cubre el 50% de escenarios centrales (P25-P75). La <b>banda clara</b> cubre el 80% de escenarios (P10-P90).
            Cuanto mas ancha la banda, mas incertidumbre hay.
        </div>""", unsafe_allow_html=True)

        cr1, cr2, cr3, cr4, cr5 = st.columns(5)
        cr1.metric("Pesimista", f"{p10[-1]:,.0f}EUR", help="P10: solo el 10% de escenarios acaban peor que esto.")
        cr2.metric("Cauto", f"{p25[-1]:,.0f}EUR", help="P25: el 25% de escenarios acaban por debajo.")
        cr3.metric("Probable", f"{p50[-1]:,.0f}EUR", help="P50 (mediana): resultado mas tipico, la mitad acaban mejor y la mitad peor.")
        cr4.metric("Optimista", f"{p75[-1]:,.0f}EUR", help="P75: solo el 25% de escenarios son mejores que esto.")
        cr5.metric("Muy optimista", f"{p90[-1]:,.0f}EUR", help="P90: solo el 10% de escenarios acaban mejor.")

        total_aportado = cap + aport * 12 * ys
        prob_loss = np.sum(paths[:, -1] < total_aportado) / n_sims * 100

        # Resumen en lenguaje sencillo
        st.markdown(f"""<div class='info-tip'>
            <span class='tip-icon'>📊</span> <b>Resumen en lenguaje sencillo:</b><br>
            Empiezas con <b>{cap:,.0f}EUR</b>{'  y aportarias <b>' + f'{aport * 12 * ys:,.0f}EUR</b> en total' if aport > 0 else ''}.
            En <b>{ys} anos</b>, lo mas probable es que tu cartera valga alrededor de <b>{p50[-1]:,.0f}EUR</b>,
            con una ganancia estimada de <b>{p50[-1] - total_aportado:+,.0f}EUR</b>.
            {'Hay un <b>' + f'{prob_loss:.0f}%</b> de probabilidad de acabar con menos de lo aportado.' if prob_loss > 0 else '<b>En practicamente todos los escenarios acabarias con ganancia.</b>'}
        </div>""", unsafe_allow_html=True)

# ======================================================================
# HISTORIAL
# ======================================================================
elif pagina == "Historial":
    st.markdown("## Historial de Operaciones")
    st.markdown("""<div class='info-tip'>
        <span class='tip-icon'>📝</span> <b>Registro de todas tus operaciones.</b>
        Cada compra, venta, ingreso o retiro queda registrado aqui. Puedes filtrar por tipo
        y exportar a CSV para tu contabilidad.
    </div>""", unsafe_allow_html=True)

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
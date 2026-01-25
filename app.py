import streamlit as st
from supabase import create_client, Client
import pandas as pd
import yfinance as yf
from yahooquery import search
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from datetime import datetime, timedelta
import re
from duckduckgo_search import DDGS 

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial Ultra", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# ==============================================================================
# 🌑 DISEÑO DARK MODE (CSS PROFESIONAL)
# ==============================================================================
st.markdown("""
<style>
    /* IMPORTAR FUENTE */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    /* GENERAL */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, p, div, span, label {
        color: #E6E6E6 !important;
    }

    /* BARRA LATERAL */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* TARJETAS DE MÉTRICAS */
    div[data-testid="stMetric"] {
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.9rem !important;
        color: #8B949E !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.8rem !important;
        color: #FFFFFF !important;
        font-weight: 700;
    }

    /* BOTONES */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #30363D;
        background-color: #21262D;
        color: white;
    }
    .stButton > button:hover {
        border-color: #00CC96;
        color: #00CC96;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00CC96 0%, #007d5c 100%);
        border: none;
        color: black !important;
    }
    .stButton > button[kind="primary"] p {
        color: black !important;
    }

    /* --- CHAT DE NOTICIAS (DERECHA) --- */
    /* Contenedor con Scroll Independiente */
    .news-scroll-area {
        height: 70vh; /* Altura fija */
        overflow-y: auto; /* Scroll vertical */
        padding: 15px;
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        margin-top: 15px;
    }
    
    /* Scrollbar Personalizado */
    .news-scroll-area::-webkit-scrollbar { width: 8px; }
    .news-scroll-area::-webkit-scrollbar-track { background: #0E1117; border-radius: 4px;}
    .news-scroll-area::-webkit-scrollbar-thumb { background: #30363D; border-radius: 4px; }
    .news-scroll-area::-webkit-scrollbar-thumb:hover { background: #00CC96; }

    /* Tarjetas de Noticias */
    .news-card {
        background-color: #21262D;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 15px;
        border: 1px solid #30363D;
        transition: transform 0.2s;
        display: flex;
        flex-direction: column;
    }
    .news-card:hover {
        transform: scale(1.02);
        border-color: #00CC96;
    }
    .news-img {
        width: 100%;
        height: 120px;
        object-fit: cover;
        border-radius: 6px;
        margin-bottom: 10px;
        opacity: 0.9;
    }
    .news-source {
        font-size: 0.75rem;
        font-weight: 700;
        color: #8B949E;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    .news-title a {
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 0.95rem;
        text-decoration: none;
        line-height: 1.4;
        display: block;
    }
    .news-title a:hover {
        color: #00CC96 !important;
    }
    
    /* Gráficos Plotly Transparentes */
    .js-plotly-plot .plotly .main-svg {
        background-color: rgba(0,0,0,0) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- CONEXIÓN SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("❌ Error Crítico: Faltan los secretos de Supabase.")
    st.stop()

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- SESIÓN ---
if 'user' not in st.session_state: st.session_state['user'] = None
if 'show_news' not in st.session_state: st.session_state['show_news'] = True

# ==============================================================================
# 🔄 LOGIN Y REGISTRO
# ==============================================================================
query_params = st.query_params
if "code" in query_params and not st.session_state['user']:
    try:
        session = supabase.auth.exchange_code_for_session({"auth_code": query_params["code"]})
        st.session_state['user'] = session.user
        st.query_params.clear()
        st.rerun()
    except: pass

if not st.session_state['user']:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; color: #00CC96;'>🏦 Carterapro Ultra</h1>", unsafe_allow_html=True)
        st.caption("Gestión Patrimonial Institucional (Dark Edition)")
        
        tab_login, tab_signup = st.tabs(["🔐 Entrar", "📝 Registro"])
        
        with tab_login:
            if st.button("🇬 Entrar con Google", type="primary", use_container_width=True):
                try:
                    data = supabase.auth.sign_in_with_oauth({
                        "provider": "google",
                        "options": {"redirect_to": "https://carterapro.streamlit.app"}
                    })
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
                except Exception as e: st.error(f"Error: {e}")
            st.divider()
            email = st.text_input("Email", key="l_em")
            password = st.text_input("Contraseña", type="password", key="l_pa")
            if st.button("Entrar", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['user'] = res.user
                    st.rerun()
                except: st.error("Credenciales incorrectas.")

        with tab_signup:
            r_email = st.text_input("Email", key="r_em")
            r_pass = st.text_input("Contraseña", type="password", key="r_pa")
            if st.button("Crear Cuenta", type="primary", use_container_width=True):
                try:
                    res = supabase.auth.sign_up({"email": r_email, "password": r_pass})
                    st.success("Cuenta creada. Revisa tu correo.")
                except: st.error("Error al crear cuenta.")
    st.stop()

user = st.session_state['user']

# --- SIDEBAR ---
with st.sidebar:
    avatar_url = user.user_metadata.get('avatar_url', 'https://cdn-icons-png.flaticon.com/512/3135/3135715.png')
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px; padding: 10px; background: #21262D; border-radius: 10px; border: 1px solid #30363D;">
        <img src="{avatar_url}" style="width: 40px; height: 40px; border-radius: 50%; border: 2px solid #00CC96;">
        <div>
            <h3 style="margin: 0; font-size: 0.9rem; color: white !important;">{user.user_metadata.get('full_name', 'Inversor')}</h3>
            <span style="color: #00CC96; font-size: 0.7rem;">● En línea</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Toggle Noticias
    c_t1, c_t2 = st.columns([1, 4])
    with c_t1: st.write("📰")
    with c_t2: 
        if st.toggle("Panel de Noticias", value=st.session_state['show_news']):
            st.session_state['show_news'] = True
        else: st.session_state['show_news'] = False

    st.divider()
    pagina = st.radio("MENÚ", [
        "📊 Dashboard & Alpha", 
        "💰 Liquidez (Cash)",
        "➕ Gestión de Inversiones",
        "🤖 Asesor de Riesgos", 
        "🔮 Monte Carlo", 
        "⚖️ Rebalanceo"
    ])
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Cerrar Sesión", use_container_width=True):
        supabase.auth.sign_out(); st.session_state['user'] = None; st.rerun()

# --- FUNCIONES MATEMÁTICAS ---
def safe_metric_calc(series):
    clean = series.fillna(method='ffill').dropna()
    if len(clean) < 5: return 0, 0, 0, 0
    returns = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty: return 0, 0, 0, 0
    try: total_ret = (clean.iloc[-1] / clean.iloc[0]) - 1
    except: total_ret = 0
    vol = returns.std() * np.sqrt(252)
    if np.isnan(vol): vol = 0
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()
    if np.isnan(max_dd): max_dd = 0
    rf = 0.03
    mean_ret = returns.mean() * 252
    sharpe = (mean_ret - rf) / vol if vol > 0.01 else 0
    return total_ret, vol, max_dd, sharpe

def sanitize_input(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^\w\s\-\.]', '', text).strip().upper()

# --- CACHE & DATOS ---
@st.cache_data(ttl=60)
def get_assets_db(uid):
    try: return pd.DataFrame(supabase.table('assets').select("*").eq('user_id', uid).execute().data)
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def get_liquidity_db(uid):
    try:
        response = supabase.table('liquidity').select("*").eq('user_id', uid).execute()
        data = response.data
        if data: return pd.DataFrame(data)
        else:
            new_data = {"user_id": uid, "name": "Principal", "amount": 0.0}
            supabase.table('liquidity').insert(new_data).execute()
            return pd.DataFrame([new_data]) 
    except: return pd.DataFrame(columns=['id', 'user_id', 'name', 'amount', 'created_at'])

@st.cache_data(ttl=300)
def get_market_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        tickers_api = list(set(tickers + ['SPY', 'GLD'])) 
        data = yf.download(tickers_api, period="1y", progress=False)['Close']
        data.index = data.index.tz_localize(None)
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=600) 
def get_global_news(tickers, time_filter='d'):
    results = []
    if tickers:
        main_ticker = tickers[0]
        queries_to_try = [f"{main_ticker} noticias finanzas", "Noticias mercado valores economía"]
    else:
        queries_to_try = ["Noticias economía inversiones españa"]
    
    try:
        with DDGS() as ddgs:
            for q in queries_to_try:
                ddg_news = list(ddgs.news(q, region="es-es", safesearch="off", timelimit=time_filter, max_results=10))
                if ddg_news:
                    for n in ddg_news:
                        results.append({
                            'title': n.get('title'),
                            'source': n.get('source'),
                            'date': n.get('date'), 
                            'url': n.get('url'),
                            'image': n.get('image', None)
                        })
                    break 
    except: pass
    return results

def clear_cache(): st.cache_data.clear()

# --- ESCRITURA DB ---
def add_asset_db(t, n, s, p, pl):
    supabase.table('assets').insert({"user_id": user.id, "ticker": t, "nombre": n, "shares": s, "avg_price": p, "platform": pl}).execute()
    clear_cache()

def update_asset_db(asset_id, shares, avg_price):
    supabase.table('assets').update({"shares": shares, "avg_price": avg_price}).eq('id', asset_id).execute()
    clear_cache()

def delete_asset_db(id_del):
    supabase.table('assets').delete().eq('id', id_del).execute()
    clear_cache()

def update_liquidity_balance(liq_id, new_amount):
    supabase.table('liquidity').update({"amount": new_amount}).eq('id', liq_id).execute()
    clear_cache()

# --- CARGA DATOS ---
df_assets = get_assets_db(user.id)
df_cash = get_liquidity_db(user.id)

if not df_cash.empty:
    total_liquidez = df_cash.iloc[0]['amount']
    cash_id = df_cash.iloc[0]['id']
else: total_liquidez = 0.0; cash_id = None

df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()
my_tickers = []

if not df_assets.empty:
    my_tickers = df_assets['ticker'].unique().tolist()
    data_raw = get_market_data(my_tickers)
    
    if not data_raw.empty:
        if 'SPY' in data_raw.columns:
            benchmark_data = data_raw['SPY']
            history = data_raw.drop(columns=['SPY', 'GLD'], errors='ignore')
        else: history = data_raw
        history_data = history
        
        prices_dict = {}; rsi_dict = {}; metrics_dict = {}
        for t in my_tickers:
            if t in history.columns:
                s = history[t]
                prices_dict[t] = s.iloc[-1]
                if len(s)>15:
                    delta = s.diff(); up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
                    rs = up.ewm(com=13).mean() / down.ewm(com=13).mean()
                    rsi_dict[t] = 100 - (100 / (1 + rs)).iloc[-1]
                else: rsi_dict[t] = 50
                ret, vol, dd, sha = safe_metric_calc(s)
                metrics_dict[t] = {'vol': vol, 'dd': dd, 'sharpe': sha}
            else:
                prices_dict[t]=0; rsi_dict[t]=50; metrics_dict[t]={'vol':0,'dd':0,'sharpe':0}

        df_assets['Precio Actual'] = df_assets['ticker'].map(prices_dict)
        df_assets['RSI'] = df_assets['ticker'].map(rsi_dict)
        df_assets['Volatilidad'] = df_assets['ticker'].apply(lambda x: metrics_dict[x]['vol']*100)
        df_assets['Max Drawdown'] = df_assets['ticker'].apply(lambda x: metrics_dict[x]['dd']*100)
        df_assets['Sharpe'] = df_assets['ticker'].apply(lambda x: metrics_dict[x]['sharpe'])
        
        df_assets['Valor Acciones'] = df_assets['shares'] * df_assets['Precio Actual']
        df_assets['Dinero Invertido'] = df_assets['shares'] * df_assets['avg_price']
        df_assets['Ganancia'] = df_assets['Valor Acciones'] - df_assets['Dinero Invertido']
        df_assets['Rentabilidad'] = df_assets.apply(lambda r: (r['Ganancia']/r['Dinero Invertido']*100) if r['Dinero Invertido']>0 else 0, axis=1)
        total_inv = df_assets['Valor Acciones'].sum()
        df_assets['Peso %'] = df_assets.apply(lambda r: (r['Valor Acciones']/total_inv*100) if total_inv>0 else 0, axis=1)
        df_final = df_assets.rename(columns={'nombre': 'Nombre'})

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0
patrimonio_total = total_inversiones + total_liquidez

# ==============================================================================
# LAYOUT DINÁMICO (APP + CHAT)
# ==============================================================================
if st.session_state['show_news']:
    col_main, col_news = st.columns([3.5, 1.3])
else:
    col_main = st.container()
    col_news = st.container()

# --- APP PRINCIPAL (IZQUIERDA) ---
with col_main:
    if pagina == "📊 Dashboard & Alpha":
        st.title("📊 Visión Global")
        col_kpi, col_date = st.columns([3, 1])
        with col_date: start_date = st.date_input("📅 Rendimiento desde:", value=datetime.now()-timedelta(days=180))
        with col_kpi:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Patrimonio", f"{patrimonio_total:,.2f} €")
            if total_inversiones>0:
                gan = df_final['Ganancia'].sum()
                c2.metric("📈 P&L", f"{gan:+,.2f} €", f"{(gan/df_final['Dinero Invertido'].sum()*100):.2f}%")
            else: c2.metric("📈 P&L", "0€")
            c3.metric("💧 Liquidez", f"{total_liquidez:,.2f} €")
            c4.metric("⚡ Volatilidad", f"{df_final['Volatilidad'].mean() if not df_final.empty else 0:.1f}%")
        
        st.divider()
        c_ch, c_pi = st.columns([2,1])
        with c_ch:
            st.subheader("Rendimiento")
            if not history_data.empty:
                my_norm = history_data.sum(axis=1); my_norm = my_norm/my_norm.iloc[0]*100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=my_norm.index, y=my_norm, name="Tú", line=dict(color='#00CC96')))
                if not benchmark_data.empty:
                    bn = benchmark_data/benchmark_data.iloc[0]*100
                    fig.add_trace(go.Scatter(x=bn.index, y=bn, name="S&P500", line=dict(color='gray', dash='dot')))
                fig.update_layout(template="plotly_dark", height=300, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("Faltan datos.")
        
        with c_pi:
            st.subheader("Cartera Total")
            if patrimonio_total > 0:
                vals = [total_liquidez] + df_final['Valor Acciones'].tolist() if not df_final.empty else [total_liquidez]
                nams = ['💧 Liquidez'] + df_final['Nombre'].tolist() if not df_final.empty else ['Liquidez']
                fig = px.pie(values=vals, names=nams, hole=0.6, color_discrete_sequence=px.colors.qualitative.Prism)
                fig.update_layout(template="plotly_dark", height=300, showlegend=False, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        c_tree, c_bar = st.columns([1.5, 1.5])
        with c_tree:
            st.subheader("🗺️ Mapa de Calor (P&L)")
            if not df_final.empty:
                fig_tree = px.treemap(df_final, path=['Nombre'], values='Valor Acciones', color='Rentabilidad', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
                fig_tree.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tree, use_container_width=True)
            else: st.info("Sin inversiones.")
        with c_bar:
            st.subheader("🏆 Ranking (€)")
            if not df_final.empty:
                df_sorted = df_final.sort_values('Ganancia', ascending=False)
                fig_bar = px.bar(df_sorted, x='Ganancia', y='Nombre', orientation='h', color='Ganancia', color_continuous_scale='RdYlGn', text_auto='.2s')
                fig_bar.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
            else: st.info("Sin inversiones.")

    elif pagina == "💰 Liquidez (Cash)":
        st.title("💰 Mi Liquidez")
        st.markdown(f"""
        <div style="text-align:center; padding: 40px; background-color: #21262D; border-radius: 15px; margin-bottom: 30px; border: 1px solid #30363D;">
            <h3 style="color:#8B949E; margin:0; font-size: 1.1rem; text-transform: uppercase;">Saldo Disponible</h3>
            <h1 style="font-size: 4rem; color:#FFFFFF; margin: 10px 0;">{total_liquidez:,.2f} €</h1>
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("#### 📥 Ingresar Dinero")
                a = st.number_input("Cantidad (€)", 0.0, step=50.0, key="in")
                if st.button("Confirmar Ingreso", type="primary", use_container_width=True) and a>0:
                    cid = get_liquidity_db(user.id).iloc[0]['id']
                    update_liquidity_balance(int(cid), total_liquidez + a)
                    st.success("Hecho"); time.sleep(0.5); st.rerun()
        with c2:
            with st.container(border=True):
                st.markdown("#### 📤 Retirar Dinero")
                b = st.number_input("Cantidad (€)", 0.0, step=50.0, key="out")
                if st.button("Confirmar Retirada", use_container_width=True) and b>0:
                    if b > total_liquidez: st.error("Sin saldo")
                    else:
                        cid = get_liquidity_db(user.id).iloc[0]['id']
                        update_liquidity_balance(int(cid), total_liquidez - b)
                        st.success("Hecho"); time.sleep(0.5); st.rerun()

    elif pagina == "➕ Gestión de Inversiones":
        st.title("➕ Inversiones")
        t1, t2, t3 = st.tabs(["🆕 Añadir", "💰 Operar", "✏️ Editar"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                q = st.text_input("Buscar:", placeholder="ISIN o Nombre...")
                if st.button("🔍 Buscar") and q:
                    r = search(sanitize_input(q))
                    if 'quotes' in r: st.session_state['s'] = r['quotes']
                if 's' in st.session_state:
                    opts = {f"{x['symbol']} | {x.get('longname','')}" : x for x in st.session_state['s']}
                    sel = st.selectbox("Elegir:", list(opts.keys()))
                    st.session_state['sel_add'] = opts[sel]
            with c2:
                if 'sel_add' in st.session_state:
                    a = st.session_state['sel_add']; t = a['symbol']
                    try:
                        inf = yf.Ticker(t).history(period='1d')
                        if not inf.empty:
                            cp = inf['Close'].iloc[-1]
                            st.metric("Precio", f"{cp:.2f} €")
                            with st.form("n"):
                                i = st.number_input("Invertido (€)", 0.0)
                                v = st.number_input("Valor (€)", 0.0)
                                pl = st.selectbox("Plataforma", ["MyInvestor", "XTB", "TR", "Degiro"])
                                if st.form_submit_button("Guardar") and v>0:
                                    add_asset_db(t, a.get('longname', t), v/cp, i/(v/cp), pl)
                                    st.success("Ok"); time.sleep(0.5); st.rerun()
                    except: st.error("Error precio.")
        with t2:
            if df_final.empty: st.warning("Sin activos.")
            else:
                c1, c2 = st.columns([1,2])
                with c1:
                    nom = st.selectbox("Activo", df_final['Nombre'].unique())
                    r = df_final[df_final['Nombre']==nom].iloc[0]
                    st.info(f"Tienes: **{r['shares']:.4f}** accs")
                with c2:
                    lp = r['Precio Actual']
                    st.metric("Precio", f"{lp:.2f}€")
                    op = st.radio("Acción", ["Compra", "Venta"], horizontal=True)
                    amt = st.number_input("Importe (€)", 0.0, step=50.0)
                    st.caption(f"Liquidez: {total_liquidez:,.2f}€")
                    if amt > 0:
                        sh = amt / lp
                        if "Compra" in op:
                            if amt > total_liquidez: st.error("Sin liquidez")
                            else:
                                if st.button("Confirmar Compra", type="primary"):
                                    navg = ((r['shares']*r['avg_price'])+amt)/(r['shares']+sh)
                                    update_asset_db(int(r['id']), r['shares']+sh, navg)
                                    if cash_id: update_liquidity_balance(int(cash_id), total_liquidez - amt)
                                    st.rerun()
                        else:
                            if st.button("Confirmar Venta"):
                                nsh = r['shares'] - sh
                                if nsh < 0.001: delete_asset_db(int(r['id']))
                                else: update_asset_db(int(r['id']), nsh, r['avg_price'])
                                if cash_id: update_liquidity_balance(int(cash_id), total_liquidez + amt)
                                st.rerun()
        with t3:
            if not df_final.empty:
                ed = st.selectbox("Editar", df_final['Nombre'], key='ed')
                re = df_final[df_final['Nombre']==ed].iloc[0]
                c1,c2,c3 = st.columns(3)
                ns = c1.number_input("Accs", value=float(re['shares']), format="%.4f")
                na = c2.number_input("Media", value=float(re['avg_price']), format="%.4f")
                if c3.button("Actualizar"): update_asset_db(int(re['id']), ns, na); st.rerun()
                if st.button("Borrar"): delete_asset_db(int(re['id'])); st.rerun()

    elif pagina == "🤖 Asesor de Riesgos":
        st.title("🤖 IA & Riesgos")
        if not df_final.empty and not history_data.empty:
            c1, c2 = st.columns(2)
            with c1:
                try:
                    ms = history_data.sum(axis=1).pct_change().dropna()
                    if not benchmark_data.empty:
                        bs = benchmark_data.pct_change().dropna()
                        ix = ms.index.intersection(bs.index)
                        if len(ix)>10:
                            beta = ms.loc[ix].cov(bs.loc[ix])/bs.loc[ix].var()
                            st.metric("Beta", f"{beta:.2f}")
                            if beta>1.2: st.error("Cartera Agresiva")
                            elif beta<0.8: st.success("Cartera Defensiva")
                            else: st.info("Cartera Equilibrada")
                except: pass
            with c2:
                st.write("Correlaciones")
                fig = plt.figure(figsize=(5,4))
                fig.patch.set_facecolor('#0E1117')
                ax = sns.heatmap(history_data.corr(), annot=True, cmap='coolwarm', cbar_kws={'label': 'Correlación'})
                for t in ax.texts: t.set_color('white')
                ax.tick_params(colors='white')
                st.pyplot(fig)
        else: st.warning("Faltan datos.")

    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Futuro")
        if df_final.empty: st.stop()
        ys = st.slider("Años", 1, 30, 10)
        run = st.button("Simular")
        if run:
            try:
                mu = 0.07; sigma = 0.15
                if not history_data.empty:
                    d = history_data.sum(axis=1).pct_change().dropna()
                    mu = d.mean()*252; sigma = d.std()*np.sqrt(252)
                paths = []
                for _ in range(30):
                    p = [total_inversiones]
                    for _ in range(int(ys*252)):
                        p.append(p[-1] * np.exp((mu-0.5*sigma**2)/252 + sigma*np.sqrt(1/252)*np.random.normal(0,1)))
                    paths.append(p)
                x = np.linspace(0, ys, len(paths[0]))
                fig = go.Figure()
                for p in paths: fig.add_trace(go.Scatter(x=x, y=p, mode='lines', line=dict(color='rgba(0,100,200,0.1)'), showlegend=False))
                fig.add_trace(go.Scatter(x=x, y=np.mean(paths, axis=0), mode='lines', line=dict(color='#00CC96', width=3)))
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            except: st.error("Error simulación")

    elif pagina == "⚖️ Rebalanceo":
        st.title("⚖️ Rebalanceo")
        if df_final.empty: st.stop()
        c1,c2 = st.columns([1,2])
        ws = {}; tot = 0
        with c1:
            for i,r in df_final.iterrows():
                w = st.number_input(f"{r['Nombre']} %", 0, 100, int(r['Peso %']), key=i)
                ws[r['Nombre']] = w; tot += w
            st.metric("Total", f"{tot}%")
        with c2:
            if tot==100 and st.button("Calcular"):
                pat = df_final['Valor Acciones'].sum()
                mc = max([r['Valor Acciones']/(ws[r['Nombre']]/100) for i,r in df_final.iterrows() if ws[r['Nombre']]>0]+[pat])
                dat = [{'Activo':r['Nombre'], 'Comprar':max(0, (mc*ws[r['Nombre']]/100)-r['Valor Acciones'])} for i,r in df_final.iterrows()]
                fig = px.bar(pd.DataFrame(dat), x='Activo', y='Comprar')
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

# --- BOT DE NOTICIAS DERECHA (ESTILO TWITCH FIXED) ---
if st.session_state['show_news']:
    with col_news:
        # 1. CABECERA FIJA (TÍTULO Y BOTÓN)
        c_head, c_btn = st.columns([3, 1])
        with c_head:
            st.markdown("<h3 style='margin:0; padding:0;'>🤖 Bot News</h3>", unsafe_allow_html=True)
        with c_btn:
            if st.button("🔄", help="Actualizar ahora", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        # 2. FILTROS (FIJOS)
        tf_map = {'Hoy': 'd', 'Semana': 'w', 'Mes': 'm'}
        sel_tf = st.pills("📅 Filtro:", list(tf_map.keys()), default="Hoy", label_visibility="collapsed")
        time_code = tf_map[sel_tf]
        
        # 3. CONTENIDO CON SCROLL
        news_feed = get_global_news(my_tickers, time_code)
        
        html_content = ""
        if news_feed:
            for n in news_feed:
                img_tag = f"<img src='{n['image']}' class='news-img'/>" if n.get('image') else ""
                html_content += f"""
                <div class="news-card">
                    {img_tag}
                    <div class="news-source">{n.get('source', 'Internet')} • {n.get('date', '')}</div>
                    <div class="news-title">
                        <a href="{n.get('url', '#')}" target="_blank">{n.get('title', 'Sin título')}</a>
                    </div>
                </div>
                """
        else:
            html_content = """
            <div style='color:#888; text-align:center; padding:40px;'>
                <div style='font-size: 40px;'>💤</div>
                <p>No hay noticias recientes con este filtro.</p>
                <small>Intenta cambiar a 'Semana' o dale a 🔄</small>
            </div>
            """

        st.markdown(f"""
        <div class='news-scroll-area'>
            {html_content}
        </div>
        """, unsafe_allow_html=True)
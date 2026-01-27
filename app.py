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
from datetime import datetime, timedelta
import re
from duckduckgo_search import DDGS 
import openai 

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial Pro", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# ==============================================================================
# 🌑 DISEÑO DARK MODE (CSS v19)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp { background-color: #0B0E11; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #12151A; border-right: 1px solid #2D333B; }

    /* KPIs Flotantes */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1C2128, #16191D);
        border: 1px solid #2D333B; border-radius: 12px;
        padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; color: white !important; font-weight: 700; }
    div[data-testid="stMetricDelta"] svg { display: none; } 
    
    /* Botones */
    .stButton>button { border-radius: 8px; font-weight: 600; border: 1px solid #30363D; background-color: #21262D; color: white; transition: 0.3s; height: 45px; }
    .stButton>button:hover { border-color: #00CC96; color: #00CC96; }
    .stButton>button[kind="primary"] { background: linear-gradient(135deg, #00CC96 0%, #007d5c 100%); border: none; color: #000 !important; font-weight: 800; }

    /* Chat Noticias */
    .news-scroll-area { height: 68vh; overflow-y: auto; padding: 10px; background-color: #161B22; border: 1px solid #30363D; border-radius: 12px; margin-top: 15px; }
    .news-card { background-color: #21262D; border-radius: 8px; padding: 12px; margin-bottom: 12px; border: 1px solid #30363D; display: flex; flex-direction: column; }
    .news-img { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; margin-bottom: 8px; opacity: 0.8; }
    .news-title a { color: #FFFFFF !important; text-decoration: none; font-weight: 600; font-size: 0.9rem; }
    .news-source { font-size: 0.7rem; color: #00CC96; text-transform: uppercase; margin-bottom: 4px; }

    /* Gráficos */
    .js-plotly-plot .plotly .main-svg { background-color: rgba(0,0,0,0) !important; }
</style>
""", unsafe_allow_html=True)

# --- CONEXIONES ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    HAS_GROQ = "GROQ_API_KEY" in st.secrets
except:
    st.error("❌ Error Crítico: Faltan secretos."); st.stop()

@st.cache_resource
def init_supabase(): return create_client(SUPABASE_URL, SUPABASE_KEY)
supabase = init_supabase()

# --- SESIÓN ---
if 'user' not in st.session_state: st.session_state['user'] = None
if 'show_news' not in st.session_state: st.session_state['show_news'] = True
if 'messages' not in st.session_state: st.session_state['messages'] = []

# ==============================================================================
# 🔄 LOGIN (ESTABILIZADO)
# ==============================================================================
query_params = st.query_params
if "code" in query_params and not st.session_state['user']:
    try:
        session = supabase.auth.exchange_code_for_session({"auth_code": query_params["code"]})
        st.session_state['user'] = session.user
        st.query_params.clear(); st.rerun()
    except: pass

if not st.session_state['user']:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown("<br><h1 style='text-align: center; color: #00CC96;'>🏦 Carterapro Ultra</h1>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔐 Entrar", "📝 Registro"])
        with tab1:
            if st.button("🇬 Google", type="primary", use_container_width=True):
                try:
                    data = supabase.auth.sign_in_with_oauth({"provider": "google", "options": {"redirect_to": "https://carterapro.streamlit.app"}})
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
                except Exception as e: st.error(f"Error: {e}")
            st.divider()
            with st.form("login"):
                em = st.text_input("Email")
                pa = st.text_input("Pass", type="password")
                if st.form_submit_button("Entrar", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": em, "password": pa})
                        st.session_state['user'] = res.user
                        st.rerun()
                    except: st.error("Credenciales incorrectas.")
        with tab2:
            with st.form("signup"):
                em2 = st.text_input("Email")
                pa2 = st.text_input("Pass", type="password")
                if st.form_submit_button("Crear Cuenta", use_container_width=True):
                    try: supabase.auth.sign_up({"email": em2, "password": pa2}); st.success("Cuenta creada.")
                    except: st.error("Error.")
    st.stop()

user = st.session_state['user']

# ==============================================================================
# 🧠 MOTOR DE DATOS ROBUSTO
# ==============================================================================

def sanitize_input(text): return re.sub(r'[^\w\s\-\.]', '', str(text)).strip().upper()

def safe_metric_calc(series):
    """Calcula métricas evitando errores de array vacío"""
    clean = series.dropna()
    if len(clean) < 5: return 0, 0, 0, 0
    returns = clean.pct_change().dropna()
    if returns.empty: return 0, 0, 0, 0
    try: total_ret = (clean.iloc[-1] / clean.iloc[0]) - 1
    except: total_ret = 0
    vol = returns.std() * np.sqrt(252)
    mean_ret = returns.mean() * 252
    sharpe = (mean_ret - 0.03) / vol if vol > 0.001 else 0
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    return total_ret, vol, max_dd, sharpe

@st.cache_data(ttl=60)
def get_user_data(uid):
    assets = pd.DataFrame(supabase.table('assets').select("*").eq('user_id', uid).execute().data)
    liq_res = supabase.table('liquidity').select("*").eq('user_id', uid).execute().data
    if not liq_res:
        supabase.table('liquidity').insert({"user_id": uid, "name": "Principal", "amount": 0.0}).execute()
        liquidity = 0.0; liq_id = 0
    else:
        liquidity = liq_res[0]['amount']; liq_id = liq_res[0]['id']
    return assets, liquidity, liq_id

# Precio Actual (Sin Cache al pulsar refresh)
def get_real_time_prices(tickers):
    if not tickers: return {}
    prices = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            if 'last_price' in info and info['last_price']:
                prices[t] = info['last_price']
            else:
                hist = yf.Ticker(t).history(period='2d')
                prices[t] = hist['Close'].iloc[-1] if not hist.empty else 0.0
        except: prices[t] = 0.0
    return prices

@st.cache_data(ttl=300)
def get_historical_data_robust(tickers):
    """Descarga histórico asegurando compatibilidad de fechas"""
    if not tickers: return pd.DataFrame()
    unique = list(set([str(t).strip().upper() for t in tickers] + ['SPY']))
    try:
        # Descarga
        data = yf.download(unique, period="2y", interval="1d", progress=False)['Adj Close']
        
        # Convertir a DF si es Serie
        if isinstance(data, pd.Series):
            data = data.to_frame()
            if data.columns[0] == 0: data.columns = unique
            
        if not data.empty:
            # 🔥 QUITAR TIMEZONE PARA EVITAR EL BUG DEL DASHBOARD 🔥
            data.index = data.index.tz_localize(None)
            data = data.fillna(method='ffill').fillna(method='bfill')
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=900)
def get_global_news(tickers, time_filter='d'):
    results = []
    query = f"{tickers[0]} bolsa" if tickers else "Mercado financiero"
    try:
        with DDGS() as ddgs:
            raw = list(ddgs.news(query, region="es-es", safesearch="off", timelimit=time_filter, max_results=8))
            for n in raw:
                results.append({
                    'title': n.get('title'), 'source': n.get('source'), 'date': n.get('date'),
                    'url': n.get('url'), 'image': n.get('image', None)
                })
    except: pass
    return results

def clear_cache(): st.cache_data.clear()

# --- DB WRAPPERS ---
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

# --- CARGA ---
df_assets, total_liquidez, cash_id = get_user_data(user.id)
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()
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
        else: history_data = history_raw

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0.0
patrimonio_total = total_inversiones + total_liquidez

# --- SIDEBAR ---
with st.sidebar:
    avatar = user.user_metadata.get('avatar_url', '')
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; padding:10px; background:#21262D; border-radius:8px; border:1px solid #30363D;'>
        <img src='{avatar}' style='width:35px; border-radius:50%; border:2px solid #00CC96;'>
        <div style='line-height:1.2'><b style='color:white'>{user.user_metadata.get('full_name','Inversor')}</b><br><span style='font-size:0.7em; color:#00CC96'>● Online</span></div>
    </div><br>""", unsafe_allow_html=True)
    
    if st.button("🔄 Actualizar Datos", use_container_width=True, type="primary"):
        clear_cache()
        st.rerun()
    
    st.divider()
    c1, c2 = st.columns([1,4])
    with c1: st.write("📰")
    with c2: st.session_state['show_news'] = st.toggle("Noticias", value=st.session_state['show_news'])
    
    pagina = st.radio("MENÚ", ["📊 Dashboard & Alpha", "💰 Liquidez (Cash)", "➕ Inversiones", "💬 Asesor AI", "🔮 Monte Carlo", "⚖️ Rebalanceo"])
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Cerrar Sesión", use_container_width=True): supabase.auth.sign_out(); st.session_state['user']=None; st.rerun()

# --- MAIN ---
if st.session_state['show_news']: col_main, col_news = st.columns([3.5, 1.3])
else: col_main = st.container(); col_news = st.container()

with col_main:
    
    # ------------------------------------------------------------------
    # 📊 DASHBOARD (ARREGLADO)
    # ------------------------------------------------------------------
    if pagina == "📊 Dashboard & Alpha":
        st.title("📊 Control de Mando Integral")
        
        c_date, c_score = st.columns([3, 1])
        with c_date: 
            # Fecha por defecto 1 año atrás
            d_def = datetime.now() - timedelta(days=365)
            start_date = st.date_input("📅 Rango de Análisis:", value=d_def)
        
        # Cálculos Métricas
        vol_anual = 0; sharpe_ratio = 0; max_drawdown = 0; beta = 1.0; health = 50
        
        if not history_data.empty:
            # 1. Asegurar fecha sin zona horaria
            dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
            
            # 2. Filtrado seguro
            try:
                hist_filt = history_data[history_data.index >= dt_start].copy()
            except:
                hist_filt = history_data # Fallback si falla filtro
            
            if not hist_filt.empty:
                # Retorno diario promedio de activos
                daily_returns = hist_filt.pct_change().mean(axis=1).dropna()
                
                if not daily_returns.empty:
                    # Métricas
                    _, vol_anual, max_drawdown_dec, sharpe_ratio = safe_metric_calc(daily_returns + 1)
                    vol_anual *= 100; max_drawdown = max_drawdown_dec * 100
                    
                    # Score Salud
                    health = 50
                    if sharpe_ratio > 1: health += 15
                    if max_drawdown > -20: health += 15
                    if vol_anual < 15: health += 10
                    if len(df_final) >= 3: health += 10
                    
                    # Beta
                    if not benchmark_data.empty:
                        try:
                            bench_ret = benchmark_data[benchmark_data.index >= dt_start].pct_change().dropna()
                            common = daily_returns.index.intersection(bench_ret.index)
                            if len(common) > 10:
                                cov = daily_returns.loc[common].cov(bench_ret.loc[common])
                                var = bench_ret.loc[common].var()
                                beta = cov / var if var != 0 else 1.0
                        except: pass

        with c_score:
            st.metric("🛡️ Salud Cartera", f"{health}/100", help="Puntuación algorítmica basada en diversificación y riesgo.")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💰 Patrimonio Neto", f"{patrimonio_total:,.2f} €", help="Total Activos + Liquidez")
        
        if total_inversiones > 0:
            pnl = df_final['Ganancia'].sum()
            pct = (pnl / df_final['Dinero Invertido'].sum()) * 100 if df_final['Dinero Invertido'].sum()>0 else 0
            k2.metric("📈 P&L Latente", f"{pnl:+,.2f} €", f"{pct:+.2f}%", delta_color="normal" if pnl>=0 else "inverse", help="Ganancia no realizada")
        else: k2.metric("📈 P&L Latente", "0.00 €")
        
        k3.metric("📉 Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse", help="La peor caída que has sufrido en este periodo")
        k4.metric("⚖️ Ratio Sharpe", f"{sharpe_ratio:.2f}", help=">1: Bueno. Cuánto retorno obtienes por cada unidad de riesgo.")

        st.divider()

        # GRÁFICO RENDIMIENTO
        c_chart, c_donut = st.columns([2, 1.2])
        with c_chart:
            st.subheader("🏁 Rendimiento (Base 100)")
            if not history_data.empty and 'hist_filt' in locals() and not hist_filt.empty:
                port_ret = hist_filt.pct_change().mean(axis=1).fillna(0)
                port_cum = (1 + port_ret).cumprod() * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Tu Cartera", line=dict(color='#00CC96', width=2)))
                
                if not benchmark_data.empty:
                    try:
                        bench_filt = benchmark_data[benchmark_data.index >= dt_start].copy()
                        bench_cum = (1 + bench_filt.pct_change().fillna(0)).cumprod() * 100
                        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500", line=dict(color='gray', dash='dot')))
                    except: pass
                
                fig.update_layout(template="plotly_dark", height=320, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', yaxis_title="Base 100")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay datos históricos suficientes para graficar. Dale a 'Actualizar Datos'.")

        with c_donut:
            st.subheader("🍰 Asset Allocation")
            if patrimonio_total > 0:
                labels = ['Liquidez'] + df_final['Nombre'].tolist() if not df_final.empty else ['Liquidez']
                values = [total_liquidez] + df_final['Valor Acciones'].tolist() if not df_final.empty else [total_liquidez]
                fig_pie = px.pie(names=labels, values=values, hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(template="plotly_dark", height=320, showlegend=True, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.info("Cartera vacía.")

    # ------------------------------------------------------------------
    # 💰 LIQUIDEZ
    # ------------------------------------------------------------------
    elif pagina == "💰 Liquidez (Cash)":
        st.title("💰 Gestión de Liquidez")
        st.markdown(f"""<div style="text-align:center; padding: 40px; background-color: #21262D; border-radius: 15px; margin-bottom: 30px; border: 1px solid #30363D;"><h1 style="font-size: 4.5rem; color:#FFFFFF; margin: 0;">{total_liquidez:,.2f} €</h1><p style="color:#8B949E;">SALDO DISPONIBLE</p></div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            a = st.number_input("Ingreso (€)", 0.0, step=50.0, key="in")
            if st.button("Ingresar", type="primary") and a>0:
                update_liquidity_balance(int(cash_id), total_liquidez+a); st.toast("✅"); time.sleep(0.5); st.rerun()
        with c2:
            b = st.number_input("Retiro (€)", 0.0, step=50.0, key="out")
            if st.button("Retirar") and b>0:
                if b>total_liquidez: st.error("Sin saldo")
                else: update_liquidity_balance(int(cash_id), total_liquidez-b); st.toast("✅"); time.sleep(0.5); st.rerun()

    # ------------------------------------------------------------------
    # ➕ INVERSIONES
    # ------------------------------------------------------------------
    elif pagina == "➕ Inversiones":
        st.title("➕ Gestión de Activos")
        t1, t2, t3 = st.tabs(["🆕 Añadir", "💰 Operar", "✏️ Editar"])
        
        with t1:
            c1, c2 = st.columns([1, 1])
            with c1:
                q = st.text_input("🔍 Buscar (ISIN/Ticker/Nombre):")
                if st.button("Buscar") and q:
                    try:
                        res = search(sanitize_input(q))
                        if 'quotes' in res: st.session_state['s'] = res['quotes']
                    except: st.error("Error búsqueda")
                if 's' in st.session_state and st.session_state['s']:
                    opts = {f"{x['symbol']} | {x.get('shortname','')}" : x for x in st.session_state['s']}
                    if opts:
                        sel = st.selectbox("Elegir:", list(opts.keys()))
                        if sel in opts: st.session_state['sel_add'] = opts[sel]
            with c2:
                if 'sel_add' in st.session_state:
                    tk = st.session_state['sel_add']['symbol']
                    try:
                        inf = yf.Ticker(tk).fast_info
                        p = inf['last_price']
                        mon = inf['currency']
                        if p:
                            st.metric("Precio", f"{p:.2f} {mon}")
                            if mon != 'EUR': st.warning(f"⚠️ Activo en {mon}")
                            with st.form("new"):
                                i = st.number_input("Invertido (EUR)", 0.0)
                                v = st.number_input("Valor Actual (EUR)", 0.0)
                                pl = st.selectbox("Broker", ["MyInvestor", "XTB", "TR", "Degiro"])
                                if st.form_submit_button("Guardar") and v>0:
                                    sh = v/p
                                    av = i/sh if sh>0 else 0
                                    add_asset_db(tk, st.session_state['sel_add'].get('shortname',tk), sh, av, pl)
                                    st.success("Guardado"); time.sleep(1); st.rerun()
                    except: st.error("Error al obtener precio")

        with t2:
            if df_final.empty: st.info("Añade activos.")
            else:
                nom = st.selectbox("Activo", df_final['Nombre'].unique())
                r = df_final[df_final['Nombre']==nom].iloc[0]
                st.info(f"Tienes {r['shares']:.4f} accs")
                op = st.radio("Acción", ["Compra","Venta"], horizontal=True)
                m = st.number_input("Euros", 0.0)
                if m>0:
                    sh = m/r['Precio Actual']
                    if "Compra" in op:
                        if st.button("Confirmar") and m<=total_liquidez:
                            navg = ((r['shares']*r['avg_price'])+m)/(r['shares']+sh)
                            update_asset_db(int(r['id']), r['shares']+sh, navg)
                            update_liquidity_balance(int(cash_id), total_liquidez-m)
                            st.rerun()
                    else:
                        if st.button("Confirmar"):
                            ns = r['shares']-sh
                            if ns<0.001: delete_asset_db(int(r['id']))
                            else: update_asset_db(int(r['id']), ns, r['avg_price'])
                            update_liquidity_balance(int(cash_id), total_liquidez+m)
                            st.rerun()

        with t3:
            if not df_final.empty:
                e = st.selectbox("Editar", df_final['Nombre'], key='edd')
                er = df_final[df_final['Nombre']==e].iloc[0]
                nsh = st.number_input("Accs", value=float(er['shares']))
                nav = st.number_input("Media", value=float(er['avg_price']))
                if st.button("Guardar"): update_asset_db(int(er['id']), nsh, nav); st.rerun()
                if st.button("Borrar"): delete_asset_db(int(er['id'])); st.rerun()

    # ------------------------------------------------------------------
    # 💬 ASESOR AI
    # ------------------------------------------------------------------
    elif pagina == "💬 Asesor AI":
        st.title("💬 Asesor IA")
        ctx = f"Liquidez: {total_liquidez}€. "
        if not df_final.empty:
            for i,r in df_final.iterrows(): ctx += f"{r['Nombre']}: {r['Valor Acciones']:.0f}€ (P&L: {r['Ganancia']:.0f}€). "
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])
            
        if p := st.chat_input("Pregunta..."):
            st.session_state.messages.append({"role":"user","content":p})
            with st.chat_message("user"): st.markdown(p)
            
            if HAS_GROQ:
                with st.chat_message("assistant"):
                    try:
                        cli = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
                        stream = cli.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role":"system","content":f"Eres asesor experto. Datos: {ctx}"}, *st.session_state.messages], stream=True)
                        resp = st.write_stream(stream)
                        st.session_state.messages.append({"role":"assistant","content":resp})
                    except: st.error("Error AI")

    # ------------------------------------------------------------------
    # 🔮 MONTE CARLO (MEJORADO)
    # ------------------------------------------------------------------
    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Futuro (Monte Carlo)")
        st.caption("Proyección basada en la volatilidad real de tus activos.")
        
        c_p, c_g = st.columns([1, 3])
        with c_p:
            ys = st.slider("Años", 5, 40, 20)
            mu, sig = 0.07, 0.15 # Defaults
            if not history_data.empty:
                log_ret = np.log(history_data/history_data.shift(1)).dropna()
                mu = log_ret.mean().mean()*252
                sig = log_ret.mean().std()*np.sqrt(252)
            
            st.metric("Retorno Histórico", f"{mu*100:.1f}%")
            st.metric("Volatilidad Histórica", f"{sig*100:.1f}%")
            
            run = st.button("Simular", type="primary")

        with c_g:
            if run:
                paths = []
                S0 = patrimonio_total if patrimonio_total > 0 else 10000
                dt = 1/252
                
                # 500 Escenarios
                for _ in range(500):
                    p = [S0]
                    for _ in range(int(ys*252)):
                        p.append(p[-1]*np.exp((mu-0.5*sig**2)*dt + sig*np.sqrt(dt)*np.random.normal(0,1)))
                    paths.append(p)
                
                # Percentiles
                paths = np.array(paths)
                p10 = np.percentile(paths, 10, axis=0)
                p50 = np.percentile(paths, 50, axis=0)
                p90 = np.percentile(paths, 90, axis=0)
                
                x = np.linspace(0, ys, len(p50))
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=p90, mode='lines', line=dict(width=0), showlegend=False))
                fig.add_trace(go.Scatter(x=x, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0,204,150,0.2)', name='Rango Probable (80%)'))
                fig.add_trace(go.Scatter(x=x, y=p50, mode='lines', line=dict(color='#00CC96', width=3), name='Escenario Base'))
                
                fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', title="Proyección de Riqueza", yaxis_title="€")
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"En el escenario base (mediana), tendrías **{p50[-1]:,.0f} €** dentro de {ys} años.")

    # ------------------------------------------------------------------
    # ⚖️ REBALANCEO (NUEVA LÓGICA NO-SELL)
    # ------------------------------------------------------------------
    elif pagina == "⚖️ Rebalanceo":
        st.title("⚖️ Planificador de Aportaciones (Smart DCA)")
        st.caption("Calcula cuánto capital inyectar para rebalancear sin vender.")
        
        if df_final.empty: st.warning("Sin activos.")
        else:
            c1,c2 = st.columns([1,1.5])
            
            with c1:
                st.subheader("1. Objetivos")
                target_weights = {}
                total_target = 0
                for i,r in df_final.iterrows():
                    w = st.number_input(f"{r['Nombre']} (%)", 0, 100, int(r['Peso %']), key=i)
                    target_weights[r['Nombre']] = w
                    total_target += w
                
                st.metric("Total Objetivo", f"{total_target}%", delta="OK" if total_target==100 else "Ajustar", delta_color="normal" if total_target==100 else "inverse")
                
                months = st.number_input("Plazo para completar (Meses)", 1, 60, 12)
            
            with c2:
                st.subheader("2. Plan de Aportación")
                if total_target == 100:
                    # Lógica de cálculo "No-Sell"
                    # 1. Encontrar el activo más sobreponderado relativo (ratio valor/target)
                    # Esto define el tamaño mínimo de la cartera equilibrada
                    
                    max_ratio = 0
                    for i, r in df_final.iterrows():
                        target_pct = target_weights[r['Nombre']] / 100
                        if target_pct > 0:
                            ratio = r['Valor Acciones'] / target_pct
                            if ratio > max_ratio: max_ratio = ratio
                    
                    # Cartera objetivo mínima para no vender nada
                    target_portfolio_size = max_ratio
                    capital_needed = target_portfolio_size - df_final['Valor Acciones'].sum()
                    
                    # Si ya está equilibrado o necesitamos meter muy poco, ajustamos
                    if capital_needed < 0: capital_needed = 0
                    
                    monthly_injection = capital_needed / months
                    
                    st.success(f"Para equilibrar sin vender, necesitas inyectar **{capital_needed:,.2f} €**.")
                    st.metric("Aportación Mensual Sugerida", f"{monthly_injection:,.2f} €")
                    
                    # Desglose
                    plan = []
                    prompt_txt = f"Tengo una cartera de {patrimonio_total}€. Quiero rebalancear en {months} meses aportando {monthly_injection:.2f}€/mes para llegar a estos %:\n"
                    
                    for i, r in df_final.iterrows():
                        target_val = target_portfolio_size * (target_weights[r['Nombre']] / 100)
                        gap = target_val - r['Valor Acciones']
                        monthly_buy = gap / months if gap > 0 else 0
                        
                        plan.append({
                            "Activo": r['Nombre'],
                            "Peso Actual": f"{r['Peso %']:.1f}%",
                            "Objetivo": f"{target_weights[r['Nombre']]}%",
                            "Comprar/Mes (€)": monthly_buy
                        })
                        prompt_txt += f"- {r['Nombre']}: Objetivo {target_weights[r['Nombre']]}%. Comprar {monthly_buy:.2f}€/mes.\n"
                    
                    st.dataframe(pd.DataFrame(plan))
                    
                    st.divider()
                    with st.expander("🤖 Consultar a Gemini (Prompt)"):
                        st.code(prompt_text + "\n¿Es este plan eficiente?", language='text')
                    
                    st.subheader("✅ Calendario")
                    cols = st.columns(6)
                    for m in range(1, months+1):
                        with cols[(m-1)%6]:
                            st.checkbox(f"Mes {m}", key=f"chk_{m}")

# --- BOT NOTICIAS ---
if st.session_state['show_news']:
    with col_news:
        c1, c2 = st.columns([3, 1])
        with c1: st.markdown("### 🤖 News")
        with c2: 
            if st.button("🔄", key="ref"): clear_cache(); st.rerun()
        
        tm = {'Hoy': 'd', 'Semana': 'w'}; sel = st.pills("Filtro", list(tm.keys()), default="Hoy", label_visibility="collapsed")
        news = get_global_news(my_tickers, tm[sel])
        
        h = ""
        if news:
            for n in news:
                im = f"<img src='{n['image']}' class='news-img'/>" if n.get('image') else ""
                h += f"""<div class="news-card">{im}<div class="news-source">{n.get('source','Web')} • {n.get('date','')}</div><div class="news-title"><a href="{n['url']}" target="_blank">{n['title']}</a></div></div>"""
        else: h = "<div style='text-align:center;color:#666;padding:20px'>💤 Sin noticias</div>"
        
        st.markdown(f"<div class='news-scroll-area'>{h}</div>", unsafe_allow_html=True)
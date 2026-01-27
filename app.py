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
import openai 

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial Ultra", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# ==============================================================================
# 🌑 DISEÑO DARK MODE PROFESIONAL
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp { background-color: #0E1117; color: #FAFAFA; font-family: 'Inter', sans-serif; }
    h1, h2, h3, p, div, span, label { color: #E6E6E6 !important; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }

    /* KPIs Flotantes */
    div[data-testid="stMetric"] {
        background-color: #21262D; border: 1px solid #30363D; border-radius: 12px;
        padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; color: white !important; font-weight: 700; }
    div[data-testid="stMetricDelta"] svg { display: none; } 
    
    /* Botones */
    .stButton>button { border-radius: 8px; font-weight: 600; border: 1px solid #30363D; background-color: #21262D; color: white; transition: 0.3s; }
    .stButton>button:hover { border-color: #00CC96; color: #00CC96; }
    .stButton>button[kind="primary"] { background: linear-gradient(135deg, #00CC96 0%, #007d5c 100%); border: none; color: black !important; }

    /* Chat Noticias (Fixed) */
    .news-scroll-area {
        height: 68vh; overflow-y: auto; padding: 15px; background-color: #161B22;
        border: 1px solid #30363D; border-radius: 12px; margin-top: 15px;
    }
    .news-scroll-area::-webkit-scrollbar { width: 6px; }
    .news-scroll-area::-webkit-scrollbar-thumb { background: #30363D; border-radius: 4px; }
    
    .news-card {
        background-color: #0d1117; border-radius: 8px; padding: 12px; margin-bottom: 12px;
        border: 1px solid #30363D; display: flex; flex-direction: column; transition: transform 0.2s;
    }
    .news-card:hover { transform: translateY(-2px); border-color: #00CC96; }
    .news-img { width: 100%; height: 100px; object-fit: cover; border-radius: 6px; margin-bottom: 8px; opacity: 0.8; }
    .news-title a { color: #58a6ff !important; text-decoration: none; font-weight: 600; font-size: 0.9rem; }
    .news-source { font-size: 0.7rem; color: #8b949e; text-transform: uppercase; margin-bottom: 4px; }

    /* Gráficos Limpios */
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
# 🔄 LOGIN
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
            em = st.text_input("Email", key="l1"); pa = st.text_input("Pass", type="password", key="l2")
            if st.button("Entrar", use_container_width=True):
                try:
                    res = supabase.auth.sign_in_with_password({"email": em, "password": pa})
                    st.session_state['user'] = res.user; st.rerun()
                except: st.error("Credenciales incorrectas.")
        with tab2:
            em2 = st.text_input("Email", key="r1"); pa2 = st.text_input("Pass", type="password", key="r2")
            if st.button("Crear", type="primary", use_container_width=True):
                try: supabase.auth.sign_up({"email": em2, "password": pa2}); st.success("Creado.")
                except: st.error("Error.")
    st.stop()

user = st.session_state['user']

# ==============================================================================
# 🧠 LÓGICA DE DATOS ROBUSTA (EL CEREBRO DE LA APP)
# ==============================================================================

def sanitize_input(text): return re.sub(r'[^\w\s\-\.]', '', str(text)).strip().upper()

def safe_metric_calc(series):
    """Calcula métricas financieras de forma segura"""
    clean = series.dropna()
    if len(clean) < 2: return 0, 0, 0, 0
    
    # Retornos logarítmicos son más precisos, pero usamos simples para velocidad
    returns = clean.pct_change().dropna()
    if returns.empty: return 0, 0, 0, 0
    
    # 1. Retorno Total
    try: total_ret = (clean.iloc[-1] / clean.iloc[0]) - 1
    except: total_ret = 0
    
    # 2. Volatilidad Anual
    vol = returns.std() * np.sqrt(252)
    
    # 3. Sharpe Ratio (RF = 3%)
    mean_ret = returns.mean() * 252
    sharpe = (mean_ret - 0.03) / vol if vol > 0.001 else 0
    
    # 4. Max Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    return total_ret, vol, max_dd, sharpe

@st.cache_data(ttl=60)
def get_user_data(uid):
    """Obtiene datos de Supabase"""
    assets = pd.DataFrame(supabase.table('assets').select("*").eq('user_id', uid).execute().data)
    liq_res = supabase.table('liquidity').select("*").eq('user_id', uid).execute().data
    
    if not liq_res:
        supabase.table('liquidity').insert({"user_id": uid, "name": "Principal", "amount": 0.0}).execute()
        liquidity = 0.0; liq_id = 0
    else:
        liquidity = liq_res[0]['amount']; liq_id = liq_res[0]['id']
        
    return assets, liquidity, liq_id

@st.cache_data(ttl=600) # Cache 10 min, se limpia con el botón
def get_market_data_robust(tickers):
    """Obtiene Precios Actuales + Históricos de forma robusta"""
    if not tickers: return {}, pd.DataFrame(), pd.Series()
    
    unique_tickers = list(set(tickers + ['SPY']))
    current_prices = {}
    history_df = pd.DataFrame()
    benchmark_series = pd.Series()
    
    try:
        # 1. Descarga Masiva (Periodo largo para asegurar datos)
        # Usamos 'Adj Close' para el histórico y 'Close' del último día para el precio actual
        data = yf.download(unique_tickers, period="2y", interval="1d", group_by='ticker', progress=False)
        
        # 2. Procesar cada ticker
        for t in unique_tickers:
            try:
                # Extraer serie del ticker (Manejo de MultiIndex de Yahoo)
                if len(unique_tickers) > 1:
                    df_t = data[t]
                else:
                    df_t = data # Si solo hay 1, no tiene nivel superior
                
                # --- Precio Actual (Último valor no nulo) ---
                # Esto arregla el problema de "no coge bien el valor"
                last_price = df_t['Close'].dropna().iloc[-1]
                current_prices[t] = last_price
                
                # --- Histórico ---
                series_hist = df_t['Adj Close'].dropna()
                series_hist.index = series_hist.index.tz_localize(None) # Quitar zona horaria
                
                if t == 'SPY':
                    benchmark_series = series_hist
                else:
                    # Renombrar serie para el DataFrame conjunto
                    series_hist.name = t
                    if history_df.empty: history_df = pd.DataFrame(series_hist)
                    else: history_df = history_df.join(series_hist, how='outer')
                    
            except Exception as e:
                print(f"Error procesando {t}: {e}")
                current_prices[t] = 0.0
                
        # Limpiar histórico combinado
        history_df = history_df.fillna(method='ffill').fillna(method='bfill')
        benchmark_series = benchmark_series.fillna(method='ffill').fillna(method='bfill')
        
    except Exception as e:
        print(f"Error descarga masiva: {e}")
        
    return current_prices, history_df, benchmark_series

@st.cache_data(ttl=900)
def get_global_news(tickers, time_filter='d'):
    results = []
    query = f"{tickers[0]} finanzas mercado" if tickers else "Mercado financiero noticias"
    try:
        with DDGS() as ddgs:
            raw_news = list(ddgs.news(query, region="es-es", safesearch="off", timelimit=time_filter, max_results=10))
            for n in raw_news:
                results.append({
                    'title': n.get('title'), 'source': n.get('source'), 'date': n.get('date'),
                    'url': n.get('url'), 'image': n.get('image', None)
                })
    except: pass
    return results

# Funciones DB
def add_asset_db(t, n, s, p, pl):
    supabase.table('assets').insert({"user_id": user.id, "ticker": t, "nombre": n, "shares": s, "avg_price": p, "platform": pl}).execute()
    st.cache_data.clear()

def update_asset_db(asset_id, shares, avg_price):
    supabase.table('assets').update({"shares": shares, "avg_price": avg_price}).eq('id', asset_id).execute()
    st.cache_data.clear()

def delete_asset_db(id_del):
    supabase.table('assets').delete().eq('id', id_del).execute()
    st.cache_data.clear()

def update_liquidity_balance(liq_id, new_amount):
    supabase.table('liquidity').update({"amount": new_amount}).eq('id', liq_id).execute()
    st.cache_data.clear()

# ==============================================================================
# PROCESAMIENTO DE DATOS MAESTRO
# ==============================================================================
df_assets, total_liquidez, cash_id = get_user_data(user.id)
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()
current_prices_dict = {}

if not df_assets.empty:
    my_tickers = df_assets['ticker'].unique().tolist()
    
    # LLAMADA ÚNICA Y ROBUSTA A YAHOO
    current_prices_dict, history_data, benchmark_data = get_market_data_robust(my_tickers)
    
    # Construir DataFrame Final
    df_assets['Precio Actual'] = df_assets['ticker'].map(current_prices_dict).fillna(0.0)
    df_assets['Valor Acciones'] = df_assets['shares'] * df_assets['Precio Actual']
    df_assets['Dinero Invertido'] = df_assets['shares'] * df_assets['avg_price']
    df_assets['Ganancia'] = df_assets['Valor Acciones'] - df_assets['Dinero Invertido']
    
    # Rentabilidad segura (evitar div/0)
    df_assets['Rentabilidad %'] = df_assets.apply(
        lambda r: (r['Ganancia'] / r['Dinero Invertido'] * 100) if r['Dinero Invertido'] > 0 else 0, axis=1
    )
    
    # Peso en cartera
    total_inv_val = df_assets['Valor Acciones'].sum()
    df_assets['Peso %'] = df_assets.apply(
        lambda r: (r['Valor Acciones'] / total_inv_val * 100) if total_inv_val > 0 else 0, axis=1
    )
    
    df_final = df_assets.rename(columns={'nombre': 'Nombre'})

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0.0
patrimonio_total = total_inversiones + total_liquidez

# ==============================================================================
# UI: SIDEBAR
# ==============================================================================
with st.sidebar:
    avatar = user.user_metadata.get('avatar_url', '')
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; padding:10px; background:#21262D; border-radius:8px; border:1px solid #30363D;'>
        <img src='{avatar}' style='width:35px; border-radius:50%; border:2px solid #00CC96;'>
        <div style='line-height:1.2'><b style='color:white'>{user.user_metadata.get('full_name','Inversor')}</b><br><span style='font-size:0.7em; color:#00CC96'>● Online</span></div>
    </div><br>""", unsafe_allow_html=True)
    
    # BOTÓN DE ACTUALIZAR PRECIOS (Lo que pediste)
    if st.button("🔄 Actualizar Precios", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    
    c1, c2 = st.columns([1,4])
    with c1: st.write("📰")
    with c2: st.session_state['show_news'] = st.toggle("Noticias", value=st.session_state['show_news'])
    
    pagina = st.radio("MENÚ", ["📊 Dashboard Pro", "💰 Liquidez", "➕ Inversiones", "💬 Asesor AI", "🔮 Monte Carlo", "⚖️ Rebalanceo"])
    
    st.divider()
    if st.button("Cerrar Sesión", use_container_width=True): 
        supabase.auth.sign_out(); st.session_state['user']=None; st.rerun()

# ==============================================================================
# UI: PÁGINAS PRINCIPALES
# ==============================================================================
if st.session_state['show_news']: col_main, col_news = st.columns([3.5, 1.3])
else: col_main = st.container(); col_news = st.container()

with col_main:
    
    # --- DASHBOARD PRO ---
    if pagina == "📊 Dashboard Pro":
        st.title("📊 Sala de Mando")
        
        # Filtros Fecha
        c_date, _ = st.columns([1, 3])
        with c_date: 
            start_date = st.date_input("📅 Análisis desde:", value=datetime.now()-timedelta(days=365))
        
        # --- CÁLCULOS COMPLEJOS PARA DASHBOARD ---
        cagr = 0; sharpe = 0; max_dd = 0; vol = 0
        portfolio_cum_ret = pd.Series()
        
        if not history_data.empty:
            # Filtrar por fecha
            dt_start = pd.to_datetime(start_date)
            hist_filt = history_data[history_data.index >= dt_start]
            
            if not hist_filt.empty:
                # Calcular Retorno Diario de cada activo
                daily_rets = hist_filt.pct_change().dropna()
                # Retorno de la Cartera (Equiponderado por simplicidad, o ponderado si tuvieramos historial de pesos)
                # Aproximación: Media de retornos de los activos presentes
                port_daily_ret = daily_rets.mean(axis=1)
                
                # Métricas usando la función segura
                tot_r, vol, max_dd, sharpe = safe_metric_calc(port_daily_ret + 1) # Sumamos 1 para simular precio
                
                # Corrección métricas
                vol = port_daily_ret.std() * np.sqrt(252) * 100
                mean_ret = port_daily_ret.mean() * 252
                sharpe = (mean_ret - 0.03) / (vol/100) if vol > 0 else 0
                
                cum = (1 + port_daily_ret).cumprod()
                peak = cum.cummax()
                dd = (cum - peak) / peak
                max_dd = dd.min() * 100
                
                # CAGR
                days = (hist_filt.index[-1] - hist_filt.index[0]).days
                if days > 30:
                    total_ret_period = (1 + port_daily_ret).prod() - 1
                    cagr = ((1 + total_ret_period) ** (365/days) - 1) * 100
        
        # --- FILA 1: KPIs FINANCIEROS ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💰 Patrimonio Neto", f"{patrimonio_total:,.2f} €")
        
        if total_inversiones > 0:
            pnl = df_final['Ganancia'].sum()
            k2.metric("📈 Ganancia Latente", f"{pnl:+,.2f} €", delta_color="normal" if pnl>=0 else "inverse")
        else: k2.metric("📈 Ganancia", "0.00 €")
        
        k3.metric("📊 CAGR Est.", f"{cagr:.2f}%", help="Crecimiento Anual Compuesto estimado")
        k4.metric("⚖️ Ratio Sharpe", f"{sharpe:.2f}", help=">1 Bueno, >2 Excelente")
        
        st.divider()
        
        # --- FILA 2: GRÁFICOS (ARREGLADO SP500) ---
        g1, g2 = st.columns([2, 1.2])
        
        with g1:
            st.subheader("🚀 Rendimiento vs S&P 500")
            if not history_data.empty:
                # Lógica Robusta de Comparación (Base 100)
                # 1. Alinear fechas
                common_idx = history_data.index.intersection(benchmark_data.index)
                common_idx = common_idx[common_idx >= pd.to_datetime(start_date)]
                
                if len(common_idx) > 1:
                    # Datos Cartera
                    h_aligned = history_data.loc[common_idx]
                    # Retorno acumulado cartera (promedio activos)
                    port_ret = h_aligned.pct_change().mean(axis=1).fillna(0)
                    port_cum = (1 + port_ret).cumprod() * 100 # Base 100
                    
                    # Datos Benchmark
                    b_aligned = benchmark_data.loc[common_idx]
                    bench_ret = b_aligned.pct_change().fillna(0)
                    bench_cum = (1 + bench_ret).cumprod() * 100 # Base 100
                    
                    # Graficar
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Tu Cartera", line=dict(color='#00CC96', width=2)))
                    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500", line=dict(color='gray', dash='dot')))
                    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', 
                                      yaxis_title="Base 100")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Faltan datos coincidentes.")
            else: st.info("Sin histórico suficiente.")
            
        with g2:
            st.subheader("⚠️ Riesgo (Max Drawdown)")
            st.metric("Caída Máxima Histórica", f"{max_dd:.2f}%", delta_color="inverse")
            st.metric("Volatilidad Anual", f"{vol:.2f}%", help="Cuanto se mueve tu cartera")
            
            # Gráfico Mini Donut
            if patrimonio_total > 0:
                fig_d = px.pie(values=[total_liquidez, total_inversiones], names=['Liquidez', 'Inversión'], hole=0.7, color_discrete_sequence=['#636EFA', '#00CC96'])
                fig_d.update_layout(template="plotly_dark", height=180, showlegend=False, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_d, use_container_width=True)

        st.divider()
        
        # --- FILA 3: PROYECCIÓN JUBILACIÓN (Directo en Dashboard) ---
        st.subheader("🔮 Tu Futuro Financiero")
        c_fut1, c_fut2 = st.columns([1, 3])
        with c_fut1:
            yrs = st.number_input("Años vista", 5, 40, 15)
            contr = st.number_input("Ahorro Mensual (€)", 0, 5000, 200)
            tasa = cagr if cagr > 0 else 7.0
            st.caption(f"Creciendo al {tasa:.1f}% anual")
        
        with c_fut2:
            # Cálculo rápido
            r = tasa / 100
            months = yrs * 12
            future_val = patrimonio_total * (1+r)**yrs
            # Formula serie anualidad
            future_cont = contr * 12 * (((1+r)**yrs - 1) / r) if r!=0 else contr*12*yrs
            total_est = future_val + future_cont
            
            st.metric(f"Patrimonio en {yrs} años", f"{total_est:,.0f} €")
            # Barra de progreso visual
            st.progress(min(100, int(patrimonio_total/100000 * 100))) # Meta ejemplo 100k
            st.caption("Progreso hacia los 100k (Ejemplo)")

    # ------------------------------------------------------------------
    # 💰 PÁGINA LIQUIDEZ
    # ------------------------------------------------------------------
    elif pagina == "💰 Liquidez (Cash)":
        st.title("💰 Gestión de Liquidez")
        st.markdown(f"<div style='text-align:center; padding:40px; background:#21262D; border-radius:15px; border:1px solid #30363D;'><h1 style='color:white'>{total_liquidez:,.2f} €</h1></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            a = st.number_input("Ingreso (€)", 0.0, step=50.0)
            if st.button("Ingresar", type="primary") and a>0:
                update_liquidity_balance(int(cash_id), total_liquidez+a); st.rerun()
        with c2:
            b = st.number_input("Retiro (€)", 0.0, step=50.0)
            if st.button("Retirar") and b>0:
                if b>total_liquidez: st.error("No hay saldo")
                else: update_liquidity_balance(int(cash_id), total_liquidez-b); st.rerun()

    # ------------------------------------------------------------------
    # ➕ PÁGINA INVERSIONES (CORREGIDA KEYERROR)
    # ------------------------------------------------------------------
    elif pagina == "➕ Inversiones":
        st.title("➕ Gestión de Activos")
        t1, t2, t3 = st.tabs(["Añadir", "Operar", "Editar"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                q = st.text_input("Buscar Activo:")
                if st.button("🔍") and q:
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
                        # Fetch price safely
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
                                    av = i/sh
                                    add_asset_db(tk, st.session_state['sel_add'].get('shortname',tk), sh, av, pl)
                                    st.success("Guardado"); time.sleep(1); st.rerun()
                    except: st.error("Error datos")

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
    # 🔮 MONTE CARLO
    # ------------------------------------------------------------------
    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Futuro")
        ys = st.slider("Años", 5, 40, 20)
        if st.button("Simular"):
            mu, sig = 0.07, 0.15
            if not history_data.empty:
                d = history_data.pct_change().mean(axis=1).dropna()
                mu = d.mean()*252; sig = d.std()*np.sqrt(252)
            
            paths = []
            for _ in range(30):
                p = [patrimonio_total]
                for _ in range(int(ys*252)):
                    p.append(p[-1]*np.exp((mu-0.5*sig**2)/252 + sig*np.sqrt(1/252)*np.random.normal(0,1)))
                paths.append(p)
            
            fig = go.Figure()
            x = np.linspace(0, ys, len(paths[0]))
            for p in paths: fig.add_trace(go.Scatter(x=x, y=p, line=dict(color='rgba(0,204,150,0.1)'), showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=np.mean(paths,axis=0), line=dict(color='white',width=3)))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # ⚖️ REBALANCEO
    # ------------------------------------------------------------------
    elif pagina == "⚖️ Rebalanceo":
        st.title("⚖️ Rebalanceo")
        if df_final.empty: st.warning("Sin activos")
        else:
            c1,c2=st.columns(2)
            ws={}; tot=0
            with c1:
                for i,r in df_final.iterrows():
                    w = st.number_input(f"{r['Nombre']} %", 0, 100, int(r['Peso %']), key=i)
                    ws[r['Nombre']]=w; tot+=w
                st.metric("Total", f"{tot}%")
            with c2:
                if tot==100 and st.button("Calcular"):
                    mc = max([r['Valor Acciones']/(ws[r['Nombre']]/100) for i,r in df_final.iterrows() if ws[r['Nombre']]>0]+[patrimonio_total])
                    dat = [{'Activo':r['Nombre'], 'Comprar':max(0, (mc*ws[r['Nombre']]/100)-r['Valor Acciones'])} for i,r in df_final.iterrows()]
                    fig=px.bar(pd.DataFrame(dat), x='Activo', y='Comprar')
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

# --- NEWS FIXED ---
if st.session_state['show_news']:
    with col_news:
        c1,c2=st.columns([3,1])
        with c1: st.markdown("### 🤖 News")
        with c2: 
            if st.button("🔄"): clear_cache(); st.rerun()
        
        tm = {'Hoy':'d','Semana':'w'}; sel = st.pills("Filtro", list(tm.keys()), default="Hoy")
        news = get_global_news(my_tickers, tm[sel])
        
        h=""
        if news:
            for n in news:
                im = f"<img src='{n['image']}' class='news-img'/>" if n.get('image') else ""
                h+=f"<div class='news-card'>{im}<div class='news-source'>{n.get('source','')}</div><div class='news-title'><a href='{n['url']}' target='_blank'>{n['title']}</a></div></div>"
        else: h="<div style='text-align:center;color:#666;padding:20px'>💤 Nada reciente</div>"
        
        st.markdown(f"<div class='news-scroll-area'>{h}</div>", unsafe_allow_html=True)
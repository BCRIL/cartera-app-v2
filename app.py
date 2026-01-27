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
            
            with st.form("login_form"):
                em = st.text_input("Email")
                pa = st.text_input("Pass", type="password")
                if st.form_submit_button("Entrar", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": em, "password": pa})
                        st.session_state['user'] = res.user
                        st.rerun()
                    except: st.error("Credenciales incorrectas.")
        with tab2:
            with st.form("signup_form"):
                em2 = st.text_input("Email")
                pa2 = st.text_input("Pass", type="password")
                if st.form_submit_button("Crear Cuenta", use_container_width=True):
                    try: supabase.auth.sign_up({"email": em2, "password": pa2}); st.success("Creado.")
                    except: st.error("Error.")
    st.stop()

user = st.session_state['user']

# --- LÓGICA DE DATOS ROBUSTA ---
def sanitize_input(text): return re.sub(r'[^\w\s\-\.]', '', str(text)).strip().upper()

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

# 🔥 FUNCIÓN CLAVE: PRECIOS EN TIEMPO REAL REAL (SIN CACHE CUANDO SE PIDE)
# No usamos st.cache_data aquí para que el botón fuerce la recarga
def get_real_time_prices(tickers):
    """Obtiene el precio del último segundo usando fast_info"""
    if not tickers: return {}
    prices = {}
    for t in tickers:
        try:
            # fast_info es más rápido y preciso para el "ahora" que .download
            info = yf.Ticker(t).fast_info
            if 'last_price' in info and info['last_price']:
                prices[t] = info['last_price']
            else:
                # Fallback si falla fast_info
                hist = yf.Ticker(t).history(period='1d')
                prices[t] = hist['Close'].iloc[-1] if not hist.empty else 0.0
        except:
            prices[t] = 0.0
    return prices

@st.cache_data(ttl=300)
def get_historical_data(tickers):
    """Histórico para gráficos (Sincronizado)"""
    if not tickers: return pd.DataFrame()
    try:
        tickers_api = list(set(tickers + ['SPY']))
        # Descargamos suficiente historia
        data = yf.download(tickers_api, period="2y", interval="1d", progress=False)['Adj Close']
        
        # Si solo hay un ticker, data es una Series, convertir a DF
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers_api[0])
            
        data.index = data.index.tz_localize(None)
        return data.fillna(method='ffill').fillna(method='bfill')
    except: return pd.DataFrame()

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

def clear_cache(): 
    st.cache_data.clear()

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

# --- CARGA Y PROCESAMIENTO ---
df_assets, total_liquidez, cash_id = get_user_data(user.id)
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()
my_tickers = []

if not df_assets.empty:
    my_tickers = df_assets['ticker'].unique().tolist()
    
    # 1. Obtener Precios Actuales (Sin caché si se pulsa actualizar)
    current_prices = get_real_time_prices(my_tickers)
    
    # 2. Obtener Histórico (Cacheado, salvo force refresh)
    history_raw = get_historical_data(my_tickers)
    
    # Mapear precios
    df_assets['Precio Actual'] = df_assets['ticker'].map(current_prices).fillna(0.0)
    
    # Cálculos
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
    
    # Separar Benchmark
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
    avatar = user.user_metadata.get('avatar_url', '')
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; padding:10px; background:#21262D; border-radius:8px;'>
        <img src='{avatar}' style='width:35px; border-radius:50%; border:2px solid #00CC96;'>
        <div style='line-height:1.2'><b style='color:white'>{user.user_metadata.get('full_name','Inversor')}</b><br><span style='font-size:0.7em; color:#00CC96'>● Online</span></div>
    </div><br>""", unsafe_allow_html=True)
    
    # 🔥 BOTÓN DE ACTUALIZACIÓN REAL 🔥
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
    # 📊 PÁGINA DASHBOARD
    # ------------------------------------------------------------------
    if pagina == "📊 Dashboard & Alpha":
        st.title("📊 Control de Mando Integral")
        
        col_kpi, col_date = st.columns([3, 1])
        with col_date: 
            start_date = st.date_input("📅 Rango de Análisis:", value=datetime.now()-timedelta(days=365))
        
        # --- CÁLCULO DE MÉTRICAS ROBUSTO ---
        vol_anual = 0
        sharpe_ratio = 0
        max_drawdown = 0
        cagr = 0
        beta_portfolio = 1.0
        
        if not history_data.empty:
            # 1. Simular la cartera actual sobre el pasado (Backtest teórico)
            # Para simplificar: Promedio de los retornos de los activos actuales
            daily_returns = history_data.pct_change().mean(axis=1).dropna()
            
            # Volatilidad
            vol_anual = daily_returns.std() * np.sqrt(252) * 100
            
            # Sharpe
            rf = 0.03
            mean_ret = daily_returns.mean() * 252
            sharpe_ratio = (mean_ret - rf) / (vol_anual/100) if vol_anual > 0 else 0
            
            # Max Drawdown
            cumulative = (1 + daily_returns).cumprod()
            peak = cumulative.cummax()
            dd = (cumulative - peak) / peak
            max_drawdown = dd.min() * 100
            
            # Beta
            if not benchmark_data.empty:
                bench_ret = benchmark_data.pct_change().dropna()
                common_idx = daily_returns.index.intersection(bench_ret.index)
                if len(common_idx) > 20:
                    cov = daily_returns.loc[common_idx].cov(bench_ret.loc[common_idx])
                    var = bench_ret.loc[common_idx].var()
                    beta_portfolio = cov / var if var != 0 else 1.0

        with col_kpi:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("💰 Patrimonio Neto", f"{patrimonio_total:,.2f} €")
            
            if total_inversiones > 0:
                pnl_total = df_final['Ganancia'].sum()
                rent_total_pct = (pnl_total / df_final['Dinero Invertido'].sum()) * 100 if df_final['Dinero Invertido'].sum() > 0 else 0
                delta_color = "normal" if pnl_total >= 0 else "inverse"
                k2.metric("📈 P&L Latente", f"{pnl_total:+,.2f} €", f"{rent_total_pct:+.2f}%", delta_color=delta_color)
            else:
                k2.metric("📈 P&L Latente", "0.00 €")
            
            k3.metric("💧 Liquidez", f"{total_liquidez:,.2f} €", f"{(total_liquidez/patrimonio_total*100 if patrimonio_total>0 else 0):.1f}%")
            k4.metric("⚖️ Ratio Sharpe", f"{sharpe_ratio:.2f}")

        st.divider()
        
        # --- FILA 2: RIESGO ---
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("⚡ Volatilidad", f"{vol_anual:.2f}%")
        r2.metric("📉 Max Drawdown", f"{max_drawdown:.2f}%", delta_color="inverse")
        r3.metric("🌊 Beta Cartera", f"{beta_portfolio:.2f}")
        r4.metric("🔄 Activos", f"{len(df_final)}")
        
        st.divider()

        # --- FILA 3: GRÁFICO COMPARATIVO (ARREGLADO) ---
        c_chart, c_donut = st.columns([2, 1.2])
        with c_chart:
            st.subheader("🏁 Rendimiento Relativo (Base 100)")
            if not history_data.empty:
                # Alinear fechas
                dt_start = pd.to_datetime(start_date)
                
                # Intersección segura de índices
                if not benchmark_data.empty:
                    common_dates = history_data.index.intersection(benchmark_data.index)
                    common_dates = common_dates[common_dates >= dt_start]
                else:
                    common_dates = history_data.index[history_data.index >= dt_start]

                if len(common_dates) > 5:
                    # Datos filtrados y alineados
                    hist_aligned = history_data.loc[common_dates]
                    
                    # Cartera: Promedio de los activos (Base 100)
                    # Calculamos el retorno acumulado desde el día 1 del gráfico
                    port_ret = hist_aligned.pct_change().mean(axis=1).fillna(0)
                    port_cum = (1 + port_ret).cumprod() * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Tu Cartera", line=dict(color='#00CC96', width=2)))
                    
                    if not benchmark_data.empty:
                        bench_aligned = benchmark_data.loc[common_dates]
                        bench_ret = bench_aligned.pct_change().fillna(0)
                        bench_cum = (1 + bench_ret).cumprod() * 100
                        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500", line=dict(color='gray', dash='dot')))
                    
                    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', 
                                      hovermode="x unified", yaxis_title="Valor (Base 100)")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Datos insuficientes en el rango seleccionado.")
            else: st.info("Añade activos para ver el histórico.")

        with c_donut:
            st.subheader("🍰 Asset Allocation")
            if patrimonio_total > 0:
                labels = ['Liquidez'] + df_final['Nombre'].tolist() if not df_final.empty else ['Liquidez']
                values = [total_liquidez] + df_final['Valor Acciones'].tolist() if not df_final.empty else [total_liquidez]
                fig_pie = px.pie(names=labels, values=values, hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(template="plotly_dark", height=320, showlegend=True, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.info("Cartera vacía.")

        st.divider()

        # --- FILA 4: MAPA CALOR ---
        c_tree, c_bar = st.columns([1.5, 1.5])
        with c_tree:
            st.subheader("🗺️ Mapa de Calor")
            if not df_final.empty:
                fig_tree = px.treemap(df_final, path=['Nombre'], values='Valor Acciones', color='Rentabilidad %', 
                                      color_continuous_scale=['#EF553B', '#1e1e1e', '#00CC96'], color_continuous_midpoint=0)
                fig_tree.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tree, use_container_width=True)
            else: st.info("Sin inversiones.")

        with c_bar:
            st.subheader("🏆 Ganadores (€)")
            if not df_final.empty:
                df_sorted = df_final.sort_values('Ganancia', ascending=True)
                colors = ['#EF553B' if x < 0 else '#00CC96' for x in df_sorted['Ganancia']]
                fig_bar = go.Figure(go.Bar(x=df_sorted['Ganancia'], y=df_sorted['Nombre'], orientation='h', marker_color=colors, text=df_sorted['Ganancia'].apply(lambda x: f"{x:,.2f}€"), textposition='auto'))
                fig_bar.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
            else: st.info("Sin inversiones.")

    # ------------------------------------------------------------------
    # 💰 PÁGINA LIQUIDEZ
    # ------------------------------------------------------------------
    elif pagina == "💰 Liquidez (Cash)":
        st.title("💰 Gestión de Liquidez")
        st.markdown(f"""<div style="text-align:center; padding: 40px; background-color: #21262D; border-radius: 15px; margin-bottom: 30px; border: 1px solid #30363D;"><h1 style="font-size: 4.5rem; color:#FFFFFF; margin: 0;">{total_liquidez:,.2f} €</h1><p style="color:#8B949E;">SALDO DISPONIBLE</p></div>""", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("### 📥 Ingresar")
                a = st.number_input("Ingreso (€)", 0.0, step=50.0, key="in")
                if st.button("Confirmar Ingreso", type="primary", use_container_width=True) and a > 0:
                    update_liquidity_balance(int(cash_id), total_liquidez + a)
                    st.toast(f"✅ Ingresados {a}€"); time.sleep(1); st.rerun()
        with c2:
            with st.container(border=True):
                st.markdown("### 📤 Retirar")
                b = st.number_input("Retiro (€)", 0.0, step=50.0, key="out")
                if st.button("Confirmar Retirada", use_container_width=True) and b > 0:
                    if b > total_liquidez: st.error("Sin saldo.")
                    else:
                        update_liquidity_balance(int(cash_id), total_liquidez - b)
                        st.toast(f"✅ Retirados {b}€"); time.sleep(1); st.rerun()

    # ------------------------------------------------------------------
    # ➕ PÁGINA INVERSIONES
    # ------------------------------------------------------------------
    elif pagina == "➕ Inversiones":
        st.title("➕ Gestión de Activos")
        t1, t2, t3 = st.tabs(["🆕 Añadir Nuevo", "💰 Operar", "✏️ Editar"])
        
        with t1:
            c1, c2 = st.columns([1, 1])
            with c1:
                q = st.text_input("🔍 Buscar (Nombre/ISIN/Ticker):")
                if st.button("Buscar") and q:
                    try:
                        res = search(sanitize_input(q))
                        if 'quotes' in res and res['quotes']: st.session_state['s'] = res['quotes']
                        else: st.warning("No encontrado."); st.session_state['s'] = []
                    except: st.error("Error búsqueda."); st.session_state['s'] = []
                
                if 's' in st.session_state and st.session_state['s']:
                    opts = {f"{x['symbol']} | {x.get('shortname', x.get('longname','N/A'))} ({x.get('exchDisp','Unknown')})" : x for x in st.session_state['s']}
                    if opts:
                        sel = st.selectbox("Selecciona:", list(opts.keys()))
                        if sel in opts: st.session_state['sel_add'] = opts[sel]
            
            with c2:
                if 'sel_add' in st.session_state:
                    tk = st.session_state['sel_add']['symbol']
                    try:
                        inf = yf.Ticker(tk).fast_info
                        p = inf['last_price']
                        mon = inf['currency']
                        if p:
                            st.metric("Precio Actual", f"{p:.2f} {mon}")
                            if mon != 'EUR': st.warning(f"⚠️ Activo en **{mon}**. El sistema usará este precio.")
                            
                            with st.form("new"):
                                i = st.number_input("Dinero Invertido (EUR)", 0.0)
                                v = st.number_input("Valor Actual (EUR)", 0.0)
                                pl = st.selectbox("Broker", ["MyInvestor", "XTB", "TR", "Degiro"])
                                if st.form_submit_button("Guardar") and v > 0:
                                    sh = v / p
                                    av = i / sh if sh > 0 else 0
                                    add_asset_db(tk, st.session_state['sel_add'].get('longname',tk), sh, av, pl)
                                    st.success("Guardado"); time.sleep(1); st.rerun()
                    except: st.error("Error precio")

        with t2:
            if df_final.empty: st.warning("Añade activos.")
            else:
                c1, c2 = st.columns([1, 1])
                with c1:
                    nom = st.selectbox("Activo:", df_final['Nombre'].unique())
                    row = df_final[df_final['Nombre']==nom].iloc[0]
                    st.info(f"Tienes: `{row['shares']:.4f}` accs. Valor: `{row['Valor Acciones']:.2f} €`")
                with c2:
                    tipo = st.radio("Acción:", ["🟢 Comprar", "🔴 Vender"], horizontal=True)
                    m = st.number_input("Importe (€)", 0.0)
                    if m > 0:
                        p = row['Precio Actual']
                        sh_op = m / p
                        if "Comprar" in tipo:
                            if m > total_liquidez: st.error("Falta liquidez")
                            elif st.button("Confirmar"):
                                navg = ((row['shares']*row['avg_price'])+m)/(row['shares']+sh_op)
                                update_asset_db(int(row['id']), row['shares']+sh_op, navg)
                                update_liquidity_balance(int(cash_id), total_liquidez-m)
                                st.rerun()
                        else:
                            if st.button("Confirmar"):
                                nsh = row['shares'] - sh_op
                                if nsh < 0.001: delete_asset_db(int(row['id']))
                                else: update_asset_db(int(row['id']), nsh, row['avg_price'])
                                update_liquidity_balance(int(cash_id), total_liquidez+m)
                                st.rerun()

        with t3:
            if not df_final.empty:
                e = st.selectbox("Editar:", df_final['Nombre'], key='edd')
                er = df_final[df_final['Nombre']==e].iloc[0]
                nsh = st.number_input("Accs", value=float(er['shares']))
                nav = st.number_input("Media", value=float(er['avg_price']))
                if st.button("Guardar Cambios"): update_asset_db(int(er['id']), nsh, nav); st.rerun()
                if st.button("Eliminar", type="secondary"): delete_asset_db(int(er['id'])); st.rerun()

    # ------------------------------------------------------------------
    # 💬 ASESOR AI
    # ------------------------------------------------------------------
    elif pagina == "💬 Asesor AI":
        st.title("💬 Carterapro Bot")
        ctx = f"Liquidez: {total_liquidez:.2f}€. "
        if not df_final.empty:
            for i, r in df_final.iterrows():
                ctx += f"[{r['Nombre']}: {r['Valor Acciones']:.2f}€, P&L {r['Ganancia']:.2f}€]. "
        
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Duda sobre tu cartera..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            if HAS_GROQ:
                with st.chat_message("assistant"):
                    try:
                        client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
                        stream = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "system", "content": f"Eres un asesor experto. Datos: {ctx}. Responde breve."}, *st.session_state.messages],
                            stream=True
                        )
                        response = st.write_stream(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e: st.error(f"Error: {e}")
            else: st.warning("Falta GROQ_API_KEY")

    # ------------------------------------------------------------------
    # 🔮 MONTE CARLO
    # ------------------------------------------------------------------
    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Futuro")
        ys = st.slider("Años", 1, 30, 10)
        if st.button("Simular", type="primary"):
            mu, sigma = 0.07, 0.15
            if not history_data.empty:
                d = history_data.pct_change().dropna()
                mu = d.mean().mean()*252; sigma = d.mean().std()*np.sqrt(252)
            
            paths = []
            for _ in range(30):
                p = [total_inversiones]
                for _ in range(int(ys*252)):
                    p.append(p[-1]*np.exp((mu-0.5*sigma**2)/252 + sigma*np.sqrt(1/252)*np.random.normal(0,1)))
                paths.append(p)
            
            fig = go.Figure()
            x = np.linspace(0, ys, len(paths[0]))
            for p in paths: fig.add_trace(go.Scatter(x=x, y=p, mode='lines', line=dict(color='rgba(0,204,150,0.1)'), showlegend=False))
            fig.add_trace(go.Scatter(x=x, y=np.mean(paths,axis=0), mode='lines', line=dict(color='#00CC96', width=3)))
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', title="Proyección Patrimonial")
            st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------------
    # ⚖️ REBALANCEO
    # ------------------------------------------------------------------
    elif pagina == "⚖️ Rebalanceo":
        st.title("⚖️ Rebalanceo")
        if df_final.empty: st.warning("Sin activos.")
        else:
            c1,c2 = st.columns([1,1.5])
            ws = {}; tot = 0
            with c1:
                for i,r in df_final.iterrows():
                    w = st.number_input(f"{r['Nombre']} %", 0, 100, int(r['Peso %']), key=i)
                    ws[r['Nombre']] = w; tot += w
                st.metric("Total", f"{tot}%")
            with c2:
                if tot==100 and st.button("Calcular", type="primary"):
                    mc = max([r['Valor Acciones']/(ws[r['Nombre']]/100) for i,r in df_final.iterrows() if ws[r['Nombre']]>0]+[patrimonio_total])
                    dat = [{'Activo':r['Nombre'], 'Comprar':max(0, (mc*ws[r['Nombre']]/100)-r['Valor Acciones'])} for i,r in df_final.iterrows()]
                    fig = px.bar(pd.DataFrame(dat), x='Activo', y='Comprar')
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

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
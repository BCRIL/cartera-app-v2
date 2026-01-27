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
from scipy.stats import norm

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial Ultra", layout="wide", page_icon="🏦", initial_sidebar_state="expanded")

# ==============================================================================
# 🌑 DISEÑO DARK MODE "GLASS" (CSS v17)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .stApp { background-color: #0B0E11; color: #E0E0E0; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700; letter-spacing: -0.5px; }
    p, label, span { color: #A0AAB5 !important; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #12151A; border-right: 1px solid #2D333B; }

    /* KPIs Flotantes */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1C2128, #16191D);
        border: 1px solid #2D333B; border-radius: 16px;
        padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"]:hover { border-color: #00CC96; transform: translateY(-3px); transition: all 0.3s ease; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; color: #FFFFFF !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #8B949E !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetricDelta"] svg { display: none; } 
    
    /* Botones */
    .stButton>button { border-radius: 10px; font-weight: 600; border: 1px solid #30363D; background-color: #21262D; color: white; transition: 0.3s; height: 45px; }
    .stButton>button:hover { border-color: #00CC96; color: #00CC96; background-color: #1C2128; }
    .stButton>button[kind="primary"] { background: linear-gradient(135deg, #00CC96 0%, #008f6b 100%); border: none; color: #0B0E11 !important; font-weight: 800; }

    /* Chat Noticias */
    .news-scroll-area {
        height: 72vh; overflow-y: auto; padding: 10px; background-color: #12151A;
        border: 1px solid #2D333B; border-radius: 12px; margin-top: 15px;
    }
    .news-scroll-area::-webkit-scrollbar { width: 5px; }
    .news-scroll-area::-webkit-scrollbar-thumb { background: #30363D; border-radius: 10px; }
    
    .news-card {
        background-color: #1C2128; border-radius: 10px; padding: 15px; margin-bottom: 12px;
        border: 1px solid #2D333B; display: flex; flex-direction: column; transition: transform 0.2s;
    }
    .news-card:hover { transform: scale(1.02); border-color: #00CC96; }
    .news-img { width: 100%; height: 110px; object-fit: cover; border-radius: 8px; margin-bottom: 10px; opacity: 0.85; }
    .news-title a { color: #FFFFFF !important; text-decoration: none; font-weight: 600; font-size: 0.95rem; line-height: 1.4; }
    .news-title a:hover { color: #00CC96 !important; }
    .news-source { font-size: 0.7rem; color: #00CC96; text-transform: uppercase; margin-bottom: 5px; font-weight: bold; letter-spacing: 0.5px; }

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
            if st.button("🇬 Entrar con Google", type="primary", use_container_width=True):
                try:
                    data = supabase.auth.sign_in_with_oauth({"provider": "google", "options": {"redirect_to": "https://carterapro.streamlit.app"}})
                    st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
                except Exception as e: st.error(f"Error: {e}")
            st.divider()
            with st.form("login_form"):
                em = st.text_input("Email")
                pa = st.text_input("Contraseña", type="password")
                if st.form_submit_button("Iniciar Sesión", use_container_width=True):
                    try:
                        res = supabase.auth.sign_in_with_password({"email": em, "password": pa})
                        st.session_state['user'] = res.user
                        st.rerun()
                    except: st.error("Credenciales incorrectas.")
        with tab2:
            with st.form("signup_form"):
                em2 = st.text_input("Email")
                pa2 = st.text_input("Contraseña", type="password")
                if st.form_submit_button("Crear Cuenta", use_container_width=True):
                    try: supabase.auth.sign_up({"email": em2, "password": pa2}); st.success("Creado. Revisa email.")
                    except: st.error("Error.")
    st.stop()

user = st.session_state['user']

# ==============================================================================
# 🧠 ALGORITMOS FINANCIEROS AVANZADOS
# ==============================================================================

def sanitize_input(text): return re.sub(r'[^\w\s\-\.]', '', str(text)).strip().upper()

def calculate_advanced_metrics(series, benchmark=None):
    """Calcula Sharpe, Sortino, VaR, MaxDD"""
    clean = series.dropna()
    if len(clean) < 10: return 0,0,0,0,0
    
    returns = clean.pct_change().dropna()
    if returns.empty: return 0,0,0,0,0
    
    # 1. Volatilidad Anualizada
    vol = returns.std() * np.sqrt(252)
    
    # 2. Sharpe Ratio (RF = 3%)
    mean_ret = returns.mean() * 252
    sharpe = (mean_ret - 0.03) / vol if vol > 0 else 0
    
    # 3. Sortino Ratio (Solo penaliza volatilidad negativa)
    downside = returns[returns < 0]
    downside_dev = downside.std() * np.sqrt(252)
    sortino = (mean_ret - 0.03) / downside_dev if downside_dev > 0 else 0
    
    # 4. Max Drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    
    # 5. Value at Risk (VaR 95% - Histórico)
    var_95 = np.percentile(returns, 5) # El 5% de los peores días
    
    return vol, sharpe, sortino, max_dd, var_95

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

# PRECIO ACTUAL (SIN CACHE)
def get_real_time_prices(tickers):
    if not tickers: return {}
    prices = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).fast_info
            if 'last_price' in info and info['last_price'] is not None:
                prices[t] = info['last_price']
            else:
                hist = yf.Ticker(t).history(period='2d')
                prices[t] = hist['Close'].iloc[-1] if not hist.empty else 0.0
        except:
            prices[t] = 0.0
    return prices

@st.cache_data(ttl=300)
def get_historical_data_robust(tickers):
    if not tickers: return pd.DataFrame()
    unique_tickers = list(set([str(t).strip().upper() for t in tickers] + ['SPY']))
    try:
        # Descarga con auto_adjust para dividendos/splits
        data = yf.download(unique_tickers, period="5y", interval="1d", progress=False, auto_adjust=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            if data.columns[0] == 0: data.columns = unique_tickers
        if not data.empty:
            data.index = data.index.tz_localize(None)
            data = data.fillna(method='ffill').fillna(method='bfill')
        return data
    except Exception as e:
        print(f"Error descarga: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_global_news(tickers, time_filter='d'):
    results = []
    query = f"{tickers[0]} finances" if tickers else "Stock market news"
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

# --- CARGA INICIAL ---
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
        else:
            history_data = history_raw

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0.0
patrimonio_total = total_inversiones + total_liquidez

# --- SIDEBAR ---
with st.sidebar:
    avatar = user.user_metadata.get('avatar_url', '')
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; padding:10px; background:#1C2128; border-radius:10px; border:1px solid #30363D;'>
        <img src='{avatar}' style='width:35px; border-radius:50%; border:2px solid #00CC96;'>
        <div style='line-height:1.2'><b style='color:white'>{user.user_metadata.get('full_name','Inversor')}</b><br><span style='font-size:0.7em; color:#00CC96'>● En línea</span></div>
    </div><br>""", unsafe_allow_html=True)
    
    if st.button("🔄 Actualizar Datos", use_container_width=True, type="primary"):
        clear_cache()
        st.rerun()
    
    st.divider()
    c1, c2 = st.columns([1,4])
    with c1: st.write("📰")
    with c2: st.session_state['show_news'] = st.toggle("Panel de Noticias", value=st.session_state['show_news'])
    
    pagina = st.radio("MENÚ", ["📊 Dashboard & Alpha", "💰 Liquidez (Cash)", "➕ Inversiones", "💬 Asesor AI", "🔮 Monte Carlo", "⚖️ Rebalanceo"])
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Cerrar Sesión", use_container_width=True): supabase.auth.sign_out(); st.session_state['user']=None; st.rerun()

# --- MAIN LAYOUT ---
if st.session_state['show_news']: col_main, col_news = st.columns([3.5, 1.3])
else: col_main = st.container(); col_news = st.container()

with col_main:
    
    # ------------------------------------------------------------------
    # 📊 DASHBOARD & ALPHA
    # ------------------------------------------------------------------
    if pagina == "📊 Dashboard & Alpha":
        st.title("📊 Control de Mando Integral")
        
        c_filter, c_score = st.columns([3, 1])
        with c_filter:
            # Default to 1 year back
            d_start = datetime.now() - timedelta(days=365)
            start_date = st.date_input("📅 Rango de Análisis:", value=d_start)
        
        # Calcular métricas globales
        vol = 0; sharpe = 0; sortino = 0; max_dd = 0; var_95 = 0; beta = 1.0
        
        if not history_data.empty:
            dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
            hist_filt = history_data[history_data.index >= dt_start].copy()
            
            if not hist_filt.empty:
                # Retorno diario cartera (Equiponderado simple)
                # OJO: Si tienes pesos reales, deberíamos ponderar
                port_daily = hist_filt.pct_change().mean(axis=1).dropna()
                
                if not port_daily.empty:
                    vol, sharpe, sortino, max_dd, var_95 = calculate_advanced_metrics(port_daily)
                    
                    # Formato porcentajes
                    vol *= 100
                    max_dd *= 100
                    var_95 *= 100
                    
                    # Beta
                    if not benchmark_data.empty:
                        bench = benchmark_data[benchmark_data.index >= dt_start].pct_change().dropna()
                        common = port_daily.index.intersection(bench.index)
                        if len(common) > 20:
                            cov = port_daily.loc[common].cov(bench.loc[common])
                            var_m = bench.loc[common].var()
                            beta = cov / var_m if var_m != 0 else 1.0

        with c_score:
            # Puntuación de Salud Financiera (Algoritmo simple)
            score = 50
            if sharpe > 1: score += 10
            if sharpe > 2: score += 10
            if max_dd > -15: score += 10
            if vol < 15: score += 10
            if len(df_final) > 3: score += 10
            st.metric("🛡️ Financial Health Score", f"{score}/100")

        # --- FILA 1: KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💰 Patrimonio Neto", f"{patrimonio_total:,.2f} €")
        
        if total_inversiones > 0:
            pnl = df_final['Ganancia'].sum()
            pct = (pnl / df_final['Dinero Invertido'].sum()) * 100 if df_final['Dinero Invertido'].sum() > 0 else 0
            k2.metric("📈 P&L Total", f"{pnl:+,.2f} €", f"{pct:+.2f}%", delta_color="normal" if pnl>=0 else "inverse")
        else: k2.metric("📈 P&L Total", "0.00 €")
        
        k3.metric("📉 Max Drawdown", f"{max_dd:.2f}%", delta_color="inverse", help="Peor caída en el periodo")
        k4.metric("⚖️ Ratio Sharpe", f"{sharpe:.2f}", help="Retorno ajustado al riesgo")

        st.divider()
        
        # --- FILA 2: RIESGO ---
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("⚡ Volatilidad Anual", f"{vol:.2f}%")
        r2.metric("🛡️ Ratio Sortino", f"{sortino:.2f}", help="Mejor que Sharpe, solo cuenta riesgo negativo")
        r3.metric("💣 VaR (95%) Diario", f"{var_95:.2f}%", help="Máxima pérdida esperada en 1 día con 95% confianza", delta_color="inverse")
        r4.metric("🌊 Beta vs S&P500", f"{beta:.2f}")
        
        st.divider()

        # --- GRÁFICOS ---
        c_chart, c_donut = st.columns([2, 1.2])
        with c_chart:
            st.subheader("🏁 Rendimiento Acumulado")
            if not history_data.empty:
                dt_start = pd.to_datetime(start_date).replace(tzinfo=None)
                hist_filt = history_data[history_data.index >= dt_start].copy()
                
                if not hist_filt.empty:
                    # Cartera
                    port_ret = hist_filt.pct_change().mean(axis=1).fillna(0)
                    port_cum = (1 + port_ret).cumprod() * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Tu Cartera", line=dict(color='#00CC96', width=2)))
                    
                    # Benchmark
                    if not benchmark_data.empty:
                        bench_filt = benchmark_data[benchmark_data.index >= dt_start].copy()
                        if not bench_filt.empty:
                            bench_ret = bench_filt.pct_change().fillna(0)
                            bench_cum = (1 + bench_ret).cumprod() * 100
                            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum, name="S&P 500", line=dict(color='gray', dash='dot')))
                    
                    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=0,r=0,t=20,b=0), paper_bgcolor='rgba(0,0,0,0)', 
                                      hovermode="x unified", yaxis_title="Base 100")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Datos insuficientes en el rango.")
            else: st.info("Sin histórico.")

        with c_donut:
            st.subheader("🍰 Asset Allocation")
            if patrimonio_total > 0:
                labels = ['Liquidez'] + df_final['Nombre'].tolist() if not df_final.empty else ['Liquidez']
                values = [total_liquidez] + df_final['Valor Acciones'].tolist() if not df_final.empty else [total_liquidez]
                fig_pie = px.pie(names=labels, values=values, hole=0.6, color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(template="plotly_dark", height=320, showlegend=True, margin=dict(t=0,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.info("Cartera vacía.")

        # --- CORRELACIONES ---
        st.divider()
        st.subheader("🔗 Matriz de Correlación")
        if not history_data.empty:
            corr_matrix = history_data.pct_change().corr()
            fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            fig_corr.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_corr, use_container_width=True)

    # ------------------------------------------------------------------
    # 🔮 MONTE CARLO (ALGORITMO MEJORADO)
    # ------------------------------------------------------------------
    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Simulación Monte Carlo Profesional")
        st.caption("Proyección basada en la media y volatilidad histórica REAL de tus activos, simulando 1000 escenarios posibles.")
        
        c_param, c_sim = st.columns([1, 3])
        
        with c_param:
            ys = st.slider("Años a simular", 5, 40, 20)
            sims = 1000 # Fijo para performance
            
            # Calcular parámetros reales
            mu_anual = 0.07 # Default
            sigma_anual = 0.15 # Default
            
            if not history_data.empty:
                # Media de retornos logarítmicos
                log_returns = np.log(history_data / history_data.shift(1)).dropna()
                # Peso equiponderado (simple)
                port_log_ret = log_returns.mean(axis=1)
                
                mu_anual = port_log_ret.mean() * 252
                sigma_anual = port_log_ret.std() * np.sqrt(252)
                
            st.metric("Retorno Histórico (μ)", f"{mu_anual*100:.2f}%")
            st.metric("Volatilidad Histórica (σ)", f"{sigma_anual*100:.2f}%")
            
            run_mc = st.button("🚀 Ejecutar Simulación", type="primary")

        with c_sim:
            if run_mc:
                with st.spinner("Ejecutando 1,000 universos paralelos..."):
                    dt = 1/252
                    S0 = patrimonio_total if patrimonio_total > 0 else 10000
                    
                    # VECTORIZACIÓN NUMPY (Mucho más rápido)
                    # Matriz: [dias, simulaciones]
                    days = int(ys * 252)
                    
                    # Retornos aleatorios: N( (mu - 0.5*sigma^2)*dt, sigma*sqrt(dt) )
                    drift = (mu_anual - 0.5 * sigma_anual**2) * dt
                    vol = sigma_anual * np.sqrt(dt)
                    
                    # Generar matriz de shocks aleatorios
                    Z = np.random.normal(0, 1, (days, sims))
                    daily_returns = np.exp(drift + vol * Z)
                    
                    # Caminos de precios
                    price_paths = np.zeros((days + 1, sims))
                    price_paths[0] = S0
                    
                    for t in range(1, days + 1):
                        price_paths[t] = price_paths[t-1] * daily_returns[t-1]
                    
                    # Percentiles para el Cono
                    p10 = np.percentile(price_paths, 10, axis=1)
                    p50 = np.percentile(price_paths, 50, axis=1)
                    p90 = np.percentile(price_paths, 90, axis=1)
                    
                    x_axis = np.linspace(0, ys, days + 1)
                    
                    fig = go.Figure()
                    # Área P10-P90
                    fig.add_trace(go.Scatter(x=x_axis, y=p90, mode='lines', line=dict(width=0), showlegend=False, name='P90'))
                    fig.add_trace(go.Scatter(x=x_axis, y=p10, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 204, 150, 0.2)', name='Rango 80% Probabilidad'))
                    
                    # Media
                    fig.add_trace(go.Scatter(x=x_axis, y=p50, mode='lines', line=dict(color='#00CC96', width=3), name='Escenario Base (P50)'))
                    
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                                      title="Proyección de Riqueza (Cono de Incertidumbre)", 
                                      xaxis_title="Años", yaxis_title="Valor (€)")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c_res1, c_res2, c_res3 = st.columns(3)
                    c_res1.metric("Escenario Pesimista", f"{p10[-1]:,.0f} €", help="Solo el 10% de las veces te irá peor que esto")
                    c_res2.metric("Escenario Base", f"{p50[-1]:,.0f} €", help="Lo más probable")
                    c_res3.metric("Escenario Optimista", f"{p90[-1]:,.0f} €", help="Si tienes suerte (Top 10%)")

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
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
# 🌑 DISEÑO DARK MODE PROFESIONAL (CSS v10)
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
    div[data-testid="stMetricDelta"] svg { display: none; } /* Ocultar flecha default fea */
    
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
                except: st.error("Error.")
        with tab2:
            em2 = st.text_input("Email", key="r1"); pa2 = st.text_input("Pass", type="password", key="r2")
            if st.button("Crear", type="primary", use_container_width=True):
                try: supabase.auth.sign_up({"email": em2, "password": pa2}); st.success("Creado.")
                except: st.error("Error.")
    st.stop()

user = st.session_state['user']

# --- FUNCIONES DE DATOS PRECISOS ---
def sanitize_input(text): return re.sub(r'[^\w\s\-\.]', '', str(text)).strip().upper()

@st.cache_data(ttl=60)
def get_user_data(uid):
    """Obtiene activos y liquidez de la BD"""
    assets = pd.DataFrame(supabase.table('assets').select("*").eq('user_id', uid).execute().data)
    liq_res = supabase.table('liquidity').select("*").eq('user_id', uid).execute().data
    
    if not liq_res: # Auto-crear liquidez si es usuario nuevo
        supabase.table('liquidity').insert({"user_id": uid, "name": "Principal", "amount": 0.0}).execute()
        liquidity = 0.0; liq_id = 0
    else:
        liquidity = liq_res[0]['amount']; liq_id = liq_res[0]['id']
        
    return assets, liquidity, liq_id

@st.cache_data(ttl=300) # Cache 5 min
def get_live_prices_bulk(tickers):
    """Obtiene PRECIO ACTUAL REAL para valoración"""
    if not tickers: return {}
    prices = {}
    # Añadimos SPY para benchmark
    all_tickers = list(set(tickers + ['SPY']))
    try:
        # Descarga masiva de datos recientes (último día)
        data = yf.download(all_tickers, period="5d", progress=False)['Close']
        for t in all_tickers:
            if t in data.columns:
                # Cogemos el último valor válido (el de hoy o el del viernes)
                prices[t] = data[t].dropna().iloc[-1]
            else:
                prices[t] = 0.0
    except: pass
    return prices

@st.cache_data(ttl=300)
def get_historical_data(tickers):
    """Obtiene HISTORIAL para gráficos (Adj Close)"""
    if not tickers: return pd.DataFrame()
    try:
        tickers = list(set(tickers + ['SPY']))
        data = yf.download(tickers, period="1y", interval="1d", progress=False)['Adj Close']
        # Limpieza de datos
        data.index = data.index.tz_localize(None)
        return data.fillna(method='ffill').fillna(method='bfill')
    except: return pd.DataFrame()

@st.cache_data(ttl=900)
def get_global_news(tickers, time_filter='d'):
    """Noticias con DuckDuckGo"""
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

def clear_cache(): st.cache_data.clear()

# --- PROCESAMIENTO DE DATOS ---
df_assets, total_liquidez, cash_id = get_user_data(user.id)
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()
my_tickers = []

if not df_assets.empty:
    my_tickers = df_assets['ticker'].unique().tolist()
    
    # 1. PRECIOS EN VIVO (Para KPIs y Valoración Actual)
    live_prices = get_live_prices_bulk(my_tickers)
    
    # 2. DATOS HISTÓRICOS (Para Gráficos)
    history_raw = get_historical_data(my_tickers)
    
    # Procesar Tabla Maestra
    df_assets['Precio Actual'] = df_assets['ticker'].map(live_prices)
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
    
    # Procesar Histórico
    if not history_raw.empty:
        if 'SPY' in history_raw.columns:
            benchmark_data = history_raw['SPY']
            history_data = history_raw.drop(columns=['SPY'], errors='ignore')
        else:
            history_data = history_raw

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0.0
patrimonio_total = total_inversiones + total_liquidez

# --- SIDEBAR & MENU ---
with st.sidebar:
    avatar = user.user_metadata.get('avatar_url', '')
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:10px; padding:10px; background:#21262D; border-radius:8px;'>
        <img src='{avatar}' style='width:35px; border-radius:50%; border:2px solid #00CC96;'>
        <div style='line-height:1.2'><b style='color:white'>{user.user_metadata.get('full_name','Inversor')}</b><br><span style='font-size:0.7em; color:#00CC96'>● Online</span></div>
    </div><br>""", unsafe_allow_html=True)
    
    c1, c2 = st.columns([1,4])
    with c1: st.write("📰")
    with c2: st.session_state['show_news'] = st.toggle("Noticias", value=st.session_state['show_news'])
    
    st.divider()
    pagina = st.radio("MENÚ", ["📊 Dashboard & Alpha", "💰 Liquidez (Cash)", "➕ Inversiones", "💬 Asesor AI", "🔮 Monte Carlo", "⚖️ Rebalanceo"])
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Cerrar Sesión", use_container_width=True): supabase.auth.sign_out(); st.session_state['user']=None; st.rerun()

# --- LAYOUT PRINCIPAL ---
if st.session_state['show_news']: col_main, col_news = st.columns([3.5, 1.3])
else: col_main = st.container(); col_news = st.container()

with col_main:
    
    # ------------------------------------------------------------------
    # 📊 PÁGINA DASHBOARD (MEJORADA)
    # ------------------------------------------------------------------
    if pagina == "📊 Dashboard & Alpha":
        st.title("📊 Visión Global")
        
        # Filtro de fecha para gráficos
        col_kpi, col_date = st.columns([3, 1])
        with col_date: 
            start_date = st.date_input("📅 Desde:", value=datetime.now()-timedelta(days=365))
        
        # KPIs CON COLORES
        with col_kpi:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("💰 Patrimonio Neto", f"{patrimonio_total:,.2f} €")
            
            if total_inversiones > 0:
                pnl_total = df_final['Ganancia'].sum()
                rent_total_pct = (pnl_total / df_final['Dinero Invertido'].sum()) * 100
                delta_color = "normal" if pnl_total >= 0 else "inverse"
                k2.metric("📈 Rendimiento Total", f"{pnl_total:+,.2f} €", f"{rent_total_pct:+.2f}%", delta_color=delta_color)
            else:
                k2.metric("📈 Rendimiento", "0.00 €")
            
            k3.metric("💧 Liquidez", f"{total_liquidez:,.2f} €")
            
            # Cálculo rápido de volatilidad (si hay datos)
            vol = 0
            if not history_data.empty:
                daily_ret = history_data.pct_change().dropna()
                if not daily_ret.empty:
                    # Volatilidad media de los activos
                    vol = daily_ret.std().mean() * np.sqrt(252) * 100
            k4.metric("⚡ Volatilidad Anual", f"{vol:.1f}%")

        st.divider()

        # GRÁFICO 1: RENDIMIENTO COMPARATIVO (BASE 0%)
        c_chart, c_donut = st.columns([2, 1.2])
        with c_chart:
            st.subheader("🏁 Rendimiento Relativo (%)")
            if not history_data.empty:
                # Filtrar fecha
                dt_start = pd.to_datetime(start_date)
                hist_filt = history_data[history_data.index >= dt_start].copy()
                bench_filt = benchmark_data[benchmark_data.index >= dt_start].copy()
                
                if not hist_filt.empty:
                    # Normalizar a 0%
                    fig = go.Figure()
                    
                    # 1. Cartera (Suma ponderada aproximada - Simplificación visual: Suma de precios normalizados)
                    # Para hacerlo exacto necesitaríamos historial de transacciones. 
                    # Aproximación: Usamos la media del rendimiento de los activos actuales.
                    portfolio_norm = hist_filt.apply(lambda x: (x / x.iloc[0] - 1) * 100)
                    portfolio_avg = portfolio_norm.mean(axis=1) # Promedio equiponderado de tus activos
                    
                    fig.add_trace(go.Scatter(x=portfolio_avg.index, y=portfolio_avg, name="Tus Activos (Avg)", 
                                             line=dict(color='#00CC96', width=3)))
                    
                    # 2. Benchmark (SPY)
                    if not bench_filt.empty:
                        bench_norm = (bench_filt / bench_filt.iloc[0] - 1) * 100
                        fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name="S&P 500", 
                                                 line=dict(color='gray', dash='dot')))
                    
                    fig.update_layout(template="plotly_dark", height=320, margin=dict(l=0,r=0,t=20,b=0),
                                      hovermode="x unified", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      yaxis_title="Rendimiento %")
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Selecciona un rango de fechas con datos.")
            else: st.info("Añade activos para ver el gráfico.")

        # GRÁFICO 2: DONUT (ASSET ALLOCATION REAL)
        with c_donut:
            st.subheader("🍰 Composición")
            if patrimonio_total > 0:
                # Preparamos datos
                labels = ['Liquidez'] + df_final['Nombre'].tolist() if not df_final.empty else ['Liquidez']
                values = [total_liquidez] + df_final['Valor Acciones'].tolist() if not df_final.empty else [total_liquidez]
                
                fig_pie = px.pie(names=labels, values=values, hole=0.6, 
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(template="plotly_dark", height=320, showlegend=True, 
                                      margin=dict(t=0,b=0,l=0,r=0),
                                      legend=dict(orientation="h", y=-0.1),
                                      paper_bgcolor='rgba(0,0,0,0)')
                fig_pie.update_traces(textinfo='percent')
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.info("Cartera vacía.")

        st.divider()

        # GRÁFICO 3 Y 4: DETALLES
        c_tree, c_bar = st.columns([1.5, 1.5])
        
        with c_tree:
            st.subheader("🗺️ Mapa de Calor (P&L)")
            if not df_final.empty:
                # Treemap donde el color indica si ganas o pierdes
                fig_tree = px.treemap(df_final, path=['Nombre'], values='Valor Acciones',
                                      color='Rentabilidad %', 
                                      color_continuous_scale=['#EF553B', '#1e1e1e', '#00CC96'], # Rojo - Negro - Verde
                                      color_continuous_midpoint=0)
                fig_tree.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_tree, use_container_width=True)
            else: st.info("Sin inversiones.")

        with c_bar:
            st.subheader("🏆 Ganadores vs Perdedores (€)")
            if not df_final.empty:
                df_sorted = df_final.sort_values('Ganancia', ascending=True) # De perdedor a ganador
                
                # Colores condicionales
                colors = ['#EF553B' if x < 0 else '#00CC96' for x in df_sorted['Ganancia']]
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=df_sorted['Ganancia'],
                    y=df_sorted['Nombre'],
                    orientation='h',
                    marker_color=colors,
                    text=df_sorted['Ganancia'].apply(lambda x: f"{x:,.2f}€"),
                    textposition='auto'
                ))
                
                fig_bar.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,t=0,b=0), 
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_bar, use_container_width=True)
            else: st.info("Sin inversiones.")

    # ------------------------------------------------------------------
    # 💰 PÁGINA LIQUIDEZ
    # ------------------------------------------------------------------
    elif pagina == "💰 Liquidez (Cash)":
        st.title("💰 Gestión de Liquidez")
        st.markdown(f"""
        <div style="text-align:center; padding: 40px; background-color: #21262D; border-radius: 15px; margin-bottom: 30px; border: 1px solid #30363D; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            <h3 style="color:#8B949E; margin:0; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 2px;">Saldo Disponible</h3>
            <h1 style="font-size: 4.5rem; color:#FFFFFF; margin: 10px 0; font-family: 'Inter', sans-serif;">{total_liquidez:,.2f} €</h1>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            with st.container(border=True):
                st.markdown("### 📥 Ingresar Fondos")
                a = st.number_input("Cantidad a ingresar (€)", 0.0, step=50.0, key="in")
                if st.button("Confirmar Ingreso", type="primary", use_container_width=True) and a > 0:
                    update_liquidity_balance(int(cash_id), total_liquidez + a)
                    st.toast(f"✅ Ingresados {a}€ correctamente")
                    time.sleep(1); st.rerun()
        with c2:
            with st.container(border=True):
                st.markdown("### 📤 Retirar Fondos")
                b = st.number_input("Cantidad a retirar (€)", 0.0, step=50.0, key="out")
                if st.button("Confirmar Retirada", use_container_width=True) and b > 0:
                    if b > total_liquidez: st.error("❌ Saldo insuficiente.")
                    else:
                        update_liquidity_balance(int(cash_id), total_liquidez - b)
                        st.toast(f"✅ Retirados {b}€ correctamente")
                        time.sleep(1); st.rerun()

    # ------------------------------------------------------------------
    # ➕ PÁGINA INVERSIONES
    # ------------------------------------------------------------------
    elif pagina == "➕ Inversiones":
        st.title("➕ Gestión de Activos")
        t1, t2, t3 = st.tabs(["🆕 Añadir Nuevo", "💰 Compra/Venta", "✏️ Editar/Borrar"])
        
        with t1:
            c1, c2 = st.columns([1, 1])
            with c1:
                q = st.text_input("🔍 Buscar Activo (Ticker/Nombre):")
                if st.button("Buscar") and q:
                    res = search(sanitize_input(q))
                    if 'quotes' in res: st.session_state['s'] = res['quotes']
                
                if 's' in st.session_state:
                    opts = {f"{x['symbol']} - {x.get('longname','N/A')}" : x for x in st.session_state['s']}
                    sel = st.selectbox("Selecciona Resultado:", list(opts.keys()))
                    st.session_state['sel_add'] = opts[sel]
            
            with c2:
                if 'sel_add' in st.session_state:
                    item = st.session_state['sel_add']
                    tk = item['symbol']
                    try:
                        # Precio actual rápido
                        curr_p = yf.Ticker(tk).fast_info['last_price']
                        st.success(f"Precio Actual: **{curr_p:.2f} €**")
                        
                        with st.form("add_new"):
                            st.write(f"Añadir **{tk}** a la cartera")
                            inv = st.number_input("Dinero Total Invertido (€)", 0.0)
                            val = st.number_input("Valor de Mercado Actual (€)", 0.0)
                            pl = st.selectbox("Broker", ["MyInvestor", "XTB", "Trade Republic", "Degiro", "IBKR"])
                            
                            if st.form_submit_button("Guardar Activo") and val > 0:
                                shares = val / curr_p
                                avg = inv / shares if shares > 0 else 0
                                add_asset_db(tk, item.get('longname', tk), shares, avg, pl)
                                st.balloons()
                                time.sleep(1); st.rerun()
                    except: st.error("No se pudo obtener el precio en vivo.")

        with t2:
            if df_final.empty: st.warning("Primero añade activos en la pestaña 'Añadir Nuevo'.")
            else:
                c1, c2 = st.columns([1, 1])
                with c1:
                    nom = st.selectbox("Selecciona Activo:", df_final['Nombre'].unique())
                    row = df_final[df_final['Nombre']==nom].iloc[0]
                    
                    st.info(f"""
                    **Posición Actual:**
                    - Acciones: `{row['shares']:.4f}`
                    - Precio Mercado: `{row['Precio Actual']:.2f} €`
                    - Valor Total: `{row['Valor Acciones']:.2f} €`
                    """)
                
                with c2:
                    tipo = st.radio("Operación:", ["🟢 Comprar", "🔴 Vender"], horizontal=True)
                    monto = st.number_input("Importe de la operación (€)", 0.0)
                    
                    if monto > 0:
                        precio = row['Precio Actual']
                        shares_op = monto / precio
                        
                        if "Comprar" in tipo:
                            if monto > total_liquidez: st.error("🚫 No tienes suficiente liquidez.")
                            else:
                                st.write(f"Comprarás **{shares_op:.4f}** acciones.")
                                if st.button("Confirmar Compra", type="primary"):
                                    # Nueva media ponderada
                                    new_shares = row['shares'] + shares_op
                                    new_avg = ((row['shares'] * row['avg_price']) + monto) / new_shares
                                    update_asset_db(int(row['id']), new_shares, new_avg)
                                    update_liquidity_balance(int(cash_id), total_liquidez - monto)
                                    st.success("Compra realizada"); time.sleep(1); st.rerun()
                        else:
                            st.write(f"Venderás **{shares_op:.4f}** acciones.")
                            if st.button("Confirmar Venta"):
                                new_shares = row['shares'] - shares_op
                                if new_shares < 0.001: delete_asset_db(int(row['id']))
                                else: update_asset_db(int(row['id']), new_shares, row['avg_price'])
                                update_liquidity_balance(int(cash_id), total_liquidez + monto)
                                st.success("Venta realizada"); time.sleep(1); st.rerun()

        with t3:
            if not df_final.empty:
                e_nom = st.selectbox("Editar Activo:", df_final['Nombre'], key='edit_sel')
                e_row = df_final[df_final['Nombre']==e_nom].iloc[0]
                
                c1, c2, c3 = st.columns(3)
                ns = c1.number_input("Acciones Totales", value=float(e_row['shares']), format="%.4f")
                na = c2.number_input("Precio Medio Compra", value=float(e_row['avg_price']), format="%.4f")
                
                c3.write("<br>", unsafe_allow_html=True)
                if c3.button("💾 Actualizar Datos"):
                    update_asset_db(int(e_row['id']), ns, na)
                    st.toast("Datos actualizados"); time.sleep(1); st.rerun()
                
                st.divider()
                if st.button(f"🗑️ Borrar {e_nom} definitivamente", type="secondary"):
                    delete_asset_db(int(e_row['id']))
                    st.rerun()

    # ------------------------------------------------------------------
    # 💬 ASESOR AI (Llama 3.3 Versatile)
    # ------------------------------------------------------------------
    elif pagina == "💬 Asesor AI":
        st.title("💬 Carterapro Bot (IA)")
        
        # Contexto
        ctx = f"Liquidez disponible: {total_liquidez:.2f} EUR. "
        if not df_final.empty:
            for i, r in df_final.iterrows():
                ctx += f"[{r['Nombre']}: {r['shares']:.2f} accs, Valor {r['Valor Acciones']:.2f}€, Ganancia {r['Ganancia']:.2f}€]. "
        else: ctx += "Sin inversiones."

        # Chat UI
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

        if prompt := st.chat_input("Ej: ¿Cómo ves mi diversificación?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            if HAS_GROQ:
                with st.chat_message("assistant"):
                    try:
                        client = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=st.secrets["GROQ_API_KEY"])
                        stream = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": f"Eres un gestor de patrimonios experto y directo. Tienes estos datos reales del usuario: {ctx}. Responde en español, sé útil y breve."},
                                *st.session_state.messages
                            ],
                            stream=True
                        )
                        response = st.write_stream(stream)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e: st.error(f"Error IA: {e}")
            else: st.warning("Falta API Key de Groq.")

    # ------------------------------------------------------------------
    # 🔮 MONTE CARLO
    # ------------------------------------------------------------------
    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Simulación de Futuro")
        st.caption("Proyección estocástica basada en volatilidad histórica")
        
        ys = st.slider("Años a simular", 1, 30, 10)
        
        if st.button("Ejecutar Simulación", type="primary"):
            # Parámetros por defecto
            mu, sigma = 0.07, 0.15 
            
            # Intentar calcular reales si hay histórico
            if not history_data.empty:
                daily_returns = history_data.pct_change().dropna()
                # Peso equiponderado simple para estimar volatilidad de cartera
                portfolio_ret = daily_returns.mean(axis=1)
                mu = portfolio_ret.mean() * 252
                sigma = portfolio_ret.std() * np.sqrt(252)
            
            start_val = patrimonio_total
            dt = 1/252
            paths = []
            
            for _ in range(50): # 50 escenarios
                prices = [start_val]
                for _ in range(int(ys*252)):
                    shock = np.random.normal(0, 1)
                    price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)
                    prices.append(price)
                paths.append(prices)
            
            # Graficar
            fig = go.Figure()
            x_axis = np.linspace(0, ys, len(paths[0]))
            
            for p in paths:
                fig.add_trace(go.Scatter(x=x_axis, y=p, mode='lines', line=dict(color='rgba(0, 204, 150, 0.1)'), showlegend=False))
            
            avg_path = np.mean(paths, axis=0)
            fig.add_trace(go.Scatter(x=x_axis, y=avg_path, mode='lines', name='Escenario Medio', line=dict(color='#00CC96', width=3)))
            
            fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', 
                              title=f"Proyección a {ys} años (Retorno Esp: {mu*100:.1f}%, Vol: {sigma*100:.1f}%)",
                              xaxis_title="Años", yaxis_title="Valor de Cartera (€)")
            st.plotly_chart(fig, use_container_width=True)
            
            final_val = avg_path[-1]
            st.success(f"En el escenario medio, tendrías: **{final_val:,.2f} €**")

    # ------------------------------------------------------------------
    # ⚖️ REBALANCEO
    # ------------------------------------------------------------------
    elif pagina == "⚖️ Rebalanceo":
        st.title("⚖️ Calculadora de Rebalanceo")
        
        if df_final.empty: st.warning("Añade activos primero.")
        else:
            c1, c2 = st.columns([1, 1.5])
            with c1:
                target_weights = {}
                total_target = 0
                st.write("Define el % objetivo para cada activo:")
                for i, r in df_final.iterrows():
                    w = st.number_input(f"{r['Nombre']} (%)", 0, 100, int(r['Peso %']), key=f"w_{i}")
                    target_weights[r['Nombre']] = w
                    total_target += w
                
                st.metric("Total Asignado", f"{total_target}%", delta="OK" if total_target==100 else "Ajustar", delta_color="normal" if total_target==100 else "inverse")
            
            with c2:
                if total_target == 100 and st.button("Calcular Ajustes", type="primary"):
                    # Algoritmo simple de aportación para rebalancear
                    # Asumimos que queremos subir el valor total hasta que el activo más pasado de peso encaje
                    # O simplemente rebalanceamos el patrimonio actual (Vender para comprar)
                    
                    # Estrategia: Rebalanceo sobre patrimonio total actual
                    moves = []
                    for i, r in df_final.iterrows():
                        target_val = patrimonio_total * (target_weights[r['Nombre']] / 100)
                        diff = target_val - r['Valor Acciones']
                        moves.append({'Activo': r['Nombre'], 'Movimiento': diff})
                    
                    df_moves = pd.DataFrame(moves)
                    
                    # Gráfico
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df_moves['Movimiento'],
                        y=df_moves['Activo'],
                        orientation='h',
                        marker_color=['#00CC96' if x > 0 else '#EF553B' for x in df_moves['Movimiento']],
                        text=df_moves['Movimiento'].apply(lambda x: f"{x:+,.2f}€"),
                        textposition='auto'
                    ))
                    fig.update_layout(template="plotly_dark", title="Movimientos necesarios (Vender/Comprar)", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 Barras Verdes: Debes COMPRAR esa cantidad. Barras Rojas: Debes VENDER esa cantidad.")

# --- BOT NOTICIAS (FIXED SIDE) ---
if st.session_state['show_news']:
    with col_news:
        c1, c2 = st.columns([3, 1])
        with c1: st.markdown("### 🤖 News")
        with c2: 
            if st.button("🔄", key="ref"): clear_cache(); st.rerun()
        
        tm = {'Hoy': 'd', 'Semana': 'w'}; sel = st.pills("Filtro", list(tm.keys()), default="Hoy", label_visibility="collapsed")
        
        news = get_global_news(my_tickers, tm[sel])
        html = ""
        if news:
            for n in news:
                img = f"<img src='{n['image']}' class='news-img'/>" if n.get('image') else ""
                html += f"""
                <div class="news-card">
                    {img}
                    <div class="news-source">{n.get('source','Web')} • {n.get('date','')}</div>
                    <div class="news-title"><a href="{n['url']}" target="_blank">{n['title']}</a></div>
                </div>"""
        else: html = "<div style='text-align:center;color:#666;padding:20px'>💤 Sin noticias recientes</div>"
        
        st.markdown(f"<div class='news-scroll-area'>{html}</div>", unsafe_allow_html=True)
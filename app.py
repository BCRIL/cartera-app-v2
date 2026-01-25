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

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial Ultra", layout="wide", page_icon="🏦")

# --- ESTILOS CSS (ESTILO CHAT TWITCH) ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; border: 1px solid #e0e0e0;}
    .stMetric {text-align: center;}
    div[data-testid="stMetricLabel"] > div > div {cursor: help;}
    div[role="tablist"] {justify-content: center;}
    
    /* Estilo Chat de Noticias */
    .news-container {
        height: 80vh;
        overflow-y: auto;
        padding-right: 10px;
        border-left: 1px solid #ddd;
        padding-left: 15px;
    }
    .news-bubble {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        border: 1px solid #eee;
        font-size: 0.9em;
    }
    .news-time { font-size: 0.75em; color: #888; }
    .news-source { font-size: 0.75em; font-weight: bold; color: #555; }
    .sentiment-pos { border-left: 4px solid #00cc96; }
    .sentiment-neg { border-left: 4px solid #ef553b; }
    .sentiment-neu { border-left: 4px solid #gray; }
    
    /* Animación suave */
    .stApp { transition: all 0.3s ease-in-out; }
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
if 'show_news' not in st.session_state: st.session_state['show_news'] = True # Por defecto activado

# ==============================================================================
# 🔄 LOGIN
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
        st.title("🏦 Carterapro Ultra")
        st.caption("Gestión Patrimonial Personal")
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
                except Exception as e: st.error("Error de credenciales.")
        with tab_signup:
            r_email = st.text_input("Email", key="r_em")
            r_pass = st.text_input("Contraseña", type="password", key="r_pa")
            if st.button("Crear Cuenta", type="primary", use_container_width=True):
                try:
                    res = supabase.auth.sign_up({"email": r_email, "password": r_pass})
                    st.success("Cuenta creada. Revisa tu correo.")
                except Exception as e: st.error(f"Error: {e}")
    st.stop()

user = st.session_state['user']

# --- SIDEBAR ---
with st.sidebar:
    st.image(user.user_metadata.get('avatar_url', 'https://cdn-icons-png.flaticon.com/512/3135/3135715.png'), width=60)
    st.markdown(f"### {user.user_metadata.get('full_name', 'Inversor')}")
    
    # Toggle de Noticias estilo Twitch
    st.divider()
    col_tog1, col_tog2 = st.columns([1, 4])
    with col_tog1:
        st.write("📰")
    with col_tog2:
        if st.toggle("Chat de Mercado", value=st.session_state['show_news']):
            st.session_state['show_news'] = True
        else:
            st.session_state['show_news'] = False
            
    st.divider()
    pagina = st.radio("Navegación", [
        "📊 Dashboard & Alpha", 
        "💰 Liquidez (Cash)",
        "➕ Gestión de Inversiones",
        "🤖 Asesor de Riesgos", 
        "🔮 Monte Carlo", 
        "⚖️ Rebalanceo"
    ])
    st.divider()
    if st.button("Cerrar Sesión"):
        supabase.auth.sign_out(); st.session_state['user'] = None; st.rerun()

# --- FUNCIONES MATEMÁTICAS & UTILIDADES ---
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

def get_sentiment(text):
    blob = TextBlob(text)
    score = blob.sentiment.polarity
    if score > 0.1: return "pos", "🟢"
    elif score < -0.1: return "neg", "🔴"
    return "neu", "⚪"

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

@st.cache_data(ttl=900) # Cachear noticias 15 min
def get_news_feed(tickers):
    news_list = []
    # Añadir siempre SPY y BTC para tener algo de chicha
    search_list = list(set(tickers + ['SPY', 'BTC-USD']))[:5] # Max 5 tickers para no saturar
    
    for t in search_list:
        try:
            stock = yf.Ticker(t)
            news = stock.news
            if news:
                for n in news[:2]: # 2 noticias por activo
                    pub = datetime.fromtimestamp(n['providerPublishTime'])
                    css_class, icon = get_sentiment(n['title'])
                    news_list.append({
                        'title': n['title'],
                        'publisher': n['publisher'],
                        'link': n['link'],
                        'time': pub,
                        'ticker': t,
                        'class': css_class,
                        'icon': icon
                    })
        except: pass
    
    # Ordenar por fecha (más reciente primero)
    news_list.sort(key=lambda x: x['time'], reverse=True)
    return news_list

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

# --- CARGA DE DATOS ---
df_assets = get_assets_db(user.id)
df_cash = get_liquidity_db(user.id)

if not df_cash.empty:
    cash_row = df_cash.iloc[0]
    total_liquidez = cash_row['amount']
    cash_id = cash_row.get('id', None)
else: total_liquidez = 0.0; cash_id = None

df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()
news_items = []

tickers = []
if not df_assets.empty:
    tickers = df_assets['ticker'].unique().tolist()
    data_raw = get_market_data(tickers)
    
    # Obtener Noticias del Bot
    news_items = get_news_feed(tickers)
    
    if not data_raw.empty:
        if 'SPY' in data_raw.columns:
            benchmark_data = data_raw['SPY']
            history = data_raw.drop(columns=['SPY', 'GLD'], errors='ignore')
        else: history = data_raw
        history_data = history
        
        prices_dict = {}; rsi_dict = {}; metrics_dict = {}
        for t in tickers:
            if t in history.columns:
                s = history[t]
                prices_dict[t] = s.iloc[-1]
                if len(s) > 15:
                    delta = s.diff()
                    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
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
else:
    # Si no hay activos, buscamos noticias generales
    news_items = get_news_feed(['SPY', 'BTC-USD', 'EURUSD=X'])

total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0
patrimonio_total = total_inversiones + total_liquidez

# ==============================================================================
# LAYOUT PRINCIPAL (COLUMNAS DINÁMICAS PARA CHAT)
# ==============================================================================

# Si el chat está activo, dividimos la pantalla. Si no, columna única.
if st.session_state['show_news']:
    col_main, col_news = st.columns([3.5, 1.2]) # 75% App / 25% Chat
else:
    col_main = st.container()
    col_news = st.container() # Vacío si oculto

# === CONTENIDO PRINCIPAL (IZQUIERDA) ===
with col_main:
    
    if pagina == "📊 Dashboard & Alpha":
        st.title("📊 Visión Global")
        col_kpi, col_date = st.columns([3, 1])
        with col_date: start_date = st.date_input("📅 Rendimiento desde:", value=datetime.now()-timedelta(days=180))
        with col_kpi:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Patrimonio NETO", f"{patrimonio_total:,.2f} €")
            if total_inversiones > 0:
                ganancia_inv = df_final['Ganancia'].sum()
                rent_inv = (ganancia_inv / df_final['Dinero Invertido'].sum() * 100)
                c2.metric("📈 P&L Inversión", f"{ganancia_inv:+,.2f} €", f"{rent_inv:+.2f}%")
            else: c2.metric("📈 P&L Inversión", "0 €")
            c3.metric("💧 Liquidez", f"{total_liquidez:,.2f} €")
            vol_avg = df_final['Volatilidad'].mean() if not df_final.empty else 0
            c4.metric("⚡ Volatilidad", f"{vol_avg:.1f}%")

        st.divider()
        c_chart, c_pie = st.columns([2, 1.2])
        with c_chart:
            st.subheader("🏁 Rendimiento vs Mercado")
            if not history_data.empty and not benchmark_data.empty:
                start_dt = pd.to_datetime(start_date).tz_localize(None)
                h_filt = history_data[history_data.index >= start_dt].sum(axis=1)
                b_filt = benchmark_data[benchmark_data.index >= start_dt]
                common = h_filt.index.intersection(b_filt.index)
                if len(common) > 2:
                    norm_h = h_filt.loc[common] / h_filt.loc[common].iloc[0] * 100
                    norm_b = b_filt.loc[common] / b_filt.loc[common].iloc[0] * 100
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=norm_h.index, y=norm_h, name="Tu Cartera", line=dict(color='#00CC96', width=2)))
                    fig.add_trace(go.Scatter(x=norm_b.index, y=norm_b, name="S&P 500", line=dict(color='gray', dash='dot')))
                    fig.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified", legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig, use_container_width=True)
                else: st.info("Datos insuficientes.")
            else: st.info("Falta historial.")

        with c_pie:
            st.subheader("🍰 Distribución Real")
            if patrimonio_total > 0:
                df_chart = df_final[['Nombre', 'Valor Acciones']].copy() if not df_final.empty else pd.DataFrame(columns=['Nombre', 'Valor Acciones'])
                if total_liquidez > 0:
                    new_row = pd.DataFrame([{'Nombre': '💧 Liquidez', 'Valor Acciones': total_liquidez}])
                    df_chart = pd.concat([df_chart, new_row], ignore_index=True)
                fig_pie = px.pie(df_chart, values='Valor Acciones', names='Nombre', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
                fig_pie.update_layout(height=320, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else: st.info("Cartera vacía.")

        st.divider()
        c_tree, c_bar = st.columns([1.5, 1.5])
        with c_tree:
            st.subheader("🗺️ Mapa de Calor")
            if not df_final.empty:
                fig_tree = px.treemap(df_final, path=['Nombre'], values='Valor Acciones', color='Rentabilidad', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
                fig_tree.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig_tree, use_container_width=True)
            else: st.info("Sin inversiones.")
        with c_bar:
            st.subheader("🏆 Ranking (€)")
            if not df_final.empty:
                df_sorted = df_final.sort_values('Ganancia', ascending=False)
                fig_bar = px.bar(df_sorted, x='Ganancia', y='Nombre', orientation='h', color='Ganancia', color_continuous_scale='RdYlGn', text_auto='.2s')
                fig_bar.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else: st.info("Sin inversiones.")

    elif pagina == "💰 Liquidez (Cash)":
        st.title("💰 Mi Liquidez")
        st.markdown(f"""
        <div style="text-align:center; padding: 40px; background-color: #e8f5e9; border-radius: 15px; margin-bottom: 30px;">
            <h3 style="color:#2e7d32; margin:0;">Saldo Total Disponible</h3>
            <h1 style="font-size: 60px; color:#1b5e20; margin:0;">{total_liquidez:,.2f} €</h1>
        </div>
        """, unsafe_allow_html=True)
        c_in, c_out = st.columns(2)
        with c_in:
            with st.expander("📥 INGRESAR DINERO", expanded=True):
                add_amt = st.number_input("Ingreso (€)", 0.0, step=50.0, key="add")
                if st.button("Confirmar Ingreso", type="primary", use_container_width=True):
                    if add_amt > 0:
                        current_db = get_liquidity_db(user.id)
                        cid = current_db.iloc[0]['id']; cbal = current_db.iloc[0]['amount']
                        update_liquidity_balance(int(cid), cbal + add_amt)
                        st.success(f"Añadidos {add_amt}€"); time.sleep(0.5); st.rerun()
        with c_out:
            with st.expander("📤 RETIRAR DINERO", expanded=True):
                sub_amt = st.number_input("Retirada (€)", 0.0, step=50.0, key="sub")
                if st.button("Confirmar Retirada", type="secondary", use_container_width=True):
                    if sub_amt > 0:
                        current_db = get_liquidity_db(user.id)
                        cid = current_db.iloc[0]['id']; cbal = current_db.iloc[0]['amount']
                        if sub_amt > cbal: st.error("Saldo insuficiente.")
                        else:
                            update_liquidity_balance(int(cid), cbal - sub_amt)
                            st.success(f"Retirados {sub_amt}€"); time.sleep(0.5); st.rerun()

    elif pagina == "➕ Gestión de Inversiones":
        st.title("➕ Inversiones")
        t1, t2, t3 = st.tabs(["🆕 Añadir", "💰 Operar", "✏️ Editar"])
        with t1:
            c1, c2 = st.columns(2)
            with c1:
                q = st.text_input("Buscar:", placeholder="ISIN o Nombre...")
                if st.button("🔍") and q:
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
                            with st.form("new"):
                                inv = st.number_input("Invertido (€)", 0.0)
                                val = st.number_input("Valor (€)", 0.0)
                                pl = st.selectbox("Broker", ["MyInvestor", "XTB", "TR", "Degiro"])
                                if st.form_submit_button("Guardar") and val > 0:
                                    shares = val / cp
                                    avg_p = inv / shares if shares > 0 else 0
                                    add_asset_db(t, a.get('longname', t), shares, avg_p, pl)
                                    st.success("Guardado."); time.sleep(0.5); st.rerun()
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
                            if amt > total_liquidez: st.error("Sin liquidez.")
                            else:
                                st.success(f"Comprarás {sh:.4f} accs")
                                if st.button("Confirmar"):
                                    navg = ((r['shares']*r['avg_price'])+amt)/(r['shares']+sh)
                                    update_asset_db(int(r['id']), r['shares']+sh, navg)
                                    if cash_id: update_liquidity_balance(int(cash_id), total_liquidez - amt)
                                    st.rerun()
                        else:
                            st.warning(f"Venderás {sh:.4f} accs")
                            if st.button("Confirmar"):
                                nsh = r['shares'] - sh
                                if nsh <= 0.001: delete_asset_db(int(r['id']))
                                else: update_asset_db(int(r['id']), nsh, r['avg_price'])
                                if cash_id: update_liquidity_balance(int(cash_id), total_liquidez + amt)
                                st.rerun()
        with t3:
            if not df_final.empty:
                ed = st.selectbox("Editar", df_final['Nombre'], key='ed')
                re = df_final[df_final['Nombre']==ed].iloc[0]
                c1, c2, c3 = st.columns(3)
                ns = c1.number_input("Accs", value=float(re['shares']), format="%.4f")
                na = c2.number_input("Media", value=float(re['avg_price']), format="%.4f")
                if c3.button("Actualizar"): update_asset_db(int(re['id']), ns, na); st.rerun()
                if st.button(f"Borrar {ed}"): delete_asset_db(int(re['id'])); st.rerun()

    elif pagina == "🤖 Asesor de Riesgos":
        st.title("🤖 Diagnóstico IA")
        if df_final.empty: st.stop()
        c1, c2 = st.columns([1,1])
        with c1:
            msgs = []; beta = 1.0
            if not history_data.empty and not benchmark_data.empty:
                try:
                    m_sum = history_data.sum(axis=1).pct_change().dropna()
                    b_ret = benchmark_data.pct_change().dropna()
                    idx = m_sum.index.intersection(b_ret.index)
                    if len(idx)>20: beta = m_sum.loc[idx].cov(b_ret.loc[idx])/b_ret.loc[idx].var()
                except: pass
            st.metric("Beta", f"{beta:.2f}")
            pc = (total_liquidez / patrimonio_total * 100) if patrimonio_total > 0 else 0
            if pc > 40: msgs.append("ℹ️ Mucha liquidez.")
            if beta > 1.3: msgs.append("🚨 Cartera agresiva.")
            for m in msgs: st.info(m)
            if not msgs: st.success("Cartera equilibrada.")
        with c2:
            if len(tickers)>1 and not history_data.empty:
                sns.heatmap(history_data.corr(), annot=True, cmap='coolwarm'); st.pyplot(plt)

    elif pagina == "🔮 Monte Carlo":
        st.title("🔮 Futuro")
        if df_final.empty: st.stop()
        c1, c2 = st.columns([1,2])
        with c1:
            ys = st.slider("Años", 1, 30, 10)
            mu, sigma = 0.07, 0.15
            if not history_data.empty:
                ds = history_data.sum(axis=1).replace(0, np.nan).dropna()
                if len(ds)>10: mu = ds.pct_change().mean()*252; sigma = ds.pct_change().std()*np.sqrt(252)
            if np.isnan(mu): mu=0.07
            if np.isnan(sigma) or sigma==0: sigma=0.15
            st.metric("Retorno", f"{mu*100:.1f}%"); st.metric("Volatilidad", f"{sigma*100:.1f}%")
            run = st.button("Simular")
        with c2:
            if run:
                paths = []
                for _ in range(50):
                    p = [total_inversiones]
                    for _ in range(int(ys*252)):
                        p.append(p[-1] * np.exp((mu - 0.5*sigma**2)*(1/252) + sigma*np.sqrt(1/252)*np.random.normal(0,1)))
                    paths.append(p)
                fig = go.Figure()
                for p in paths: fig.add_trace(go.Scatter(y=p, mode='lines', line=dict(color='rgba(0,100,200,0.1)'), showlegend=False))
                fig.add_trace(go.Scatter(y=np.mean(paths, axis=0), mode='lines', line=dict(color='blue', width=3)))
                st.plotly_chart(fig, use_container_width=True)

    elif pagina == "⚖️ Rebalanceo":
        st.title("⚖️ Rebalanceo")
        if df_final.empty: st.stop()
        c1, c2 = st.columns([1,2])
        with c1:
            pl = st.number_input("Meses", 1, 24, 6)
            ws = {}; tot = 0
            for i, r in df_final.iterrows():
                w = st.number_input(f"{r['ticker']} %", 0, 100, int(r['Peso %']), key=i)
                ws[r['Nombre']] = w; tot += w
            st.metric("Total", f"{tot}%")
            calc = st.button("Calcular")
        with c2:
            if calc and tot==100:
                pat = df_final['Valor Acciones'].sum()
                mc = max([r['Valor Acciones']/(ws[r['Nombre']]/100) for i,r in df_final.iterrows() if ws[r['Nombre']]>0]+[pat])
                st.metric("Aportación", f"{(mc-pat)/pl:,.2f} €")
                dat = [{'Activo':r['Nombre'], 'Mes':max(0, (mc*ws[r['Nombre']]/100)-r['Valor Acciones'])/pl} for i,r in df_final.iterrows()]
                st.plotly_chart(px.bar(pd.DataFrame(dat), x='Activo', y='Mes'), use_container_width=True)

# === CHAT LATERAL DERECHO (ESTILO TWITCH) ===
if st.session_state['show_news']:
    with col_news:
        st.markdown("<div class='news-container'>", unsafe_allow_html=True)
        st.subheader("🤖 MarketBot")
        
        # Noticias en burbujas
        if news_items:
            for n in news_items:
                st.markdown(f"""
                <div class="news-bubble sentiment-{n['class']}">
                    <div class="news-source">{n['icon']} {n['ticker']} • {n['publisher']}</div>
                    <div style="font-weight:600; margin: 5px 0;"><a href="{n['link']}" style="text-decoration:none; color:#333;">{n['title']}</a></div>
                    <div class="news-time">{n['time'].strftime('%H:%M')}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No hay noticias relevantes ahora mismo.")
            
        st.markdown("</div>", unsafe_allow_html=True)
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

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; border: 1px solid #e0e0e0;}
    .stMetric {text-align: center;}
    div[data-testid="stMetricLabel"] > div > div {cursor: help;}
    div[role="tablist"] {justify-content: center;}
    .big-font {font-size:24px !important; font-weight: bold; color: #2E7D32;}
    
    /* Animación suave al cargar */
    .stApp {
        transition: all 0.3s ease-in-out;
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
        if st.button("🇬 Iniciar con Google", type="primary", use_container_width=True):
            try:
                data = supabase.auth.sign_in_with_oauth({
                    "provider": "google",
                    "options": {"redirect_to": "https://carterapro.streamlit.app"}
                })
                st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
            except Exception as e: st.error(f"Error: {e}")
        st.divider()
        email = st.text_input("Email")
        password = st.text_input("Contraseña", type="password")
        if st.button("Entrar"):
            try:
                res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                st.session_state['user'] = res.user
                st.rerun()
            except Exception as e: st.error(f"Error: {e}")
    st.stop()

user = st.session_state['user']

# --- SIDEBAR ---
with st.sidebar:
    st.image(user.user_metadata.get('avatar_url', 'https://cdn-icons-png.flaticon.com/512/3135/3135715.png'), width=60)
    st.markdown(f"### {user.user_metadata.get('full_name', 'Inversor')}")
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

def get_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.1: return "🟢"
    elif blob.sentiment.polarity < -0.1: return "🔴"
    return "⚪"

def sanitize_input(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[^\w\s\-\.]', '', text).strip().upper()

# --- FUNCIONES CACHEADAS (LA CLAVE DE LA VELOCIDAD) ---

@st.cache_data(ttl=60) # Cachear datos de DB por 60 segundos
def get_assets_db(uid):
    try: return pd.DataFrame(supabase.table('assets').select("*").eq('user_id', uid).execute().data)
    except: return pd.DataFrame()

@st.cache_data(ttl=60) # Cachear datos de DB por 60 segundos
def get_liquidity_db(uid):
    """Obtiene la liquidez. Si no existe, crea una entrada 'Principal' a 0"""
    try:
        response = supabase.table('liquidity').select("*").eq('user_id', uid).execute()
        data = response.data
        if data:
            return pd.DataFrame(data)
        else:
            # Autocrear cartera de liquidez si no existe
            new_data = {"user_id": uid, "name": "Principal", "amount": 0.0}
            supabase.table('liquidity').insert(new_data).execute()
            return pd.DataFrame([new_data]) 
    except:
        return pd.DataFrame(columns=['id', 'user_id', 'name', 'amount', 'created_at'])

@st.cache_data(ttl=300) # Cachear Mercado por 5 minutos (Lo más lento)
def get_market_data(tickers):
    if not tickers: return pd.DataFrame()
    try:
        tickers_api = list(set(tickers + ['SPY', 'GLD'])) 
        data = yf.download(tickers_api, period="1y", progress=False)['Close']
        data.index = data.index.tz_localize(None)
        data = data.fillna(method='ffill').fillna(method='bfill')
        return data
    except: return pd.DataFrame()

# --- FUNCIONES DE ESCRITURA (LIMPIAN CACHÉ) ---
def clear_cache():
    st.cache_data.clear()

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

# --- PRE-PROCESAMIENTO RÁPIDO ---
df_assets = get_assets_db(user.id)
df_cash = get_liquidity_db(user.id)

# Procesar Liquidez
if not df_cash.empty:
    cash_row = df_cash.iloc[0]
    total_liquidez = cash_row['amount']
    cash_id = cash_row.get('id', None)
else:
    total_liquidez = 0.0
    cash_id = None

# Procesar Inversiones
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()

if not df_assets.empty:
    tickers = df_assets['ticker'].unique().tolist()
    data_raw = get_market_data(tickers) # Usamos la versión cacheada
    
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

# Totales Globales
total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0
patrimonio_total = total_inversiones + total_liquidez

# ==============================================================================
# 📊 PÁGINA 1: DASHBOARD
# ==============================================================================
if pagina == "📊 Dashboard & Alpha":
    st.title("📊 Control de Mando Integral")
    
    col_kpi, col_date = st.columns([3, 1])
    with col_date:
        start_date = st.date_input("📅 Desde:", value=datetime.now()-timedelta(days=180))
    
    with col_kpi:
        c1, c2, c3 = st.columns(3)
        c1.metric("💰 Patrimonio NETO", f"{patrimonio_total:,.2f} €", help="Inversiones + Liquidez")
        
        if total_inversiones > 0:
            ganancia_inv = df_final['Ganancia'].sum()
            rent_inv = (ganancia_inv / df_final['Dinero Invertido'].sum() * 100)
            c2.metric("📈 Rendimiento Inversión", f"{ganancia_inv:+,.2f} €", f"{rent_inv:+.2f}%")
        else: c2.metric("📈 Rendimiento", "0 €")
        
        pct_cash = (total_liquidez / patrimonio_total * 100) if patrimonio_total > 0 else 0
        c3.metric("💧 Liquidez Total", f"{total_liquidez:,.2f} €", f"{pct_cash:.1f}% del total")

    st.divider()
    
    c_chart, c_pie = st.columns([2, 1])
    with c_chart:
        st.subheader("🏁 Inversiones vs Mercado")
        if not history_data.empty and not benchmark_data.empty:
            start_dt = pd.to_datetime(start_date).tz_localize(None)
            h_filt = history_data[history_data.index >= start_dt].sum(axis=1)
            b_filt = benchmark_data[benchmark_data.index >= start_dt]
            
            common = h_filt.index.intersection(b_filt.index)
            if len(common) > 2:
                norm_h = h_filt.loc[common] / h_filt.loc[common].iloc[0] * 100
                norm_b = b_filt.loc[common] / b_filt.loc[common].iloc[0] * 100
                
                fig = go.Figure()
                # Animación suave en el gráfico
                fig.add_trace(go.Scatter(x=norm_h.index, y=norm_h, name="Tu Cartera", line=dict(color='#00CC96', width=2)))
                fig.add_trace(go.Scatter(x=norm_b.index, y=norm_b, name="S&P 500", line=dict(color='gray', dash='dot')))
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified", transition_duration=500)
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("Datos insuficientes.")
        else: st.info("Añade inversiones para ver el gráfico.")

    with c_pie:
        st.subheader("Asset Allocation")
        labels = ['Inversiones', 'Liquidez']
        values = [total_inversiones, total_liquidez]
        fig_alloc = px.pie(names=labels, values=values, hole=0.6, color_discrete_sequence=['#00CC96', '#636EFA'])
        fig_alloc.update_layout(height=300, showlegend=True, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_alloc, use_container_width=True)

# ==============================================================================
# 💰 PÁGINA 2: LIQUIDEZ (SIMPLIFICADA MÁXIMA)
# ==============================================================================
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
        with st.expander("📥 AÑADIR DINERO (Ingreso)", expanded=True):
            add_amt = st.number_input("Cantidad a ingresar (€)", min_value=0.0, step=50.0, key="add_cash")
            if st.button("Confirmar Ingreso", type="primary", use_container_width=True):
                if add_amt > 0:
                    current_db = get_liquidity_db(user.id)
                    cid = current_db.iloc[0]['id']
                    cbal = current_db.iloc[0]['amount']
                    update_liquidity_balance(int(cid), cbal + add_amt)
                    st.success(f"Has añadido {add_amt} €")
                    time.sleep(0.5); st.rerun()

    with c_out:
        with st.expander("📤 RETIRAR DINERO (Gasto)", expanded=True):
            sub_amt = st.number_input("Cantidad a retirar (€)", min_value=0.0, step=50.0, key="sub_cash")
            if st.button("Confirmar Retirada", type="secondary", use_container_width=True):
                if sub_amt > 0:
                    current_db = get_liquidity_db(user.id)
                    cid = current_db.iloc[0]['id']
                    cbal = current_db.iloc[0]['amount']
                    if sub_amt > cbal: st.error("No tienes suficiente saldo.")
                    else:
                        update_liquidity_balance(int(cid), cbal - sub_amt)
                        st.success(f"Has retirado {sub_amt} €")
                        time.sleep(0.5); st.rerun()

# ==============================================================================
# ➕ PÁGINA 3: GESTIÓN DE INVERSIONES (CONECTADA A LIQUIDEZ)
# ==============================================================================
elif pagina == "➕ Gestión de Inversiones":
    st.title("➕ Inversiones (Stocks/ETFs)")
    
    t_new, t_trade, t_edit = st.tabs(["🆕 Añadir Nuevo", "💰 Operar (Compra/Venta)", "✏️ Editar"])
    
    with t_new:
        st.caption("Usa esto para registrar activos que YA tienes de antes. No afectará a tu liquidez.")
        c1, c2 = st.columns(2)
        with c1:
            q = st.text_input("Buscar:", placeholder="ISIN o Nombre...")
            if st.button("🔍") and q:
                r = search(sanitize_input(q))
                if 'quotes' in r: st.session_state['s'] = r['quotes']
            if 's' in st.session_state:
                opts = {f"{x['symbol']} | {x.get('longname','N/A')}" : x for x in st.session_state['s']}
                sel = st.selectbox("Resultados:", list(opts.keys()))
                st.session_state['sel_add'] = opts[sel]
        with c2:
            if 'sel_add' in st.session_state:
                a = st.session_state['sel_add']
                t = a['symbol']
                try:
                    inf = yf.Ticker(t).history(period='1d')
                    if not inf.empty:
                        cp = inf['Close'].iloc[-1]
                        st.metric("Precio", f"{cp:.2f} €")
                        with st.form("new"):
                            inv = st.number_input("Invertido Total (€)", min_value=0.0)
                            val = st.number_input("Valor Actual Total (€)", min_value=0.0)
                            pl = st.selectbox("Broker", ["MyInvestor", "XTB", "TR", "Degiro"])
                            if st.form_submit_button("Guardar Registro") and val > 0:
                                shares = val / cp
                                avg_p = inv / shares if shares > 0 else 0
                                add_asset_db(t, a.get('longname', t), shares, avg_p, pl)
                                st.success("Registrado."); time.sleep(0.5); st.rerun()
                except: st.error("Error precio.")

    with t_trade:
        if df_final.empty: st.warning("Añade inversiones primero.")
        else:
            c_sel, c_ops = st.columns([1, 2])
            with c_sel:
                nom = st.selectbox("Activo", df_final['Nombre'].unique())
                row = df_final[df_final['Nombre'] == nom].iloc[0]
                cur_shares = row['shares']
                cur_avg = row['avg_price']
                st.info(f"Posición: **{cur_shares:.4f}** accs a **{cur_avg:.2f}€**.")

            with c_ops:
                live_price = row['Precio Actual']
                st.metric("Precio Mercado", f"{live_price:.2f} €")
                
                op = st.radio("Acción", ["🟢 Compra", "🔴 Venta"], horizontal=True)
                amt = st.number_input("Importe Operación (€)", 0.0, step=50.0)
                
                st.markdown(f"**Saldo Liquidez:** {total_liquidez:,.2f} €")
                
                if amt > 0:
                    sh_imp = amt / live_price
                    if "Compra" in op:
                        if amt > total_liquidez: st.error("❌ No tienes suficiente liquidez.")
                        else:
                            st.success(f"Comprarás {sh_imp:.4f} accs. Se restarán {amt}€ de liquidez.")
                            if st.button("✅ Confirmar Compra"):
                                new_sh = cur_shares + sh_imp
                                new_avg = ((cur_shares*cur_avg)+amt)/new_sh
                                update_asset_db(int(row['id']), new_sh, new_avg)
                                if cash_id: update_liquidity_balance(int(cash_id), total_liquidez - amt)
                                st.success("Compra realizada."); time.sleep(0.5); st.rerun()
                    else: 
                        if amt > (cur_shares * live_price): st.error("No tienes tantas acciones.")
                        else:
                            st.warning(f"Venderás {sh_imp:.4f} acciones. Se añadirán {amt}€ a liquidez.")
                            if st.button("🚨 Confirmar Venta"):
                                new_sh = cur_shares - sh_imp
                                if new_sh <= 0.001: delete_asset_db(int(row['id']))
                                else: update_asset_db(int(row['id']), new_sh, cur_avg)
                                if cash_id: update_liquidity_balance(int(cash_id), total_liquidez + amt)
                                st.success("Venta realizada."); time.sleep(0.5); st.rerun()

    with t_edit:
        if not df_final.empty:
            ed = st.selectbox("Editar:", df_final['Nombre'], key='ed')
            r_ed = df_final[df_final['Nombre']==ed].iloc[0]
            c1, c2, c3 = st.columns(3)
            ns = c1.number_input("Acciones", value=float(r_ed['shares']), format="%.4f")
            na = c2.number_input("Precio Medio", value=float(r_ed['avg_price']), format="%.4f")
            if c3.button("Actualizar Dato"):
                update_asset_db(int(r_ed['id']), ns, na)
                st.success("Hecho"); st.rerun()
            if st.button(f"Borrar {ed}"):
                delete_asset_db(int(r_ed['id']))
                st.rerun()

# ==============================================================================
# 🤖 PÁGINA 4: ASESOR
# ==============================================================================
elif pagina == "🤖 Asesor de Riesgos":
    st.title("🤖 Diagnóstico IA")
    if df_final.empty: st.warning("Añade inversiones."); st.stop()
    
    c1, c2 = st.columns([1,1])
    with c1:
        msgs = []
        beta = 1.0
        if not history_data.empty and not benchmark_data.empty:
            try:
                my_sum = history_data.sum(axis=1).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                spy_ret = benchmark_data.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
                idx = my_sum.index.intersection(spy_ret.index)
                if len(idx) > 20:
                    cov = my_sum.loc[idx].cov(spy_ret.loc[idx])
                    var = spy_ret.loc[idx].var()
                    beta = cov/var if var != 0 else 1.0
            except: pass
        
        st.metric("Beta Inversora", f"{beta:.2f}", help="Sensibilidad vs S&P 500")
        
        pct_cash = (total_liquidez / patrimonio_total * 100) if patrimonio_total > 0 else 0
        if pct_cash > 40: msgs.append(("ℹ️", f"Mucha liquidez (**{pct_cash:.1f}%**). Ojo inflación."))
        elif pct_cash < 5: msgs.append(("🚨", f"Poca liquidez (**{pct_cash:.1f}%**). Riesgo imprevistos."))
        else: msgs.append(("✅", f"Liquidez saludable (**{pct_cash:.1f}%**)."))

        if beta > 1.3: msgs.append(("🚨", "Inversión agresiva."))
        
        with st.chat_message("assistant", avatar="🤖"):
            for i, m in msgs: st.write(f"{i} {m}")

    with c2:
        st.write("### 🕸️ Correlaciones")
        if len(tickers)>1 and not history_data.empty:
            corr = history_data.corr()
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
        else: st.info("Faltan datos.")

# ==============================================================================
# 🔮 PÁGINA 5: MONTE CARLO
# ==============================================================================
elif pagina == "🔮 Monte Carlo":
    st.title("🔮 Futuro (Solo Inversiones)")
    if df_final.empty: st.stop()
    
    c1, c2 = st.columns([1,2])
    with c1:
        years = st.slider("Años", 1, 30, 10)
        mu, sigma = 0.07, 0.15
        if not history_data.empty:
            daily_series = history_data.sum(axis=1).replace(0, np.nan).dropna()
            if len(daily_series) > 10:
                daily_ret = daily_series.pct_change().dropna()
                mu = daily_ret.mean() * 252
                sigma = daily_ret.std() * np.sqrt(252)
        if np.isnan(mu) or np.isinf(mu): mu = 0.07
        if np.isnan(sigma) or sigma == 0: sigma = 0.15
        st.metric("Retorno Hist.", f"{mu*100:.1f}%")
        st.metric("Volatilidad", f"{sigma*100:.1f}%")
        run = st.button("Simular")

    with c2:
        if run:
            dt = 1/252
            paths = []
            for _ in range(50): 
                p = [total_inversiones]
                for _ in range(int(years*252)):
                    shock = np.random.normal(0,1)
                    price = p[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)
                    p.append(price)
                paths.append(p)
            x = np.linspace(0, years, len(paths[0]))
            fig = go.Figure()
            for p in paths: fig.add_trace(go.Scatter(x=x, y=p, mode='lines', line=dict(color='rgba(0,100,200,0.1)'), showlegend=False))
            avg_path = np.mean(paths, axis=0)
            fig.add_trace(go.Scatter(x=x, y=avg_path, mode='lines', name='Media', line=dict(color='blue', width=3)))
            fig.update_layout(transition_duration=500)
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Escenario Medio: **{np.mean([p[-1] for p in paths]):,.0f} €**")

    st.divider()
    st.subheader("📰 Noticias")
    sel = st.selectbox("Activo", df_final['ticker'].unique())
    if st.button("Ver Noticias"):
        try:
            news = yf.Ticker(sel).news
            for n in news[:3]: st.write(f"{get_sentiment(n['title'])} [{n['title']}]({n['link']})")
        except: st.error("Sin noticias.")

# ==============================================================================
# ⚖️ PÁGINA 6: REBALANCEO
# ==============================================================================
elif pagina == "⚖️ Rebalanceo":
    st.title("⚖️ Rebalanceo (Solo Inversiones)")
    if df_final.empty: st.stop()
    c1, c2 = st.columns([1,2])
    with c1:
        plazo = st.number_input("Meses", 1, 24, 6)
        ws = {}
        tot = 0
        for i, r in df_final.iterrows():
            w = st.number_input(f"{r['ticker']} %", 0, 100, int(r['Peso %']), key=i)
            ws[r['Nombre']] = w; tot += w
        st.metric("Total", f"{tot}%")
        calc = st.button("Calcular")
    with c2:
        if calc and tot==100:
            pat = df_final['Valor Acciones'].sum()
            max_c = max([r['Valor Acciones']/(ws[r['Nombre']]/100) for i,r in df_final.iterrows() if ws[r['Nombre']]>0] + [pat])
            m = (max_c - pat)/plazo
            st.metric("Aportación", f"{m:,.2f} €")
            dat = []
            for i, r in df_final.iterrows():
                dif = max(0, (max_c * ws[r['Nombre']]/100) - r['Valor Acciones'])
                dat.append({'Activo':r['Nombre'], 'Mes':dif/plazo})
            st.plotly_chart(px.bar(pd.DataFrame(dat), x='Activo', y='Mes'), use_container_width=True)
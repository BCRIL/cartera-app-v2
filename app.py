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
        st.caption("Gestión Patrimonial Institucional")
        
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
        "🤖 Asesor de Riesgos", 
        "🔮 Monte Carlo", 
        "➕ Gestión de Inversiones", 
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

# --- FUNCIONES BASE DE DATOS (INVERSIONES) ---
def get_assets_db():
    try: return pd.DataFrame(supabase.table('assets').select("*").eq('user_id', user.id).execute().data)
    except: return pd.DataFrame()

def add_asset_db(t, n, s, p, pl):
    supabase.table('assets').insert({"user_id": user.id, "ticker": t, "nombre": n, "shares": s, "avg_price": p, "platform": pl}).execute()

def update_asset_db(asset_id, shares, avg_price):
    supabase.table('assets').update({"shares": shares, "avg_price": avg_price}).eq('id', asset_id).execute()

def delete_asset_db(id_del):
    supabase.table('assets').delete().eq('id', id_del).execute()

# --- FUNCIONES BASE DE DATOS (LIQUIDEZ) ---
def get_liquidity_db():
    try: return pd.DataFrame(supabase.table('liquidity').select("*").eq('user_id', user.id).execute().data)
    except: return pd.DataFrame()

def add_liquidity_db(name, amount, yield_pct):
    supabase.table('liquidity').insert({"user_id": user.id, "name": name, "amount": amount, "yield": yield_pct}).execute()

def delete_liquidity_db(id_del):
    supabase.table('liquidity').delete().eq('id', id_del).execute()

# --- CARGA Y PROCESAMIENTO DE DATOS ---
df_assets = get_assets_db()
df_cash = get_liquidity_db()

# Procesar Inversiones
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()

if not df_assets.empty:
    tickers = df_assets['ticker'].unique().tolist()
    tickers_api = list(set(tickers + ['SPY', 'GLD'])) 
    try:
        data_raw = yf.download(tickers_api, period="1y", progress=False)['Close']
        data_raw.index = data_raw.index.tz_localize(None)
        data_raw = data_raw.fillna(method='ffill').fillna(method='bfill')
        
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
    except Exception as e: st.error(f"Error Mercado: {e}")

# Totales Globales
total_inversiones = df_final['Valor Acciones'].sum() if not df_final.empty else 0
total_liquidez = df_cash['amount'].sum() if not df_cash.empty else 0
patrimonio_total = total_inversiones + total_liquidez

# ==============================================================================
# 📊 PÁGINA 1: DASHBOARD
# ==============================================================================
if pagina == "📊 Dashboard & Alpha":
    st.title("📊 Control de Mando Integral")
    
    # KPIs Globales
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
    
    # Gráficos
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
                fig.add_trace(go.Scatter(x=norm_h.index, y=norm_h, name="Tu Cartera", line=dict(color='#00CC96', width=2)))
                fig.add_trace(go.Scatter(x=norm_b.index, y=norm_b, name="S&P 500", line=dict(color='gray', dash='dot')))
                fig.update_layout(height=300, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
            else: st.info("Datos insuficientes.")
        else: st.info("Añade inversiones para ver el gráfico.")

    with c_pie:
        st.subheader("Asset Allocation")
        # Gráfico Donut: Inversión vs Liquidez
        labels = ['Inversiones', 'Liquidez']
        values = [total_inversiones, total_liquidez]
        fig_alloc = px.pie(names=labels, values=values, hole=0.6, color_discrete_sequence=['#00CC96', '#636EFA'])
        fig_alloc.update_layout(height=300, showlegend=True, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig_alloc, use_container_width=True)

# ==============================================================================
# 💰 PÁGINA 2: LIQUIDEZ (NUEVA)
# ==============================================================================
elif pagina == "💰 Liquidez (Cash)":
    st.title("💰 Gestión de Liquidez")
    
    col_add, col_view = st.columns([1, 2])
    
    with col_add:
        st.markdown("### Añadir Cuenta / Depósito")
        with st.form("add_cash"):
            name = st.text_input("Nombre (Ej: Cuenta BBVA)")
            amount = st.number_input("Saldo (€)", min_value=0.0, step=100.0)
            yield_pct = st.number_input("Rentabilidad Anual (%)", min_value=0.0, step=0.1, help="Si es cuenta remunerada")
            
            if st.form_submit_button("💾 Guardar Efectivo"):
                if name and amount > 0:
                    add_liquidity_db(name, amount, yield_pct)
                    st.success("Añadido.")
                    time.sleep(1); st.rerun()
                else: st.error("Rellena los datos.")

    with col_view:
        st.markdown("### Mis Cuentas")
        if not df_cash.empty:
            for i, row in df_cash.iterrows():
                with st.expander(f"🏦 {row['name']} - {row['amount']:,.2f} €", expanded=True):
                    c1, c2, c3 = st.columns([2,1,1])
                    c1.caption(f"Rentabilidad: **{row['yield']}%**")
                    interes_anual = row['amount'] * (row['yield']/100)
                    c2.caption(f"Genera: ~{interes_anual:.2f}€/año")
                    if c3.button("🗑️", key=f"del_cash_{row['id']}"):
                        delete_liquidity_db(row['id'])
                        st.rerun()
            
            st.divider()
            total_yield = (df_cash['amount'] * df_cash['yield']/100).sum()
            st.info(f"💡 Tu liquidez te genera aproximadamente **{total_yield:,.2f} €** al año en intereses pasivos.")
        else:
            st.info("No tienes efectivo registrado.")

# ==============================================================================
# 🤖 PÁGINA 3: ASESOR
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
        
        # Análisis de Liquidez en el Asesor
        pct_cash = (total_liquidez / patrimonio_total * 100)
        if pct_cash > 40: msgs.append(("ℹ️", f"Tienes mucha liquidez (**{pct_cash:.1f}%**). La inflación te está afectando. Considera invertir más."))
        elif pct_cash < 5: msgs.append(("🚨", f"Tienes poca liquidez (**{pct_cash:.1f}%**). Peligroso ante imprevistos."))
        else: msgs.append(("✅", f"Colchón de liquidez saludable (**{pct_cash:.1f}%**)."))

        if beta > 1.3: msgs.append(("🚨", "Cartera de inversión muy agresiva."))
        
        with st.chat_message("assistant", avatar="🤖"):
            for i, m in msgs: st.write(f"{i} {m}")

    with c2:
        st.write("### 🕸️ Correlaciones (Inversiones)")
        if len(tickers)>1 and not history_data.empty:
            corr = history_data.corr()
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
            
        else: st.info("Faltan datos.")

# ==============================================================================
# 🔮 PÁGINA 4: MONTE CARLO
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
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Tus inversiones podrían valer **{np.mean([p[-1] for p in paths]):,.0f} €** (sin contar liquidez).")

    st.divider()
    st.subheader("📰 Noticias")
    sel = st.selectbox("Activo", df_final['ticker'].unique())
    if st.button("Ver Noticias"):
        try:
            news = yf.Ticker(sel).news
            for n in news[:3]: st.write(f"{get_sentiment(n['title'])} [{n['title']}]({n['link']})")
        except: st.error("Sin noticias.")

# ==============================================================================
# ➕ PÁGINA 5: GESTIÓN DE INVERSIONES
# ==============================================================================
elif pagina == "➕ Gestión de Inversiones":
    st.title("➕ Inversiones (Stocks/ETFs)")
    t_new, t_trade, t_edit = st.tabs(["🆕 Añadir", "💰 Operar", "✏️ Editar"])
    
    with t_new:
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
                            inv = st.number_input("Invertido (€)", min_value=0.0)
                            val = st.number_input("Valor (€)", min_value=0.0)
                            pl = st.selectbox("Broker", ["MyInvestor", "XTB", "TR", "Degiro"])
                            if st.form_submit_button("Guardar") and val > 0:
                                shares = val / cp
                                avg_p = inv / shares if shares > 0 else 0
                                add_asset_db(t, a.get('longname', t), shares, avg_p, pl)
                                st.success("Añadido."); time.sleep(1); st.rerun()
                except: st.error("Error precio.")

    with t_trade:
        if df_final.empty: st.warning("Añade inversiones.")
        else:
            c_sel, c_ops = st.columns([1, 2])
            with c_sel:
                nom = st.selectbox("Activo", df_final['Nombre'].unique())
                row = df_final[df_final['Nombre'] == nom].iloc[0]
                cur_shares = row['shares']
                cur_avg = row['avg_price']
                st.info(f"Tienes **{cur_shares:.4f}** accs a **{cur_avg:.2f}€**.")
            with c_ops:
                live_price = row['Precio Actual']
                st.metric("Precio Mercado", f"{live_price:.2f} €")
                op = st.radio("Acción", ["🟢 Compra", "🔴 Venta"], horizontal=True)
                amt = st.number_input("Importe (€)", 0.0, step=50.0)
                if amt > 0:
                    sh_imp = amt / live_price
                    if "Compra" in op:
                        new_sh = cur_shares + sh_imp
                        new_avg = ((cur_shares*cur_avg)+amt)/new_sh
                        if st.button("Confirmar Compra"):
                            update_asset_db(int(row['id']), new_sh, new_avg)
                            st.rerun()
                    else:
                        if amt > (cur_shares * live_price): st.error("Saldo insuficiente.")
                        else:
                            new_sh = cur_shares - sh_imp
                            if st.button("Confirmar Venta"):
                                if new_sh <= 0.001: delete_asset_db(int(row['id']))
                                else: update_asset_db(int(row['id']), new_sh, cur_avg)
                                st.rerun()

    with t_edit:
        if not df_final.empty:
            ed = st.selectbox("Editar:", df_final['Nombre'], key='ed')
            r_ed = df_final[df_final['Nombre']==ed].iloc[0]
            c1, c2, c3 = st.columns(3)
            ns = c1.number_input("Acciones", value=float(r_ed['shares']), format="%.4f")
            na = c2.number_input("Precio Medio", value=float(r_ed['avg_price']), format="%.4f")
            if c3.button("Actualizar"):
                update_asset_db(int(r_ed['id']), ns, na)
                st.success("Hecho"); st.rerun()
            if st.button(f"Borrar {ed}"):
                delete_asset_db(int(r_ed['id']))
                st.rerun()

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
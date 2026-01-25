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
        "🤖 Asesor de Riesgos", 
        "🔮 Monte Carlo", 
        "➕ Gestión de Cartera", 
        "⚖️ Rebalanceo"
    ])
    st.divider()
    if st.button("Cerrar Sesión"):
        supabase.auth.sign_out(); st.session_state['user'] = None; st.rerun()

# --- FUNCIONES MATEMÁTICAS (ROBUSTAS) ---
def safe_metric_calc(series):
    """Calcula métricas evitando NaN/Inf"""
    clean = series.fillna(method='ffill').dropna()
    if len(clean) < 5: return 0, 0, 0, 0
    
    returns = clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty: return 0, 0, 0, 0

    try:
        total_ret = (clean.iloc[-1] / clean.iloc[0]) - 1
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

# --- FUNCIONES BASE DE DATOS ---
def get_assets_db():
    return pd.DataFrame(supabase.table('assets').select("*").eq('user_id', user.id).execute().data)

def add_asset_db(t, n, s, p, pl):
    supabase.table('assets').insert({"user_id": user.id, "ticker": t, "nombre": n, "shares": s, "avg_price": p, "platform": pl}).execute()

def update_asset_db(asset_id, shares, avg_price):
    supabase.table('assets').update({"shares": shares, "avg_price": avg_price}).eq('id', asset_id).execute()

def delete_asset_db(id_del):
    supabase.table('assets').delete().eq('id', id_del).execute()

# --- PROCESAMIENTO DE DATOS ---
df_db = get_assets_db()
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()

if not df_db.empty:
    tickers = df_db['ticker'].unique().tolist()
    # Añadimos SPY y GLD para análisis
    tickers_api = list(set(tickers + ['SPY', 'GLD'])) 
    
    try:
        # Descarga Datos
        data_raw = yf.download(tickers_api, period="1y", progress=False)['Close']
        # Corregir Timezone
        data_raw.index = data_raw.index.tz_localize(None)
        # Rellenar huecos
        data_raw = data_raw.fillna(method='ffill').fillna(method='bfill')
        
        if 'SPY' in data_raw.columns:
            benchmark_data = data_raw['SPY']
            history = data_raw.drop(columns=['SPY', 'GLD'], errors='ignore')
        else:
            history = data_raw
            
        history_data = history
        
        metrics_dict = {}
        prices_dict = {}
        rsi_dict = {}
        
        for t in tickers:
            if t in history.columns:
                s = history[t]
                prices_dict[t] = s.iloc[-1]
                
                # RSI
                if len(s) > 15:
                    delta = s.diff()
                    up, down = delta.clip(lower=0), -1*delta.clip(upper=0)
                    rs = up.ewm(com=13).mean() / down.ewm(com=13).mean()
                    rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
                    rsi_dict[t] = 50 if np.isnan(rsi_val) else rsi_val
                else: rsi_dict[t] = 50
                
                ret, vol, dd, sha = safe_metric_calc(s)
                metrics_dict[t] = {'vol': vol, 'dd': dd, 'sharpe': sha}
            else:
                prices_dict[t] = 0; rsi_dict[t]=50; metrics_dict[t]={'vol':0,'dd':0,'sharpe':0}

        # DataFrame Final
        df_db['Precio Actual'] = df_db['ticker'].map(prices_dict)
        df_db['RSI'] = df_db['ticker'].map(rsi_dict)
        df_db['Volatilidad'] = df_db['ticker'].apply(lambda x: metrics_dict[x]['vol']*100)
        df_db['Max Drawdown'] = df_db['ticker'].apply(lambda x: metrics_dict[x]['dd']*100)
        df_db['Sharpe'] = df_db['ticker'].apply(lambda x: metrics_dict[x]['sharpe'])
        
        df_db['Valor Acciones'] = df_db['shares'] * df_db['Precio Actual']
        df_db['Dinero Invertido'] = df_db['shares'] * df_db['avg_price']
        df_db['Ganancia'] = df_db['Valor Acciones'] - df_db['Dinero Invertido']
        
        df_db['Rentabilidad'] = df_db.apply(lambda row: (row['Ganancia']/row['Dinero Invertido']*100) if row['Dinero Invertido']>0 else 0, axis=1)
        
        total_val = df_db['Valor Acciones'].sum()
        df_db['Peso %'] = df_db.apply(lambda row: (row['Valor Acciones']/total_val*100) if total_val>0 else 0, axis=1)
        
        df_final = df_db.rename(columns={'nombre': 'Nombre'})
        
    except Exception as e: st.error(f"Error procesando datos: {e}")

# ==============================================================================
# 📊 PÁGINA 1: DASHBOARD & ALPHA (CON FILTRO DE FECHA)
# ==============================================================================
if pagina == "📊 Dashboard & Alpha":
    st.title("📊 Control de Mando")
    if df_final.empty: st.warning("Cartera vacía. Añade activos."); st.stop()
    
    # --- FILTRO DE FECHA ---
    col_kpi, col_filter = st.columns([3, 1])
    with col_filter:
        default_date = datetime.now() - timedelta(days=180)
        start_date = st.date_input("📅 Rendimiento desde:", value=default_date)
        
    # Filtrar Historial
    hist_filtered = pd.DataFrame()
    bench_filtered = pd.Series()
    if not history_data.empty:
        start_dt = pd.to_datetime(start_date).tz_localize(None)
        hist_filtered = history_data[history_data.index >= start_dt]
        if not benchmark_data.empty:
            bench_filtered = benchmark_data[benchmark_data.index >= start_dt]

    with col_kpi:
        patrimonio = df_final['Valor Acciones'].sum()
        ganancia = df_final['Ganancia'].sum()
        inv_total = df_final['Dinero Invertido'].sum()
        rent_total = (ganancia / inv_total * 100) if inv_total > 0 else 0
        sharpe_avg = df_final['Sharpe'].mean()
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💰 Patrimonio", f"{patrimonio:,.2f} €", help="Valor total actual")
        k2.metric("📈 Beneficio", f"{ganancia:+,.2f} €", f"{rent_total:+.2f}%")
        k3.metric("⚖️ Sharpe", f"{sharpe_avg:.2f}", help="Calidad del retorno (>1 es bueno)")
        k4.metric("📉 Drawdown", f"{df_final['Max Drawdown'].min():.2f}%", help="Peor caída en un año")
    
    st.divider()
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("🏁 Alpha vs Mercado")
        if not hist_filtered.empty and not bench_filtered.empty:
            my_hist = hist_filtered.sum(axis=1)
            idx = my_hist.index.intersection(bench_filtered.index)
            
            if len(idx) > 2:
                my_s = my_hist.loc[idx]
                spy_s = bench_filtered.loc[idx]
                
                # Normalizar base 100 desde FECHA ELEGIDA
                my_norm = my_s / my_s.iloc[0] * 100
                spy_norm = spy_s / spy_s.iloc[0] * 100
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=my_norm.index, y=my_norm, name="Tú", line=dict(color='#00CC96', width=2)))
                fig.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, name="S&P 500", line=dict(color='gray', dash='dot')))
                fig.update_layout(height=350, margin=dict(l=0,r=0,t=0,b=0), hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                
                alpha = my_norm.iloc[-1] - spy_norm.iloc[-1]
                if alpha > 0: st.success(f"🚀 Desde {start_date.strftime('%d/%m')}, ganas al mercado por **{alpha:.2f}%**")
                else: st.warning(f"⚠️ Desde {start_date.strftime('%d/%m')}, el mercado te gana por **{abs(alpha):.2f}%**")
            else: st.info("Datos insuficientes en este rango.")
        else: st.info("Sin datos históricos para este periodo.")
            
    with c2:
        st.subheader("Diversificación")
        fig = px.pie(df_final, values='Valor Acciones', names='Nombre', hole=0.6)
        fig.update_layout(height=300, showlegend=False, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Detalle de Activos")
    st.dataframe(df_final[['Nombre','Precio Actual','Peso %','Rentabilidad','RSI','Sharpe','Max Drawdown']].style.format({
        'Precio Actual':'{:.2f}€', 'Peso %':'{:.1f}%', 'Rentabilidad':'{:.2f}%', 'RSI':'{:.0f}', 'Sharpe':'{:.2f}', 'Max Drawdown':'{:.2f}%'
    }).background_gradient(subset=['Rentabilidad'], cmap='RdYlGn', vmin=-20, vmax=20), use_container_width=True)

# ==============================================================================
# 🤖 PÁGINA 2: ASESOR
# ==============================================================================
elif pagina == "🤖 Asesor de Riesgos":
    st.title("🤖 Diagnóstico IA")
    if df_final.empty: st.stop()
    
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
            
        st.metric("Beta (Riesgo)", f"{beta:.2f}", help=">1: Más volátil que el mercado. <1: Más estable.")
        
        if beta > 1.3: msgs.append(("🚨", "Tu cartera es **agresiva**. Considera comprar Bonos (TLT) u Oro (GLD)."))
        elif beta < 0.7: msgs.append(("🛡️", "Tu cartera es **defensiva**. Resistirás bien las crisis."))
        
        over = df_final[df_final['RSI']>75]['Nombre'].tolist()
        under = df_final[df_final['RSI']<30]['Nombre'].tolist()
        if over: msgs.append(("🔴", f"Venta sugerida: **{', '.join(over)}** (Sobrecompra)."))
        if under: msgs.append(("🟢", f"Compra sugerida: **{', '.join(under)}** (Sobreventa)."))
        
        with st.chat_message("assistant", avatar="🤖"):
            st.write("Diagnóstico:")
            for i, m in msgs: st.write(f"{i} {m}")
            if not msgs: st.write("✅ Cartera equilibrada.")

    with c2:
        st.write("### 🕸️ Mapa de Correlación")
        if len(tickers)>1 and not history_data.empty:
            corr = history_data.corr()
            fig, ax = plt.subplots(figsize=(5,4))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
            
        else: st.info("Faltan datos para correlaciones.")

# ==============================================================================
# 🔮 PÁGINA 3: MONTE CARLO (FIX NAN)
# ==============================================================================
elif pagina == "🔮 Monte Carlo":
    st.title("🔮 Simulación Futura")
    if df_final.empty: st.stop()
    
    c1, c2 = st.columns([1,2])
    with c1:
        years = st.slider("Años", 1, 30, 10)
        
        # Parámetros seguros
        mu, sigma = 0.07, 0.15 # Defaults
        if not history_data.empty:
            daily_series = history_data.sum(axis=1).replace(0, np.nan).dropna()
            if len(daily_series) > 10:
                daily_ret = daily_series.pct_change().dropna()
                mu = daily_ret.mean() * 252
                sigma = daily_ret.std() * np.sqrt(252)
        
        if np.isnan(mu) or np.isinf(mu): mu = 0.07
        if np.isnan(sigma) or sigma == 0: sigma = 0.15
            
        st.metric("Retorno Histórico", f"{mu*100:.1f}%")
        st.metric("Volatilidad", f"{sigma*100:.1f}%")
        
        curr_val = df_final['Valor Acciones'].sum()
        run = st.button("Simular")

    with c2:
        if run:
            dt = 1/252
            paths = []
            for _ in range(50): 
                p = [curr_val]
                for _ in range(int(years*252)):
                    shock = np.random.normal(0,1)
                    price = p[-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*shock)
                    p.append(price)
                paths.append(p)
            
            x = np.linspace(0, years, len(paths[0]))
            fig = go.Figure()
            for p in paths:
                fig.add_trace(go.Scatter(x=x, y=p, mode='lines', line=dict(color='rgba(0,100,200,0.1)'), showlegend=False))
            
            avg_path = np.mean(paths, axis=0)
            fig.add_trace(go.Scatter(x=x, y=avg_path, mode='lines', name='Media', line=dict(color='blue', width=3)))
            st.plotly_chart(fig, use_container_width=True)
            
            end_vals = [p[-1] for p in paths]
            st.info(f"Escenario Medio: **{np.mean(end_vals):,.0f} €**")

    st.divider()
    st.subheader("📰 Noticias")
    sel = st.selectbox("Activo", df_final['ticker'].unique())
    if st.button("Ver Noticias"):
        try:
            news = yf.Ticker(sel).news
            for n in news[:3]:
                st.write(f"{get_sentiment(n['title'])} [{n['title']}]({n['link']})")
        except: st.error("Sin noticias.")

# ==============================================================================
# ➕ PÁGINA 4: GESTIÓN DE CARTERA (NUEVO SISTEMA)
# ==============================================================================
elif pagina == "➕ Gestión de Cartera":
    st.title("➕ Gestión de Cartera")
    
    t_new, t_trade, t_edit = st.tabs(["🆕 Añadir Nuevo", "💰 Operar", "✏️ Editar"])
    
    # --- TAB 1: AÑADIR ---
    with t_new:
        c1, c2 = st.columns(2)
        with c1:
            q = st.text_input("Buscar Activo (Nombre/ISIN):", placeholder="Ej: Apple, Vanguard...")
            if st.button("🔍 Buscar") and q:
                r = search(q)
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
                        st.metric("Precio Mercado", f"{cp:.2f} €")
                        
                        with st.form("new_asset"):
                            inv = st.number_input("Invertido (€)", min_value=0.0)
                            val = st.number_input("Valor Actual (€)", min_value=0.0)
                            pl = st.selectbox("Broker", ["MyInvestor", "XTB", "Trade Republic", "Degiro", "IBKR"])
                            
                            if st.form_submit_button("💾 Guardar") and val > 0:
                                shares = val / cp
                                avg_p = inv / shares if shares > 0 else 0
                                add_asset_db(t, a.get('longname', t), shares, avg_p, pl)
                                st.success(f"✅ {t} añadido.")
                                time.sleep(1); st.rerun()
                except: st.error("Error obteniendo precio.")

    # --- TAB 2: OPERAR ---
    with t_trade:
        if df_final.empty: st.warning("Añade activos primero.")
        else:
            c_sel, c_ops = st.columns([1, 2])
            with c_sel:
                nom = st.selectbox("Activo a Operar", df_final['Nombre'].unique())
                row = df_final[df_final['Nombre'] == nom].iloc[0]
                cur_shares = row['shares']
                cur_avg = row['avg_price']
                
                st.info(f"Tienes **{cur_shares:.4f}** acciones a **{cur_avg:.2f}€** media.")

            with c_ops:
                live_price = row['Precio Actual']
                st.metric("Precio Mercado", f"{live_price:.2f} €")
                
                op = st.radio("Tipo", ["🟢 Compra", "🔴 Venta"], horizontal=True)
                amt = st.number_input("Importe (€)", min_value=0.0, step=50.0)
                
                if amt > 0:
                    sh_imp = amt / live_price
                    if "Compra" in op:
                        new_sh = cur_shares + sh_imp
                        new_inv = (cur_shares * cur_avg) + amt
                        new_avg = new_inv / new_sh
                        st.success(f"Comprarás {sh_imp:.4f} acciones. Nuevo Precio Medio: **{new_avg:.2f}€**")
                        if st.button("✅ Confirmar Compra"):
                            update_asset_db(int(row['id']), new_sh, new_avg)
                            st.rerun()
                    else:
                        if amt > (cur_shares * live_price): st.error("No tienes tanto saldo.")
                        else:
                            new_sh = cur_shares - sh_imp
                            st.warning(f"Venderás {sh_imp:.4f} acciones.")
                            if st.button("🚨 Confirmar Venta"):
                                if new_sh <= 0.001: delete_asset_db(int(row['id']))
                                else: update_asset_db(int(row['id']), new_sh, cur_avg)
                                st.rerun()

    # --- TAB 3: EDITAR ---
    with t_edit:
        if not df_final.empty:
            ed = st.selectbox("Editar:", df_final['Nombre'], key='ed')
            r_ed = df_final[df_final['Nombre']==ed].iloc[0]
            
            c1, c2, c3 = st.columns(3)
            ns = c1.number_input("Acciones", value=float(r_ed['shares']), format="%.4f")
            na = c2.number_input("Precio Medio", value=float(r_ed['avg_price']), format="%.4f")
            
            if c3.button("💾 Actualizar"):
                update_asset_db(int(r_ed['id']), ns, na)
                st.success("Hecho"); st.rerun()
            
            st.divider()
            if st.button(f"🗑️ Borrar {ed}"):
                delete_asset_db(int(r_ed['id']))
                st.rerun()

# ==============================================================================
# ⚖️ PÁGINA 5: REBALANCEO
# ==============================================================================
elif pagina == "⚖️ Rebalanceo":
    st.title("⚖️ Rebalanceo")
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
            st.metric("Aportación Mensual", f"{m:,.2f} €")
            
            dat = []
            for i, r in df_final.iterrows():
                dif = max(0, (max_c * ws[r['Nombre']]/100) - r['Valor Acciones'])
                dat.append({'Activo':r['Nombre'], 'Mes':dif/plazo})
            st.plotly_chart(px.bar(pd.DataFrame(dat), x='Activo', y='Mes'), use_container_width=True)
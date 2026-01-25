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

# --- ESTILOS CSS PERSONALIZADOS (MODO PRO) ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; border-radius: 10px; padding: 15px; border: 1px solid #e0e0e0;}
    .stMetric {text-align: center;}
    /* Tooltip personalizado */
    div[data-testid="stMetricLabel"] > div > div {
        cursor: help;
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

# --- GESTIÓN DE SESIÓN ---
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
        st.caption("Sistema de Gestión Patrimonial Institucional")
        
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

# --- SIDEBAR PROFESIONAL ---
with st.sidebar:
    st.image(user.user_metadata.get('avatar_url', 'https://cdn-icons-png.flaticon.com/512/3135/3135715.png'), width=60)
    st.markdown(f"### {user.user_metadata.get('full_name', 'Inversor')}")
    st.caption("Licencia Pro Activa ✅")
    
    pagina = st.radio("Navegación", [
        "📊 Dashboard & Alpha", 
        "🤖 Asesor de Riesgos (IA)", 
        "🔮 Monte Carlo & Futuro",
        "➕ Gestión de Activos", 
        "⚖️ Rebalanceo Táctico"
    ])
    
    st.divider()
    if st.button("Cerrar Sesión"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.rerun()

# --- FUNCIONES MATEMÁTICAS AVANZADAS (ROBUSTAS) ---
def calculate_advanced_metrics(series):
    """Calcula Sharpe, Volatilidad y Drawdown de forma segura"""
    # Limpieza de datos
    clean_series = series.dropna()
    if len(clean_series) < 2: return 0, 0, 0, 0
    
    returns = clean_series.pct_change().dropna()
    if returns.empty: return 0, 0, 0, 0
    
    # Retorno total
    try:
        total_ret = (clean_series.iloc[-1] / clean_series.iloc[0]) - 1
    except: total_ret = 0
    
    # Volatilidad Anual
    vol = returns.std() * np.sqrt(252)
    if np.isnan(vol): vol = 0
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    if np.isnan(max_dd): max_dd = 0
    
    # Sharpe Ratio (Risk Free Rate = 3%)
    rf = 0.03
    if vol != 0:
        excess_ret = (returns.mean() * 252) - rf
        sharpe = excess_ret / vol
    else:
        sharpe = 0
    
    return total_ret, vol, max_dd, sharpe

def get_sentiment(text):
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0.1: return "🟢"
    elif blob.sentiment.polarity < -0.1: return "🔴"
    else: return "⚪"

# --- CARGA DE DATOS OPTIMIZADA ---
def get_assets_db():
    resp = supabase.table('assets').select("*").eq('user_id', user.id).execute()
    return pd.DataFrame(resp.data)

def add_asset_db(ticker, nombre, shares, price, platform):
    supabase.table('assets').insert({
        "user_id": user.id, "ticker": ticker, "nombre": nombre, 
        "shares": shares, "avg_price": price, "platform": platform
    }).execute()

def delete_asset_db(id_del):
    supabase.table('assets').delete().eq('id', id_del).execute()

df_db = get_assets_db()
df_final = pd.DataFrame()
history_data = pd.DataFrame()
benchmark_data = pd.Series()

if not df_db.empty:
    tickers = df_db['ticker'].unique().tolist()
    # Añadimos SPY (S&P 500) y GLD (Oro) para comparar
    tickers_api = tickers + ['SPY', 'GLD']
    
    try:
        # Descarga masiva
        raw_data = yf.download(tickers_api, period="1y", progress=False)['Close']
        
        # --- CORRECCIÓN CRÍTICA DE ZONA HORARIA ---
        # Eliminamos la zona horaria para poder comparar fechas sin error
        raw_data.index = raw_data.index.tz_localize(None)
        
        # Separar Benchmark y Carteras
        if 'SPY' in raw_data.columns:
            benchmark_data = raw_data['SPY']
            history = raw_data.drop(columns=['SPY', 'GLD'], errors='ignore')
        else:
            history = raw_data
            
        history_data = history
        
        # Procesar métricas individuales
        metrics = {}
        prices = {}
        rsi_dict = {}
        
        for t in tickers:
            if t in history.columns:
                s = history[t].dropna()
                if not s.empty:
                    prices[t] = s.iloc[-1]
                    
                    # RSI seguro
                    if len(s) > 14:
                        delta = s.diff()
                        up = delta.clip(lower=0)
                        down = -1 * delta.clip(upper=0)
                        ema_up = up.ewm(com=13, adjust=False).mean()
                        ema_down = down.ewm(com=13, adjust=False).mean()
                        rs = ema_up / ema_down
                        # Manejar división por cero
                        rs = rs.fillna(0)
                        rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
                        rsi_dict[t] = rsi_val
                    else:
                        rsi_dict[t] = 50
                    
                    # Métricas Pro
                    ret, vol, dd, sharpe = calculate_advanced_metrics(s)
                    metrics[t] = {'vol': vol, 'dd': dd, 'sharpe': sharpe}
                else:
                    prices[t] = 0; rsi_dict[t] = 50; metrics[t] = {'vol':0, 'dd':0, 'sharpe':0}
            else:
                prices[t] = 0; rsi_dict[t] = 50; metrics[t] = {'vol':0, 'dd':0, 'sharpe':0}
        
        # Construir DataFrame Final
        df_db['Precio Actual'] = df_db['ticker'].map(prices)
        df_db['RSI'] = df_db['ticker'].map(rsi_dict).fillna(50)
        df_db['Volatilidad'] = df_db['ticker'].apply(lambda x: metrics[x]['vol']*100)
        df_db['Max Drawdown'] = df_db['ticker'].apply(lambda x: metrics[x]['dd']*100)
        df_db['Sharpe'] = df_db['ticker'].apply(lambda x: metrics[x]['sharpe'])
        
        df_db['Valor Acciones'] = df_db['shares'] * df_db['Precio Actual']
        df_db['Dinero Invertido'] = df_db['shares'] * df_db['avg_price']
        df_db['Ganancia'] = df_db['Valor Acciones'] - df_db['Dinero Invertido']
        df_db['Rentabilidad'] = (df_db['Ganancia'] / df_db['Dinero Invertido'] * 100).fillna(0)
        
        total = df_db['Valor Acciones'].sum()
        df_db['Peso %'] = (df_db['Valor Acciones'] / total * 100).fillna(0)
        
        df_final = df_db.rename(columns={'nombre': 'Nombre'})
        
    except Exception as e: st.error(f"Error procesando datos: {e}")

# ==============================================================================
# 📊 PÁGINA 1: DASHBOARD & ALPHA (Tu vs Mercado)
# ==============================================================================
if pagina == "📊 Dashboard & Alpha":
    st.title("📊 Control de Mando")
    
    if df_final.empty: st.warning("Cartera vacía. Añade activos en el menú."); st.stop()
    
    # 1. KPIs SUPERIORES CON TOOLTIPS DE AYUDA
    patrimonio = df_final['Valor Acciones'].sum()
    ganancia = df_final['Ganancia'].sum()
    rent_total = (ganancia / df_final['Dinero Invertido'].sum() * 100)
    sharpe_medio = df_final['Sharpe'].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💰 Patrimonio Neto", f"{patrimonio:,.2f} €", help="Valor total de todos tus activos a precio de mercado hoy.")
    c2.metric("📈 P&L Total", f"{ganancia:+,.2f} €", f"{rent_total:+.2f}%", help="Beneficio o Pérdida total (Profit & Loss) desde que compraste.")
    
    # Tooltips explicativos
    help_sharpe = """
    💡 **Sharpe Ratio**: Mide la calidad de tu inversión.
    • > 1.0: ¡Excelente! (Ganas mucho para el riesgo que corres).
    • 0 - 1.0: Normal.
    • Negativo: Malo (Estás perdiendo dinero o asumiendo demasiado riesgo).
    """
    c3.metric("⚖️ Sharpe Ratio", f"{sharpe_medio:.2f}", 
              delta="Bueno" if sharpe_medio > 1 else "Normal", 
              help=help_sharpe)
    
    help_dd = """
    📉 **Max Drawdown**: La 'Prueba de Dolor'.
    Indica cuál ha sido la peor caída histórica de tus activos desde su punto más alto en el último año.
    Un -23% significa que en el peor momento, el valor cayó un 23% desde la cima.
    """
    c4.metric("📉 Max Drawdown", f"{df_final['Max Drawdown'].min():.2f}%", help=help_dd)
    
    st.divider()
    
    # 2. GRÁFICO COMPARATIVO: TU CARTERA VS S&P 500
    c_chart, c_pie = st.columns([2, 1])
    
    with c_chart:
        st.subheader("🏁 Tu Cartera vs. Mercado (S&P 500)")
        if not history_data.empty and not benchmark_data.empty:
            try:
                # Crear índice sintético de tu cartera
                my_portfolio_hist = history_data.sum(axis=1)
                
                # Alinear fechas (Intersección)
                common_dates = my_portfolio_hist.index.intersection(benchmark_data.index)
                
                if len(common_dates) > 5:
                    my_pf_aligned = my_portfolio_hist.loc[common_dates]
                    spy_aligned = benchmark_data.loc[common_dates]
                    
                    # Normalizar a base 100 para comparar
                    my_norm = my_pf_aligned / my_pf_aligned.iloc[0] * 100
                    spy_norm = spy_aligned / spy_aligned.iloc[0] * 100
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=my_norm.index, y=my_norm, name="Tu Cartera", line=dict(color='#00CC96', width=3)))
                    fig.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, name="S&P 500", line=dict(color='#EF553B', dash='dot')))
                    fig.update_layout(hovermode="x unified", margin=dict(l=0, r=0, t=0, b=0), height=350)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Calcular Alpha
                    my_perf = my_norm.iloc[-1] - 100
                    spy_perf = spy_norm.iloc[-1] - 100
                    alpha = my_perf - spy_perf
                    
                    if np.isnan(alpha):
                        st.info("Calculando Alpha...")
                    elif alpha > 0: 
                        st.success(f"🚀 ¡Bates al mercado! Alpha: **+{alpha:.2f}%** (Ganas más que el índice).")
                    else: 
                        st.warning(f"⚠️ Estás **{alpha:.2f}%** por debajo del mercado.")
                else:
                    st.info("No hay suficientes días coincidentes de datos para comparar.")
            except Exception as e:
                st.info(f"Gráfico en construcción (Faltan datos históricos): {e}")
            
    with c_pie:
        st.subheader("Diversificación")
        fig_pie = px.pie(df_final, values='Valor Acciones', names='Nombre', hole=0.6, color_discrete_sequence=px.colors.qualitative.Prism)
        fig_pie.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    # 3. TABLA DETALLADA CON HEATMAP
    st.subheader("📋 Análisis de Activos")
    st.caption("Colores: Verde = Bien / Rojo = Mal o Riesgo Alto")
    st.dataframe(df_final[['Nombre', 'Precio Actual', 'Peso %', 'Rentabilidad', 'RSI', 'Sharpe', 'Max Drawdown']].style.format({
        'Precio Actual': '{:.2f}€', 'Peso %': '{:.1f}%', 'Rentabilidad': '{:+.2f}%',
        'RSI': '{:.0f}', 'Sharpe': '{:.2f}', 'Max Drawdown': '{:.2f}%'
    }).background_gradient(subset=['Rentabilidad'], cmap='RdYlGn', vmin=-20, vmax=20)
      .background_gradient(subset=['RSI'], cmap='coolwarm', vmin=30, vmax=70), use_container_width=True)

# ==============================================================================
# 🤖 PÁGINA 2: ASESOR DE RIESGOS (IA + HEDGING)
# ==============================================================================
elif pagina == "🤖 Asesor de Riesgos (IA)":
    st.title("🤖 Asesor de Inteligencia Financiera")
    
    if df_final.empty: st.stop()
    
    col_chat, col_risk = st.columns([1, 1])
    
    with col_chat:
        st.markdown("### 💬 Diagnóstico en Tiempo Real")
        
        msgs = []
        beta = 1.0 # Default
        
        # 1. CÁLCULO DE BETA (RIESGO VS MERCADO)
        if not history_data.empty and not benchmark_data.empty:
            try:
                my_hist_sum = history_data.sum(axis=1)
                common_idx = my_hist_sum.index.intersection(benchmark_data.index)
                
                if len(common_idx) > 10:
                    my_ret = my_hist_sum.loc[common_idx].pct_change().dropna()
                    spy_ret = benchmark_data.loc[common_idx].pct_change().dropna()
                    
                    # Alinear de nuevo tras pct_change
                    idx_final = my_ret.index.intersection(spy_ret.index)
                    
                    if len(idx_final) > 5:
                        cov = my_ret[idx_final].cov(spy_ret[idx_final])
                        var = spy_ret[idx_final].var()
                        if var != 0:
                            beta = cov / var
                        else: beta = 1.0
            except: beta = 1.0 # Fallback
            
            help_beta = """
            📊 **Beta de Cartera**: Sensibilidad al mercado.
            • Beta = 1.0: Tu cartera se mueve igual que el S&P 500.
            • Beta > 1.0 (Ej: 1.5): Si el mercado sube 1%, tú subes 1.5% (y al revés). ¡Agresivo!
            • Beta < 1.0 (Ej: 0.5): Si el mercado cae 10%, tú solo caes 5%. ¡Defensivo!
            """
            
            st.metric("Beta de Cartera", f"{beta:.2f}", help=help_beta)
            
            if beta > 1.2:
                msgs.append(("🚨", f"**Cartera muy agresiva (Beta {beta:.2f}):** Tienes mucho riesgo de mercado. Si la bolsa cae, tú caerás más fuerte."))
                msgs.append(("🛡️", "**Consejo de Cobertura:** Deberías añadir activos descorrelacionados como **Bonos (TLT)** o **Oro (GLD)**."))
            elif beta < 0.8:
                msgs.append(("✅", f"**Cartera defensiva (Beta {beta:.2f}):** Estás bien protegido contra caídas del mercado."))
        
        # 2. RSI Y MOMENTUM
        over = df_final[df_final['RSI'] > 75]['Nombre'].tolist()
        under = df_final[df_final['RSI'] < 30]['Nombre'].tolist()
        
        if over: msgs.append(("🔴", f"**Venta Táctica:** {', '.join(over)} están en sobrecompra extrema (>75 RSI). Históricamente suelen corregir."))
        if under: msgs.append(("🟢", f"**Compra Táctica:** {', '.join(under)} están en sobreventa (<30 RSI). Oportunidad de rebote."))
        
        # 3. RENDERIZAR CHAT
        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"Hola {user.user_metadata.get('full_name','Inversor')}. Aquí tienes mi análisis de riesgos:")
        
        for icon, txt in msgs:
            with st.chat_message("assistant", avatar=icon):
                st.markdown(txt)
                
        if not msgs:
            with st.chat_message("assistant", avatar="✨"):
                st.write("Tu cartera está perfectamente equilibrada. No se detectan anomalías graves.")

    with col_risk:
        st.markdown("### 🕸️ Mapa de Correlaciones")
        st.info("Busca colores **Azules** para diversificar. Los **Rojos** significan que se mueven igual.")
        if len(tickers) > 1 and not history_data.empty:
            corr = history_data.corr()
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=ax)
            st.pyplot(fig)
            
        else:
            st.info("Necesitas 2+ activos con histórico suficiente para ver correlaciones.")

# ==============================================================================
# 🔮 PÁGINA 3: MONTE CARLO & FUTURO (SIMULACIÓN PRO)
# ==============================================================================
elif pagina == "🔮 Monte Carlo & Futuro":
    st.title("🔮 Laboratorio de Futuro")
    st.caption("Simulación Estocástica (Monte Carlo)")
    
    if df_final.empty: st.stop()
    
    c_conf, c_res = st.columns([1, 2])
    
    with c_conf:
        st.subheader("⚙️ Parámetros")
        years = st.slider("Horizonte (Años)", 1, 30, 10)
        sims = 50 # Número de escenarios
        
        # Datos base
        if not history_data.empty:
            daily_ret = history_data.sum(axis=1).pct_change().dropna()
            mu = daily_ret.mean() * 252
            sigma = daily_ret.std() * np.sqrt(252)
        else:
            mu = 0.07; sigma = 0.15 # Defaults
            
        curr_val = df_final['Valor Acciones'].sum()
        
        st.metric("Retorno Histórico", f"{mu*100:.1f}%", help="Lo que ha rendido tu cartera anualmente en el pasado.")
        st.metric("Volatilidad Anual", f"{sigma*100:.1f}%", help="Cuánto oscila el valor de tu cartera.")
        
        run_sim = st.button("🎲 Ejecutar 50 Escenarios")

    with c_res:
        if run_sim:
            with st.spinner("Calculando futuros alternativos..."):
                dt = 1/252
                paths = []
                for _ in range(sims):
                    price = [curr_val]
                    for _ in range(int(years*252)):
                        # Fórmula de Movimiento Browniano Geométrico
                        shock = np.random.normal(0, 1)
                        p = price[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock)
                        price.append(p)
                    paths.append(price)
                
                # Gráfico Spaghetti
                fig = go.Figure()
                x_axis = np.linspace(0, years, len(paths[0]))
                
                # Escenarios tenues
                for p in paths:
                    fig.add_trace(go.Scatter(x=x_axis, y=p, mode='lines', line=dict(color='rgba(0,150,255,0.1)', width=1), showlegend=False))
                
                # Media
                avg_path = np.mean(paths, axis=0)
                fig.add_trace(go.Scatter(x=x_axis, y=avg_path, name='Escenario Medio', line=dict(color='blue', width=3)))
                
                fig.update_layout(title="Proyección Probabilística de tu Riqueza", yaxis_title="Euros", xaxis_title="Años")
                st.plotly_chart(fig, use_container_width=True)
                
                final_vals = [p[-1] for p in paths]
                p10 = np.percentile(final_vals, 10)
                p90 = np.percentile(final_vals, 90)
                
                st.success(f"En el escenario promedio tendrás **{avg_path[-1]:,.0f}€**.")
                st.info(f"Rango probable (80% certeza): entre **{p10:,.0f}€** y **{p90:,.0f}€**.")

    st.divider()
    st.subheader("📰 Noticias Inteligentes (Sentimiento IA)")
    sel_news = st.selectbox("Noticias de:", df_final['ticker'].unique())
    if st.button("Analizar Noticias"):
        try:
            news = yf.Ticker(sel_news).news
            cols = st.columns(3)
            for i, n in enumerate(news[:3]):
                tit = n.get('title', '')
                icon = get_sentiment(tit)
                with cols[i]:
                    st.markdown(f"### {icon}")
                    st.write(f"**{tit}**")
                    st.caption(n.get('publisher', ''))
                    st.markdown(f"[Leer]({n.get('link','')})")
        except: st.error("No hay noticias recientes.")

# ==============================================================================
# ➕ PÁGINA 4: GESTIÓN DE ACTIVOS (MEJORADO)
# ==============================================================================
elif pagina == "➕ Gestión de Activos":
    st.title("➕ Gestión de Cartera")
    
    tab_add, tab_del = st.tabs(["Añadir Nuevo", "Inventario Actual"])
    
    with tab_add:
        c1, c2 = st.columns(2)
        with c1:
            search_txt = st.text_input("🔍 Buscar Activo (Nombre/ISIN):")
            if search_txt:
                try:
                    res = search(search_txt)
                    if 'quotes' in res:
                        opts = {f"{x['symbol']} | {x.get('longname','N/A')} | {x.get('exchange','')}" : x for x in res['quotes']}
                        sel_key = st.selectbox("Resultados:", list(opts.keys()))
                        st.session_state['sel_add'] = opts[sel_key]
                except: st.warning("Buscando...")
        
        with c2:
            if 'sel_add' in st.session_state:
                asset = st.session_state['sel_add']
                ticker = asset['symbol']
                
                # Datos en tiempo real antes de añadir
                info = yf.Ticker(ticker)
                hist = info.history(period="1d")
                
                if not hist.empty:
                    curr_price = hist['Close'].iloc[-1]
                    
                    st.markdown(f"### {ticker}")
                    st.metric("Precio Mercado", f"{curr_price:.2f} €")
                    
                    with st.form("add_form"):
                        inv = st.number_input("Dinero Invertido Total (€)", min_value=0.0)
                        val = st.number_input("Valor Actual Total (€)", min_value=0.0)
                        plat = st.selectbox("Broker", ["MyInvestor", "XTB", "Degiro", "Trade Republic", "IBKR", "Banco"])
                        
                        if st.form_submit_button("💾 Guardar en Cartera"):
                            if val > 0:
                                shares = val / curr_price
                                avg_p = inv / shares if shares > 0 else 0
                                add_asset_db(ticker, asset.get('longname', ticker), shares, avg_p, plat)
                                st.toast("✅ Activo añadido correctamente")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("El valor actual debe ser > 0")

    with tab_del:
        if not df_final.empty:
            st.dataframe(df_final)
            to_del = st.selectbox("Selecciona activo para borrar:", df_final['Nombre'])
            if st.button("🗑️ Eliminar Activo"):
                bid = df_final[df_final['Nombre']==to_del].iloc[0]['id']
                delete_asset_db(bid)
                st.rerun()

# ==============================================================================
# ⚖️ PÁGINA 5: REBALANCEO TÁCTICO
# ==============================================================================
elif pagina == "⚖️ Rebalanceo Táctico":
    st.title("⚖️ Rebalanceo Inteligente")
    
    if df_final.empty: st.stop()
    
    c_input, c_out = st.columns([1, 2])
    
    with c_input:
        st.subheader("1. Tu Estrategia")
        plazo = st.number_input("Meses para completar", 1, 24, 6)
        
        target_w = {}
        sum_w = 0
        
        st.write("Pesos Objetivo (%):")
        for i, row in df_final.iterrows():
            col_a, col_b = st.columns([2, 1])
            with col_a: st.write(f"**{row['ticker']}**")
            with col_b: 
                w = st.number_input(f"%", 0, 100, int(row['Peso %']), key=f"rw_{i}", label_visibility="collapsed")
            target_w[row['Nombre']] = w
            sum_w += w
            
        st.metric("Total %", f"{sum_w}%", delta="OK" if sum_w == 100 else f"Faltan {100-sum_w}%", delta_color="normal" if sum_w==100 else "inverse")
        
        calc_btn = st.button("🚀 Generar Plan de Compra", type="primary", disabled=(sum_w != 100))

    with c_out:
        if calc_btn and sum_w == 100:
            patrimonio = df_final['Valor Acciones'].sum()
            
            # Lógica de "Water-filling" (Llenar lo que falta)
            max_cap_req = 0
            for _, row in df_final.iterrows():
                w_decimal = target_w[row['Nombre']] / 100
                if w_decimal > 0:
                    implied_pf = row['Valor Acciones'] / w_decimal
                    if implied_pf > max_cap_req: max_cap_req = implied_pf
            
            target_pf = max(max_cap_req, patrimonio)
            fresh_capital = target_pf - patrimonio
            monthly = fresh_capital / plazo
            
            st.success(f"### Plan Aprobado")
            m1, m2 = st.columns(2)
            m1.metric("Capital Nuevo Total", f"{fresh_capital:,.0f} €")
            m2.metric("Aportación Mensual", f"{monthly:,.0f} €")
            
            # Tabla de compras
            plan_data = []
            for _, row in df_final.iterrows():
                obj_val = target_pf * (target_w[row['Nombre']] / 100)
                diff = obj_val - row['Valor Acciones']
                to_buy = max(0, diff)
                plan_data.append({
                    "Activo": row['Nombre'],
                    "Peso Actual": f"{row['Peso %']:.1f}%",
                    "Peso Objetivo": f"{target_w[row['Nombre']]}%",
                    "Compra Total": to_buy,
                    "Cuota Mensual": to_buy / plazo
                })
            
            df_plan = pd.DataFrame(plan_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_plan['Activo'],
                y=df_plan['Cuota Mensual'],
                marker_color='#00CC96',
                name='Compra Mensual'
            ))
            fig.update_layout(title="Distribución de Compras Mensuales (DCA)", yaxis_title="Euros")
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_plan)
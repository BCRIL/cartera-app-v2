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

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial Pro", layout="wide", page_icon="🏦")

# --- CONEXIÓN SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Error: Faltan los secretos de Supabase.")
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
        st.title("🔐 Carterapro Login")
        # Botón Google
        if st.button("🇬 Iniciar con Google", type="primary", use_container_width=True):
            try:
                data = supabase.auth.sign_in_with_oauth({
                    "provider": "google",
                    "options": {"redirect_to": "https://carterapro.streamlit.app"}
                })
                st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
            except Exception as e: st.error(f"Error: {e}")
        
        st.divider()
        # Login Email
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
    st.image(user.user_metadata.get('avatar_url', 'https://cdn-icons-png.flaticon.com/512/3135/3135715.png'), width=50)
    st.write(f"Hola, **{user.user_metadata.get('full_name', 'Inversor')}**")
    if st.button("Cerrar Sesión"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.rerun()
    st.divider()
    pagina = st.radio("Menú Principal", [
        "🤖 Asesor IA en Vivo", 
        "📊 Dashboard", 
        "🔮 Proyecciones & Noticias",
        "➕ Añadir Activos", 
        "⚖️ Rebalanceo"
    ])

# --- FUNCIONES FINANCIERAS & UTILIDADES ---
def calculate_rsi(data, window=14):
    """Calcula el RSI para detectar sobrecompra/sobreventa"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_sentiment(text):
    """Analiza sentimiento de noticias"""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1: return "🟢 Positivo"
    elif analysis.sentiment.polarity < -0.1: return "🔴 Negativo"
    else: return "⚪ Neutral"

# --- CARGA DE DATOS CENTRALIZADA ---
def get_assets_db():
    resp = supabase.table('assets').select("*").eq('user_id', user.id).execute()
    return pd.DataFrame(resp.data)

def add_asset_db(ticker, nombre, shares, price, platform):
    supabase.table('assets').insert({"user_id": user.id, "ticker": ticker, "nombre": nombre, "shares": shares, "avg_price": price, "platform": platform}).execute()

def delete_asset_db(id_del):
    supabase.table('assets').delete().eq('id', id_del).execute()

df_db = get_assets_db()
df_final = pd.DataFrame()
history_data = pd.DataFrame() # Para correlaciones y gráficos

if not df_db.empty:
    tickers = df_db['ticker'].unique().tolist()
    try:
        # Descargamos 1 año de historia para todos los cálculos
        history = yf.download(tickers, period="1y", progress=False)['Close']
        
        # Ajuste por si solo hay 1 activo (yfinance devuelve Series en vez de DF)
        if len(tickers) == 1: 
            history = pd.DataFrame({tickers[0]: history})
        
        history_data = history
        
        current_prices = {}
        rsi_values = {}
        volatility = {}
        yearly_return = {}
        
        for t in tickers:
            if t in history.columns:
                series = history[t]
                
                # Datos actuales
                current_prices[t] = series.iloc[-1]
                
                # RSI
                rsi_vals = calculate_rsi(series)
                rsi_values[t] = rsi_vals.iloc[-1]
                
                # Volatilidad & Retorno (Para proyecciones)
                returns = series.pct_change()
                volatility[t] = returns.std() * np.sqrt(252) * 100
                yearly_return[t] = returns.mean() * 252 * 100
            else:
                current_prices[t] = 0
                rsi_values[t] = 50
                volatility[t] = 0
                yearly_return[t] = 0
                
    except Exception as e: st.error(f"Error descargando datos de mercado: {e}")

    # Enriquecer DataFrame
    df_db['Precio Actual'] = df_db['ticker'].map(current_prices).fillna(0)
    df_db['RSI'] = df_db['ticker'].map(rsi_values).fillna(50)
    df_db['Volatilidad'] = df_db['ticker'].map(volatility).fillna(0)
    df_db['Retorno Esperado'] = df_db['ticker'].map(yearly_return).fillna(0)
    
    df_db['Valor Acciones'] = df_db['shares'] * df_db['Precio Actual']
    df_db['Dinero Invertido'] = df_db['shares'] * df_db['avg_price']
    df_db['Ganancia'] = df_db['Valor Acciones'] - df_db['Dinero Invertido']
    df_db['Rentabilidad'] = (df_db['Ganancia'] / df_db['Dinero Invertido'] * 100).fillna(0)
    
    # Calcular Peso %
    total_patrimonio = df_db['Valor Acciones'].sum()
    df_db['Peso %'] = (df_db['Valor Acciones'] / total_patrimonio * 100).fillna(0)
    
    df_final = df_db.rename(columns={'nombre': 'Nombre'})

# ==============================================================================
# 🤖 PÁGINA 1: ASESOR IA EN VIVO (ACTUALIZADO CON HEDGING)
# ==============================================================================
if pagina == "🤖 Asesor IA en Vivo":
    st.title("🤖 Tu Asistente Financiero")
    
    if df_final.empty:
        st.warning("Necesito datos para pensar. Ve a 'Añadir Activos' primero.")
    else:
        # --- 1. LÓGICA INTERNA DE CARTERA ---
        alertas_graves = []
        consejos_compra = []
        consejos_venta = []
        estrategia_rotacion = []

        # Concentración
        max_peso = df_final['Peso %'].max()
        if max_peso > 35:
            asset_max = df_final.loc[df_final['Peso %'].idxmax(), 'Nombre']
            alertas_graves.append(f"⚠️ **Riesgo Alto:** Estás muy expuesto a **{asset_max}** ({max_peso:.1f}%). Si cae, sufres. Diversifica.")

        # RSI (Timing)
        for _, row in df_final.iterrows():
            if row['RSI'] > 75:
                consejos_venta.append(f"🔴 **{row['Nombre']}** está caro (RSI {row['RSI']:.0f}). Podría corregir.")
            elif row['RSI'] < 30:
                consejos_compra.append(f"🟢 **{row['Nombre']}** ha bajado mucho (RSI {row['RSI']:.0f}). Oportunidad de entrada.")

        # --- 2. LÓGICA DE MERCADO (NUEVO: ROTACIÓN Y COBERTURA) ---
        # Analizamos activos externos para ver qué le falta a la cartera
        benchmarks = {
            'SPY': 'S&P 500 (Mercado)',
            'GLD': 'Oro (Refugio)',
            'TLT': 'Bonos USA (Defensivo)',
            'XLE': 'Energía (Inflación)',
            'XLP': 'Consumo Básico (Estabilidad)'
        }
        
        with st.spinner("Analizando mercado global para sugerirte coberturas..."):
            try:
                # Descargar datos de benchmarks
                bench_data = yf.download(list(benchmarks.keys()), period="1y", progress=False)['Close']
                
                # Crear índice sintético de TU cartera (promedio ponderado simple)
                my_portfolio_returns = history_data.pct_change().mean(axis=1)
                
                # Calcular correlaciones de tu cartera con los benchmarks
                bench_returns = bench_data.pct_change()
                correlations = bench_returns.corrwith(my_portfolio_returns)
                
                # Analizar Beta (Sensibilidad al mercado)
                if 'SPY' in correlations:
                    beta_proxy = correlations['SPY']
                    if beta_proxy > 0.85:
                        alertas_graves.append(f"⚠️ **Cartera Agresiva:** Tu cartera se mueve casi igual que la bolsa (Corr: {beta_proxy:.2f}). En caídas sufrirás.")
                        
                        # Buscar el mejor activo para proteger (Correlación más baja o negativa)
                        best_hedge = correlations.idxmin()
                        hedge_val = correlations.min()
                        hedge_name = benchmarks.get(best_hedge, best_hedge)
                        
                        estrategia_rotacion.append(f"🛡️ **Consejo Defensivo:** Para no notar tanto los cambios, añade **{hedge_name} ({best_hedge})**. Su correlación contigo es baja ({hedge_val:.2f}), así que cuando tu cartera baje, esto debería aguantar o subir.")
                    
                    elif beta_proxy < 0.3:
                        estrategia_rotacion.append("ℹ️ **Cartera Defensiva:** Tu cartera ya es muy estable y se mueve poco con el mercado.")
                
            except Exception as e:
                # Fallo silencioso en benchmarks para no romper la app
                print(f"Error benchmark: {e}")

        # --- CHAT UI ---
        with st.chat_message("assistant", avatar="🤖"):
            st.write(f"Hola {user.user_metadata.get('full_name','Inversor')}. He analizado tus activos y el mercado global.")
        
        if alertas_graves:
            with st.chat_message("assistant", avatar="🚨"):
                for a in alertas_graves: st.error(a)
        
        # MOSTRAR ESTRATEGIA DE ROTACIÓN (LO NUEVO)
        if estrategia_rotacion:
            with st.chat_message("assistant", avatar="🛡️"):
                st.write("**Estrategia de Rotación sugerida:**")
                for e in estrategia_rotacion: st.info(e)
                
        if consejos_compra or consejos_venta:
            with st.chat_message("assistant", avatar="💰"):
                st.write("**Movimientos Tácticos en tu cartera:**")
                for c in consejos_compra: st.markdown(c)
                for v in consejos_venta: st.markdown(v)
        
        if not alertas_graves and not consejos_compra and not consejos_venta:
             with st.chat_message("assistant", avatar="✅"):
                st.write("Todo parece en orden. Tu cartera está equilibrada.")

        st.divider()
        
        # --- MAPA DE CORRELACIONES ---
        st.subheader("🕸️ Diversificación Real")
        with st.expander("Ver Mapa de Calor", expanded=True):
            if not history_data.empty and len(tickers) > 1:
                corr = history_data.corr()
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
                st.pyplot(fig)
                
            else:
                st.info("Añade al menos 2 activos para ver correlaciones.")

# ==============================================================================
# 📊 PÁGINA 2: DASHBOARD GENERAL
# ==============================================================================
elif pagina == "📊 Dashboard":
    st.title("📊 Visión Global")
    if df_final.empty: st.stop()
    
    patrimonio = df_final['Valor Acciones'].sum()
    ganancia = df_final['Ganancia'].sum()
    volatilidad_media = df_final['Volatilidad'].mean()
    
    k1, k2, k3 = st.columns(3)
    k1.metric("Patrimonio Total", f"{patrimonio:,.2f} €")
    k2.metric("Ganancia Total", f"{ganancia:,.2f} €", f"{(ganancia/patrimonio*100):.2f}%")
    k3.metric("Riesgo (Volatilidad)", f"{volatilidad_media:.2f}%")
    
    c1, c2 = st.columns([2,1])
    with c1:
        # Treemap
        fig = px.treemap(df_final, path=['platform', 'Nombre'], values='Valor Acciones', 
                         color='Rentabilidad', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Tabla resumen
        st.dataframe(df_final[['Nombre', 'Precio Actual', 'RSI', 'Peso %']].style.format({
            'Precio Actual': '{:.2f}€', 'RSI': '{:.0f}', 'Peso %': '{:.1f}%'
        }).background_gradient(subset=['RSI'], cmap='coolwarm'), use_container_width=True, height=400)

# ==============================================================================
# 🔮 PÁGINA 3: PROYECCIONES & NOTICIAS
# ==============================================================================
elif pagina == "🔮 Proyecciones & Noticias":
    st.title("🔮 Futuro y Noticias")
    if df_final.empty: st.warning("Faltan activos."); st.stop()
    
    tab_proj, tab_news = st.tabs(["📈 Proyección Patrimonial", "📰 Noticias con IA"])
    
    with tab_proj:
        st.subheader("El poder del interés compuesto")
        years = st.slider("Años vista", 1, 30, 10)
        avg_ret = df_final['Retorno Esperado'].mean() / 100
        if avg_ret == 0: avg_ret = 0.07 # Asumir 7% si no hay datos
        
        start_val = df_final['Valor Acciones'].sum()
        end_val = start_val * (1 + avg_ret)**years
        
        c1, c2 = st.columns(2)
        c1.metric("Valor Hoy", f"{start_val:,.0f} €")
        c2.metric(f"Valor en {years} años (@{avg_ret*100:.1f}%)", f"{end_val:,.0f} €", delta=f"+{end_val-start_val:,.0f} €")
        
        # Gráfico
        x = list(range(years+1))
        y = [start_val * (1 + avg_ret)**i for i in x]
        fig = px.line(x=x, y=y, labels={'x':'Años', 'y':'Euros'}, title="Crecimiento Estimado")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab_news:
        st.subheader("Análisis de Sentimiento (Yahoo Finance + TextBlob)")
        sel_asset = st.selectbox("Elige activo:", df_final['ticker'].unique())
        if st.button("Analizar Noticias"):
            with st.spinner("Leyendo noticias..."):
                try:
                    news = yf.Ticker(sel_asset).news
                    if not news: st.info("No hay noticias recientes.")
                    for n in news[:5]:
                        tit = n.get('title','')
                        sent = get_sentiment(tit)
                        with st.expander(f"{sent} | {tit}"):
                            st.write(f"Fuente: {n.get('publisher','')}")
                            st.markdown(f"[Leer más]({n.get('link','')})")
                except: st.error("Error cargando noticias.")

# ==============================================================================
# ➕ PÁGINA 4: AÑADIR ACTIVOS
# ==============================================================================
elif pagina == "➕ Añadir Activos":
    st.title("➕ Añadir Inversión")
    c_s, c_c = st.columns(2)
    with c_s:
        query = st.text_input("Buscar Activo (ISIN/Nombre):")
        if st.button("Buscar") and query:
            try:
                res = search(query)
                if 'quotes' in res: st.session_state['res'] = res['quotes']
            except: pass
        
        if 'res' in st.session_state and st.session_state['res']:
            opts = {f"{x['symbol']} - {x.get('longname','')}" : x for x in st.session_state['res']}
            sel = st.selectbox("Resultados:", list(opts.keys()))
            st.session_state['sel_asset'] = opts[sel]

    with c_c:
        if 'sel_asset' in st.session_state:
            asset = st.session_state['sel_asset']
            sym = asset['symbol']
            try:
                # Intentar bajar precio actual
                hist = yf.Ticker(sym).history(period='1d')
                if not hist.empty:
                    curr_price = hist['Close'].iloc[-1]
                    st.metric(f"Precio {sym}", f"{curr_price:.2f}€")
                    
                    invested = st.number_input("Dinero Invertido (€)", 0.0)
                    curr_val = st.number_input("Valor Actual (€)", 0.0)
                    plat = st.selectbox("Plataforma", ["MyInvestor", "XTB", "Degiro", "TR", "Banco"])
                    
                    if st.button("Guardar Activo") and curr_val > 0:
                        shares = curr_val / curr_price
                        avg = invested / shares if shares > 0 else 0
                        add_asset_db(sym, asset.get('longname', sym), shares, avg, plat)
                        st.success("Guardado!")
                        time.sleep(1)
                        st.rerun()
                else:
                    st.error("No pude obtener precio de mercado. Intenta otro activo.")
            except: st.error("Error técnico al obtener precio.")
            
    if not df_final.empty:
        st.divider()
        borrar = st.selectbox("Eliminar activo:", df_final['Nombre'])
        if st.button("Borrar Activo"):
            bid = df_final[df_final['Nombre']==borrar].iloc[0]['id']
            delete_asset_db(bid)
            st.rerun()

# ==============================================================================
# ⚖️ PÁGINA 5: REBALANCEO
# ==============================================================================
elif pagina == "⚖️ Rebalanceo":
    st.title("⚖️ Rebalanceo de Cartera")
    if df_final.empty: st.stop()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Definir Objetivos")
        plazo = st.number_input("Meses para el objetivo", 1, 24, 12)
        
        weights = {}
        total_w = 0
        saved = st.session_state.get('estrategia', {})
        
        for i, row in df_final.iterrows():
            def_w = int(row['Peso %'])
            w = st.number_input(f"{row['Nombre']} (Actual: {def_w}%)", 0, 100, def_w, key=f"w_{i}")
            weights[row['Nombre']] = w
            total_w += w
            
        st.metric("Total Peso Objetivo", f"{total_w}%", delta="Debe ser 100" if total_w!=100 else "Perfecto")
        
        if total_w == 100 and st.button("Calcular Plan"):
            patrimonio = df_final['Valor Acciones'].sum()
            max_cap = 0
            # Algoritmo DCA inteligente: Llenar el cubo que está más vacío
            for _, row in df_final.iterrows():
                tgt_money = (weights[row['Nombre']]/100)
                if tgt_money > 0:
                    implied = row['Valor Acciones'] / tgt_money
                    if implied > max_cap: max_cap = implied
            
            target_pf = max(max_cap, patrimonio)
            monthly = (target_pf - patrimonio) / plazo
            st.session_state['plan'] = {'monthly': monthly, 'weights': weights, 'target': target_pf}

    with col2:
        if 'plan' in st.session_state:
            plan = st.session_state['plan']
            st.subheader("📋 Tu Plan de Aportaciones")
            c1, c2 = st.columns(2)
            c1.metric("Aportación Mensual", f"{plan['monthly']:,.2f} €")
            c2.metric("Objetivo Total", f"{plan['target']:,.2f} €")
            
            reco = []
            for _, row in df_final.iterrows():
                tgt_val = plan['target'] * (plan['weights'][row['Nombre']]/100)
                diff = tgt_val - row['Valor Acciones']
                buy = max(0, diff)
                reco.append({'Activo': row['Nombre'], 'Aportar Total': buy, 'Cuota Mes': buy/plazo})
            
            df_reco = pd.DataFrame(reco)
            fig = px.bar(df_reco, x='Activo', y='Cuota Mes', color='Cuota Mes', title="Qué comprar cada mes (DCA Inteligente)")
            st.plotly_chart(fig, use_container_width=True)
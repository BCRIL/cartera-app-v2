import streamlit as st
from supabase import create_client, Client
import pandas as pd
import yfinance as yf
from yahooquery import search
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time
from textblob import TextBlob # Para análisis de sentimiento

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial Pro", layout="wide", page_icon="🏦")

# --- CONEXIÓN SUPABASE ---
try:
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
except:
    st.error("Faltan los secretos de Supabase.")
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
# Lógica de Google (si se configura)
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
        if st.button("🇬 Google (Si configurado)", type="primary", use_container_width=True):
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
        if st.button("Entrar con Email"):
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
    st.write(f"**{user.user_metadata.get('full_name', user.email)}**")
    if st.button("Salir"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.rerun()
    st.divider()
    pagina = st.radio("Navegación", ["📊 Dashboard & IA", "➕ Añadir Activos", "⚖️ Rebalanceo", "🔮 Proyecciones & Noticias"])

# --- FUNCIONES DE CÁLCULO ---
def calculate_rsi(data, window=14):
    """Calcula el índice de fuerza relativa para detectar sobrecompra/sobreventa"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_sentiment(text):
    """Analiza si un texto es positivo o negativo usando TextBlob"""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1: return "🟢 Positivo"
    elif analysis.sentiment.polarity < -0.1: return "🔴 Negativo"
    else: return "⚪ Neutral"

# --- DB & DATA ---
def get_assets_db():
    resp = supabase.table('assets').select("*").eq('user_id', user.id).execute()
    return pd.DataFrame(resp.data)

def add_asset_db(ticker, nombre, shares, price, platform):
    supabase.table('assets').insert({"user_id": user.id, "ticker": ticker, "nombre": nombre, "shares": shares, "avg_price": price, "platform": platform}).execute()

def delete_asset_db(id_del):
    supabase.table('assets').delete().eq('id', id_del).execute()

df_db = get_assets_db()
df_final = pd.DataFrame()

# Cargar datos financieros completos (1 año) para cálculos avanzados
historical_data = {}
if not df_db.empty:
    tickers = df_db['ticker'].unique().tolist()
    try:
        # Descargamos 1 año para ver tendencias y calcular volatilidad
        history = yf.download(tickers, period="1y", progress=False)['Close']
        
        current_prices = {}
        rsi_values = {}
        volatility = {}
        yearly_return = {}
        
        for t in tickers:
            # Manejo de si es una sola serie o dataframe
            series = history if len(tickers) == 1 else history[t]
            
            # Precio actual
            current_prices[t] = series.iloc[-1]
            
            # RSI (Ultimo valor)
            rsi_series = calculate_rsi(series)
            rsi_values[t] = rsi_series.iloc[-1]
            
            # Volatilidad (Desviación estándar anualizada)
            returns = series.pct_change()
            volatility[t] = returns.std() * np.sqrt(252) * 100 # % anual
            
            # Retorno medio anualizado (CAGR simple)
            yearly_return[t] = returns.mean() * 252 * 100 # % anual
            
    except Exception as e: st.error(f"Error cargando datos: {e}")

    # Construir DataFrame Maestro
    df_db['Precio Actual'] = df_db['ticker'].map(current_prices).fillna(0)
    df_db['RSI'] = df_db['ticker'].map(rsi_values).fillna(50)
    df_db['Volatilidad'] = df_db['ticker'].map(volatility).fillna(0)
    df_db['Retorno Esperado'] = df_db['ticker'].map(yearly_return).fillna(0)
    
    df_db['Valor Acciones'] = df_db['shares'] * df_db['Precio Actual']
    df_db['Dinero Invertido'] = df_db['shares'] * df_db['avg_price']
    df_db['Ganancia'] = df_db['Valor Acciones'] - df_db['Dinero Invertido']
    df_db['Rentabilidad'] = (df_db['Ganancia'] / df_db['Dinero Invertido'] * 100).fillna(0)
    df_final = df_db.rename(columns={'nombre': 'Nombre'})

# ==============================================================================
# 📊 PÁGINA 1: DASHBOARD CON RECOMENDACIONES TÉCNICAS
# ==============================================================================
if pagina == "📊 Dashboard & IA":
    st.title("📊 Tu Cartera Inteligente")
    
    if df_final.empty:
        st.info("Empieza añadiendo activos en el menú lateral.")
    else:
        # KPIs Principales
        patrimonio = df_final['Valor Acciones'].sum()
        ganancia = df_final['Ganancia'].sum()
        rent_total = (ganancia / df_final['Dinero Invertido'].sum() * 100)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Patrimonio", f"{patrimonio:,.2f} €")
        c2.metric("📈 Beneficio Total", f"{ganancia:,.2f} €", f"{rent_total:.2f}%")
        c3.metric("📅 Retorno Esperado (Año)", f"{df_final['Retorno Esperado'].mean():.2f}%", help="Media histórica de tus activos")
        c4.metric("⚡ Volatilidad Cartera", f"{df_final['Volatilidad'].mean():.2f}%", help="Riesgo aproximado (Desv. Std)")
        
        st.divider()
        
        # --- TABLA AVANZADA CON RECOMENDACIONES ---
        st.subheader("🤖 Análisis de Activos & Recomendaciones")
        
        def get_recommendation(row):
            rsi = row['RSI']
            rent = row['Rentabilidad']
            
            if rsi > 70: return "🔴 Vender/Reducir (Sobrecompra)"
            elif rsi < 30: return "🟢 Comprar (Sobreventa)"
            elif rent < -10: return "👀 Vigilar (Pérdidas altas)"
            else: return "⚪ Mantener"

        df_final['Recomendación IA'] = df_final.apply(get_recommendation, axis=1)
        
        # Formatear tabla para visualización
        display_cols = ['Nombre', 'Precio Actual', 'Ganancia', 'Rentabilidad', 'RSI', 'Volatilidad', 'Recomendación IA']
        
        st.dataframe(df_final[display_cols].style.format({
            'Precio Actual': '{:.2f}€', 'Ganancia': '{:+.2f}€', 'Rentabilidad': '{:+.2f}%',
            'RSI': '{:.1f}', 'Volatilidad': '{:.1f}%'
        }).background_gradient(subset=['Rentabilidad'], cmap='RdYlGn', vmin=-20, vmax=20)
          .map(lambda x: 'color: red' if 'Vender' in str(x) else 'color: green' if 'Comprar' in str(x) else '', subset=['Recomendación IA']),
        use_container_width=True)

        st.caption("**Nota sobre RSI:** RSI > 70 indica que el precio ha subido mucho y rápido (posible corrección). RSI < 30 indica que ha bajado mucho (posible rebote).")

        # Gráfico distribución
        fig = px.treemap(df_final, path=['platform', 'Nombre'], values='Valor Acciones', 
                         color='Rentabilidad', color_continuous_scale='RdYlGn', color_continuous_midpoint=0)
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 🔮 PÁGINA 4: PROYECCIONES Y NOTICIAS (NUEVO)
# ==============================================================================
elif pagina == "🔮 Proyecciones & Noticias":
    st.title("🔮 Futuro y Panorama")
    
    if df_final.empty: st.warning("Añade activos primero.")
    else:
        tab_proj, tab_news = st.tabs(["📈 Estimación Futura", "📰 Noticias y Sentimiento"])
        
        with tab_proj:
            st.subheader("Estimación de Crecimiento Compuesto")
            years = st.slider("Años a proyectar", 1, 30, 10)
            avg_return = df_final['Retorno Esperado'].mean() / 100
            
            # Escenarios
            patrimonio_actual = df_final['Valor Acciones'].sum()
            future_val_base = patrimonio_actual * (1 + avg_return)**years
            future_val_bull = patrimonio_actual * (1 + (avg_return + 0.05))**years
            future_val_bear = patrimonio_actual * (1 + (avg_return - 0.05))**years
            
            c_p1, c_p2, c_p3 = st.columns(3)
            c_p1.metric("Escenario Conservador", f"{future_val_bear:,.0f} €")
            c_p2.metric(f"Escenario Base ({avg_return*100:.1f}%)", f"{future_val_base:,.0f} €")
            c_p3.metric("Escenario Optimista", f"{future_val_bull:,.0f} €")
            
            # Gráfico de proyección
            x_years = list(range(years + 1))
            y_base = [patrimonio_actual * (1 + avg_return)**i for i in x_years]
            
            fig_proj = go.Figure()
            fig_proj.add_trace(go.Scatter(x=x_years, y=y_base, mode='lines+markers', name='Crecimiento Esperado', line=dict(color='blue', width=4)))
            fig_proj.update_layout(title="Proyección de tu Patrimonio Actual (Sin nuevas aportaciones)", xaxis_title="Años", yaxis_title="Valor (€)")
            st.plotly_chart(fig_proj, use_container_width=True)
            
        with tab_news:
            st.subheader("📰 Panorama de Mercado")
            selected_ticker = st.selectbox("Selecciona un activo para ver noticias:", df_final['ticker'].unique())
            
            if selected_ticker:
                with st.spinner(f"Analizando noticias de {selected_ticker}..."):
                    try:
                        news_data = yf.Ticker(selected_ticker).news
                        if not news_data:
                            st.info("No hay noticias recientes disponibles en Yahoo Finance.")
                        
                        for item in news_data[:5]: # Mostrar ultimas 5
                            title = item.get('title', 'Sin título')
                            link = item.get('link', '#')
                            publisher = item.get('publisher', 'Desconocido')
                            
                            # ANÁLISIS DE SENTIMIENTO CON IA
                            sentiment_label = get_sentiment(title)
                            
                            with st.expander(f"{sentiment_label} | {title}"):
                                st.write(f"**Fuente:** {publisher}")
                                st.markdown(f"[Leer noticia completa]({link})")
                                if "Positivo" in sentiment_label: st.success("Esta noticia parece optimista.")
                                elif "Negativo" in sentiment_label: st.error("Esta noticia parece pesimista.")
                                else: st.info("Noticia neutral.")
                                
                    except Exception as e:
                        st.error(f"No se pudieron cargar noticias: {e}")

# ==============================================================================
# ➕ PÁGINA 2: AÑADIR (Mantenido igual)
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
                curr_price = yf.Ticker(sym).history(period='1d')['Close'].iloc[-1]
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
            except: st.error("Error obteniendo precio.")

    if not df_final.empty:
        st.divider()
        borrar = st.selectbox("Eliminar activo:", df_final['Nombre'])
        if st.button("Borrar"):
            bid = df_final[df_final['Nombre']==borrar].iloc[0]['id']
            delete_asset_db(bid)
            st.rerun()

# ==============================================================================
# ⚖️ PÁGINA 3: REBALANCEO (Mantenido igual)
# ==============================================================================
elif pagina == "⚖️ Rebalanceo":
    st.title("⚖️ Rebalanceo")
    if df_final.empty: st.stop()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        plazo = st.number_input("Meses para el objetivo", 1, 24, 12)
        total_w = 0
        weights = {}
        saved = st.session_state.get('estrategia', {})
        
        st.write("Define Pesos Objetivo (%):")
        for i, row in df_final.iterrows():
            w = st.number_input(row['Nombre'], 0, 100, saved.get(row['Nombre'], 0), key=i)
            weights[row['Nombre']] = w
            total_w += w
        
        if total_w == 100 and st.button("Calcular Plan"):
            patrimonio = df_final['Valor Acciones'].sum()
            max_cap = 0
            # Lógica simple de rebalanceo (DCA hacia el que va peor)
            for _, row in df_final.iterrows():
                tgt_money = (weights[row['Nombre']]/100)
                if tgt_money > 0:
                    implied_total = row['Valor Acciones'] / tgt_money
                    if implied_total > max_cap: max_cap = implied_total
            
            target_portfolio = max(max_cap, patrimonio)
            monthly_add = (target_portfolio - patrimonio) / plazo
            
            st.session_state['plan'] = {'monthly': monthly_add, 'weights': weights, 'target': target_portfolio}

    with col2:
        if 'plan' in st.session_state:
            plan = st.session_state['plan']
            st.metric("Aportación Mensual Necesaria", f"{plan['monthly']:,.2f} €")
            
            reco_data = []
            for _, row in df_final.iterrows():
                target_val = plan['target'] * (plan['weights'][row['Nombre']]/100)
                diff = target_val - row['Valor Acciones']
                reco_data.append({'Activo': row['Nombre'], 'Comprar Total': max(0, diff), 'Cuota Mes': max(0, diff)/plazo})
            
            df_reco = pd.DataFrame(reco_data)
            fig = px.bar(df_reco, x='Activo', y='Cuota Mes', color='Cuota Mes', title="Plan de Compras Mensual")
            st.plotly_chart(fig, use_container_width=True)
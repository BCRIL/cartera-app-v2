import streamlit as st
from supabase import create_client, Client
import pandas as pd
import yfinance as yf
from yahooquery import search
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import time

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Gestor Patrimonial IA", layout="wide", page_icon="🏦")

# --- CONEXIÓN SUPABASE ---
# Accedemos a las claves de forma segura desde la nube
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# --- GESTIÓN DE SESIÓN ---
if 'user' not in st.session_state: st.session_state['user'] = None
if 'estrategia_guardada' not in st.session_state: st.session_state['estrategia_guardada'] = None

# ==============================================================================
# 🔄 LÓGICA DE LOGIN CON GOOGLE (CORREGIDA PARA PRODUCCIÓN)
# ==============================================================================
query_params = st.query_params
if "code" in query_params and not st.session_state['user']:
    auth_code = query_params["code"]
    try:
        session = supabase.auth.exchange_code_for_session({"auth_code": auth_code})
        st.session_state['user'] = session.user
        st.query_params.clear()
        st.rerun()
    except Exception as e:
        st.error(f"Error de conexión: {e}")

# ==============================================================================
# 🔐 PANTALLA DE LOGIN
# ==============================================================================
if not st.session_state['user']:
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.title("🔐 Acceso a Cartera")
        
        # --- BOTÓN DE GOOGLE ---
        st.markdown("### Acceso Rápido")
        
        if st.button("🇬 Iniciar con Google", type="primary", use_container_width=True):
            try:
                # --- CAMBIO IMPORTANTE AQUÍ ABAJO ---
                data = supabase.auth.sign_in_with_oauth({
                    "provider": "google",
                    "options": {
                        # AQUI PONEMOS TU URL DE STREAMLIT CLOUD
                        "redirect_to": "https://carterapro.streamlit.app" 
                    }
                })
                st.markdown(f'<meta http-equiv="refresh" content="0;url={data.url}">', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generando enlace: {e}")

        st.divider()
        st.caption("O usa credenciales:")
        
        tab1, tab2 = st.tabs(["Entrar", "Registrarse"])
        with tab1:
            email = st.text_input("Email", key="l_email")
            password = st.text_input("Contraseña", type="password", key="l_pass")
            if st.button("Entrar"):
                try:
                    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    st.session_state['user'] = res.user
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")
        with tab2:
            email_reg = st.text_input("Email", key="r_email")
            pass_reg = st.text_input("Contraseña", type="password", key="r_pass")
            if st.button("Crear Cuenta"):
                try:
                    supabase.auth.sign_up({"email": email_reg, "password": pass_reg})
                    st.success("Cuenta creada.")
                except Exception as e: st.error(f"Error: {e}")
    st.stop()

user = st.session_state['user']

# --- SIDEBAR ---
with st.sidebar:
    avatar = user.user_metadata.get('avatar_url', '')
    nombre = user.user_metadata.get('full_name', user.email)
    
    if avatar:
        st.image(avatar, width=50)
    st.info(f"Hola, {nombre}")
    
    if st.button("Cerrar Sesión"):
        supabase.auth.sign_out()
        st.session_state['user'] = None
        st.rerun()
    st.divider()
    pagina = st.radio("Menú", ["📊 Dashboard", "➕ Añadir (Buscador ISIN)", "⚖️ Rebalanceo"])

# --- FUNCIONES DB ---
def get_assets_db():
    resp = supabase.table('assets').select("*").eq('user_id', user.id).execute()
    return pd.DataFrame(resp.data)

def add_asset_db(ticker, nombre, shares, price, platform):
    data = {"user_id": user.id, "ticker": ticker, "nombre": nombre, "shares": shares, "avg_price": price, "platform": platform}
    supabase.table('assets').insert(data).execute()

def delete_asset_db(id_borrar):
    supabase.table('assets').delete().eq('id', id_borrar).execute()

# --- CARGA DATOS ---
df_db = get_assets_db()
precios = {}
df_final = pd.DataFrame()

if not df_db.empty:
    tickers = df_db['ticker'].unique().tolist()
    try:
        data = yf.download(tickers, period="1d", progress=False)['Close']
        for t in tickers:
            if isinstance(data, pd.Series): precios[t] = data.iloc[-1]
            elif not data.empty and t in data.columns: precios[t] = data[t].iloc[-1]
            else: precios[t] = 0.0
    except: pass
    
    df_db['Precio Actual'] = df_db['ticker'].map(precios).fillna(0)
    df_db['Valor Acciones'] = df_db['shares'] * df_db['Precio Actual']
    df_db['Dinero Invertido'] = df_db['shares'] * df_db['avg_price']
    df_db['Ganancia'] = df_db['Valor Acciones'] - df_db['Dinero Invertido']
    df_db['Rentabilidad'] = (df_db['Ganancia'] / df_db['Dinero Invertido'] * 100).fillna(0)
    df_final = df_db.rename(columns={'nombre': 'Nombre'})

# ==============================================================================
# 📊 PÁGINA 1: DASHBOARD
# ==============================================================================
if pagina == "📊 Dashboard":
    c_title, c_btn = st.columns([3, 1])
    with c_title:
        st.title("📊 Cartera en Tiempo Real")
    with c_btn:
        if st.button("🔄 Actualizar Precios Ahora"):
            st.cache_data.clear() 
            st.rerun() 

    if df_final.empty:
        st.warning("Cartera vacía. Ve a 'Añadir por ISIN'.")
    else:
        patrimonio = df_final['Valor Acciones'].sum()
        invertido = df_final['Dinero Invertido'].sum()
        ganancia = patrimonio - invertido
        rent = (ganancia / invertido * 100) if invertido > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("💰 Patrimonio (Valor Hoy)", f"{patrimonio:,.2f} €")
        k2.metric("📈 Beneficio", f"{ganancia:,.2f} €", delta=f"{rent:.2f}%")
        k3.metric("🏦 Dinero que pusiste", f"{invertido:,.2f} €")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df_final, values='Valor Acciones', names='Nombre', hole=0.6, 
                         color_discrete_sequence=px.colors.qualitative.Prism)
            fig.update_layout(showlegend=False)
            fig.update_traces(textposition='outside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            df_sorted = df_final.sort_values('Ganancia')
            fig_bar = px.bar(df_sorted, x='Ganancia', y='Nombre', orientation='h', 
                             color='Ganancia', color_continuous_scale='RdYlGn', text_auto='.2s')
            fig_bar.update_layout(xaxis_title="Beneficio (€)", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.subheader("📋 Detalle")
        cols = ['Nombre', 'shares', 'Precio Actual', 'Dinero Invertido', 'Valor Acciones', 'Ganancia', 'Rentabilidad']
        st.dataframe(df_final[cols].style.format({
            'shares': '{:.4f}', 'Precio Actual': '{:.2f}€', 'Dinero Invertido': '{:.2f}€',
            'Valor Acciones': '{:.2f}€', 'Ganancia': '{:+.2f}€', 'Rentabilidad': '{:+.2f}%'
        }).background_gradient(subset=['Ganancia'], cmap='RdYlGn'), use_container_width=True)

# ==============================================================================
# ➕ PÁGINA 2: AÑADIR (BUSCADOR INTELIGENTE)
# ==============================================================================
elif pagina == "➕ Añadir (Buscador ISIN)":
    st.title("➕ Añadir Inversión")
    
    col_search, col_calc = st.columns([1, 1])
    
    if 'search_results' not in st.session_state: st.session_state['search_results'] = []
    
    with col_search:
        st.subheader("1. Buscar Activo")
        query = st.text_input("Introduce ISIN (ej: IE00B5BMR087) o Nombre:")
        
        if st.button("🔍 Buscar en Mercado"):
            if query:
                with st.spinner("Buscando en mercados globales..."):
                    try:
                        response = search(query)
                        if 'quotes' in response and len(response['quotes']) > 0:
                            st.session_state['search_results'] = response['quotes']
                        else:
                            st.error("No encontrado.")
                            st.session_state['search_results'] = []
                    except: st.error("Error en búsqueda.")

        selected_asset = None
        if st.session_state['search_results']:
            st.markdown("Resultados:")
            options = {}
            for q in st.session_state['search_results']:
                s = q.get('symbol', 'N/A')
                n = q.get('longname', q.get('shortname', s))
                e = q.get('exchange', 'N/A')
                options[f"{n} ({s}) - {e}"] = {'symbol': s, 'name': n}
            
            choice = st.selectbox("Selecciona:", list(options.keys()))
            selected_asset = options[choice]
            st.info(f"Seleccionado: **{selected_asset['name']}**")

    with col_calc:
        st.subheader("2. Tus Números Reales")
        
        if selected_asset:
            ticker_sel = selected_asset['symbol']
            price_now = 0.0
            
            try:
                d = yf.Ticker(ticker_sel).history(period="1d")
                if not d.empty: price_now = d['Close'].iloc[-1]
            except: pass
            
            if price_now > 0:
                st.metric("Precio Mercado HOY", f"{price_now:.2f} €")
                
                st.markdown("Introduce tus datos reales:")
                c_inv, c_val = st.columns(2)
                
                dinero_invertido = c_inv.number_input("💰 Dinero Invertido (Coste)", min_value=0.0, step=100.0)
                valor_actual = c_val.number_input("📈 Valor Actual Total", min_value=0.0, step=100.0)
                platform = st.selectbox("Plataforma", ["MyInvestor", "XTB", "Trade Republic", "Degiro", "Otra"])
                
                if valor_actual > 0:
                    shares_calc = valor_actual / price_now
                    if dinero_invertido > 0:
                        precio_compra_medio = dinero_invertido / shares_calc
                        ganancia = valor_actual - dinero_invertido
                        rentabilidad = (ganancia / dinero_invertido) * 100
                        
                        st.divider()
                        st.write(f"📊 **Análisis:** Tienes **{shares_calc:.4f} part.** compradas a **{precio_compra_medio:.2f}€**.")
                        
                        if st.button("💾 Guardar Posición Real"):
                            add_asset_db(ticker_sel, selected_asset['name'], shares_calc, precio_compra_medio, platform)
                            st.toast("Guardado!")
                            time.sleep(1)
                            st.rerun()
            else:
                st.error("Error obteniendo precio de mercado.")

    st.divider()
    with st.expander("🗑️ Borrar Activos"):
        if not df_final.empty:
            to_del = st.selectbox("Borrar:", df_final['Nombre'].unique())
            id_del = df_final[df_final['Nombre'] == to_del].iloc[0]['id']
            if st.button("Eliminar"):
                delete_asset_db(int(id_del))
                st.rerun()

# ==============================================================================
# ⚖️ PÁGINA 3: REBALANCEO
# ==============================================================================
elif pagina == "⚖️ Rebalanceo":
    st.title("⚖️ Rebalanceo Temporal")
    if df_final.empty: st.warning("Añade activos.")
    else:
        col_t1, col_t2 = st.columns(2)
        with col_t1: unidad = st.selectbox("Unidad", ["Meses", "Semanas"])
        with col_t2: plazo = st.number_input(f"Plazo ({unidad})", 1, 24, 6)
        
        patrimonio = df_final['Valor Acciones'].sum()
        df_final['Peso Actual'] = (df_final['Valor Acciones'] / patrimonio * 100).fillna(0)
        
        df_final['Peso_Int'] = df_final['Peso Actual'].round().astype(int)
        diff = 100 - df_final['Peso_Int'].sum()
        if diff != 0 and not df_final.empty: df_final.at[df_final['Peso Actual'].idxmax(), 'Peso_Int'] += diff

        target_weights = {}
        total_w = 0
        st.subheader("Objetivos (%)")
        col_inp, col_res = st.columns([1, 2])
        
        with col_inp:
            saved = st.session_state.get('estrategia_guardada') or {}
            for i, row in df_final.iterrows():
                def_val = saved.get(row['Nombre'], int(row['Peso_Int']))
                val = st.number_input(f"{row['Nombre']}", 0, 100, def_val, key=f"w_{i}")
                target_weights[row['Nombre']] = val
                total_w += val
            
            st.metric("Total", f"{total_w}%", delta="OK" if total_w==100 else "Ajustar")
            if total_w == 100 and st.button("Guardar Estrategia"):
                st.session_state['estrategia_guardada'] = target_weights
                st.success("Guardado")

        with col_res:
            if total_w == 100:
                max_cap = 0
                for _, row in df_final.iterrows():
                    tgt = target_weights[row['Nombre']] / 100
                    if tgt > 0:
                        imp = row['Valor Acciones'] / tgt
                        if imp > max_cap: max_cap = imp
                
                if max_cap < patrimonio: max_cap = patrimonio
                inyect = max_cap - patrimonio
                cuota = inyect / plazo
                
                reco = []
                txt_comp = ""
                for _, row in df_final.iterrows():
                    tgt_val = max_cap * (target_weights[row['Nombre']]/100)
                    buy = tgt_val - row['Valor Acciones']
                    if buy < 0: buy = 0
                    cuota_fondo = buy / plazo
                    reco.append({"Fondo": row['Nombre'], f"Cuota /{unidad[:-1]}": cuota_fondo, "Total": buy})
                    txt_comp += f"- {row['Nombre']}: {row['Peso Actual']:.1f}% -> {target_weights[row['Nombre']]}%\n"
                
                df_reco = pd.DataFrame(reco)
                
                st.subheader("🛒 Plan de Compra")
                c1, c2 = st.columns(2)
                c1.metric("Cuota Periódica", f"{cuota:,.2f} €")
                c2.metric("Total a Añadir", f"{inyect:,.2f} €")
                
                fig = px.bar(df_reco, x='Fondo', y=f"Cuota /{unidad[:-1]}", 
                             color=f"Cuota /{unidad[:-1]}", color_continuous_scale='Greens', text_auto='.0f')
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("🤖 Consultar a Gemini"):
                    prompt = f"""Actúa como asesor experto.
                    Cartera actual: {patrimonio:.0f}€.
                    Objetivo: Alcanzar {max_cap:.0f}€ en {plazo} {unidad} sin vender activos.
                    
                    CAMBIOS PROPUESTOS:
                    {txt_comp}
                    
                    PLAN DE APORTACIÓN:
                    {cuota:.0f}€ cada {unidad[:-1]} durante {plazo} {unidad}.
                    
                    PREGUNTAS:
                    1. ¿Es sensato este plan de aportación (DCA)?
                    2. ¿Mejora mi diversificación el nuevo reparto de pesos?
                    """
                    st.code(prompt)
            else:
                st.error("Suma 100% para ver el plan.")
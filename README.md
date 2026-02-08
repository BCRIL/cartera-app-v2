# 🏦 Carterapro Ultra — Gestor Patrimonial Inteligente

Aplicación web de gestión patrimonial construida con **Streamlit**, **Supabase** y **yfinance**.

## Funcionalidades

### 📊 Dashboard
- KPIs en tiempo real: patrimonio, P&L, liquidez, Sharpe, Sortino, VaR, Alpha, Beta
- Gráfico de rendimiento base 100 vs S&P 500 con activos individuales
- Treemap de calor por rentabilidad
- Gráfico de P&L por activo
- Drawdown histórico y retornos rodantes (30d)
- Tabla de posiciones con export CSV
- Selector de periodo rápido (1M, 3M, 6M, 1A, 2A)

### 💰 Liquidez
- Ingresos y retiradas con concepto
- Análisis del colchón de seguridad
- Indicador visual del nivel de liquidez

### ➕ Inversiones
- Búsqueda por nombre/ISIN/ticker con yahooquery
- Compra/venta con cálculo automático de acciones y precio medio
- Edición y eliminación de posiciones
- Soporte multi-broker (MyInvestor, XTB, Trade Republic, Degiro, IBKR, eToro, Revolut)

### 📋 Historial
- Log automático de todas las operaciones (compra, venta, ingreso, retiro)
- Filtrado por tipo de operación
- Resumen con flujo neto
- Export CSV

### 🔍 Watchlist
- Seguimiento de activos sin comprarlos
- Info detallada: sector, P/E, beta, dividendos, rango 52 semanas
- Precios en tiempo real

### 🔬 Radiografía de Cartera
- Puntuación de diversificación (Herfindahl-Hirschman)
- Matriz de correlación con alertas de pares altamente correlacionados
- Distribución por sector e industria
- Distribución por broker
- Análisis individual de cada activo con gráfico histórico y precio medio

### 💬 Asesor AI
- Chat con IA (Groq/Llama 3.3 70B)
- Contexto completo de la cartera inyectado automáticamente
- Preguntas rápidas predefinidas
- Historial de conversación

### 🔮 Monte Carlo
- Simulación estocástica con 100-5000 trayectorias
- Bandas de percentiles P10/P25/P50/P75/P90
- Soporte de aportaciones periódicas mensuales
- Probabilidad de pérdida estimada
- Parámetros calibrados desde datos históricos reales

### ⚖️ Rebalanceo
- Manual: pesos objetivo personalizados con cálculo de operaciones necesarias
- Estrategias automáticas: equiponderado, momentum, contrarian, mínima volatilidad
- Gráfico comparativo actual vs objetivo

### 📰 Noticias
- Feed de noticias financieras en sidebar
- Filtro por periodo (hoy/semana)
- Imágenes con fallback

## Requisitos

```
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.streamlit/secrets.toml`:

```toml
SUPABASE_URL = "tu_url"
SUPABASE_KEY = "tu_key"
GROQ_API_KEY = "tu_groq_key"  # Opcional, para Asesor AI
```

## Ejecutar

```bash
streamlit run app.py
```
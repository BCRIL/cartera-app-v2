# 📊 Carterapro — Gestor Patrimonial Inteligente y Gratuito

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://carterapro.streamlit.app)
[![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-ff4b4b.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Controla tus inversiones en tiempo real. 100% gratis, sin comisiones, sin trucos.**

Aplicación web profesional de gestión patrimonial construida con **Streamlit**, **Supabase** y **yfinance**. Diseñada para inversores que quieren tener el control total de su cartera sin pagar por herramientas caras.

🔗 **[Accede gratis → carterapro.streamlit.app](https://carterapro.streamlit.app)**

---

## ¿Por qué Carterapro?

| Característica | Carterapro | Apps de pago |
|---|---|---|
| Dashboard en tiempo real | ✅ Gratis | 💰 10-30€/mes |
| Rebalanceo inteligente | ✅ Solo comprando | 💰 Premium |
| Simulador Monte Carlo | ✅ Incluido | ❌ Raro |
| Multi-broker | ✅ 7 brokers | 💰 Extra |
| Sin publicidad | ✅ | ❌ |
| Open Source | ✅ | ❌ |

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

---

## 🌐 Compartir

Si te gusta Carterapro, ayúdanos a crecer:

- ⭐ Dale una **estrella** a este repositorio
- 🐦 Comparte en [Twitter](https://twitter.com/intent/tweet?text=Gestiona%20tu%20cartera%20de%20inversiones%20gratis%20con%20Carterapro&url=https://carterapro.streamlit.app)
- 💬 Comparte en [WhatsApp](https://wa.me/?text=Mira%20este%20gestor%20de%20cartera%20gratis%20https://carterapro.streamlit.app)
- 📢 Publica en [Reddit r/SpainFIRE](https://reddit.com/r/SpainFIRE), [r/inversiones](https://reddit.com/r/inversiones), [r/eupersonalfinance](https://reddit.com/r/eupersonalfinance)
- 🗣️ Recomiéndalo en foros de [Rankia](https://www.rankia.com/foros), [Bogleheads](https://www.bogleheads.org/forum/) o [Finect](https://www.finect.com/)

## 📄 Licencia

MIT — Úsalo, modifícalo, compártelo libremente.
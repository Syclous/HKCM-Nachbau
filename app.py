import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

# --- SEITEN-KONFIGURATION ---
st.set_page_config(layout="wide", page_title="Custom HKCM Pro Analyzer")
st.title("📈 Custom HKCM-Style Pro Analyzer")
st.write("Elliott-Wellen Zählung, flexible Timeframes, Fibo-Zonen & Squeeze Momentum.")

# --- SEITENLEISTE (AUSWAHL) ---
st.sidebar.header("⚙️ Setup & Parameter")
asset_type = st.sidebar.selectbox("Asset-Klasse", ["Krypto", "Aktien & Rohstoffe"])

if asset_type == "Krypto":
    ticker = st.sidebar.selectbox("Ticker wählen", [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "XLM-USD", "LINK-USD", "ADA-USD"
    ])
else:
    ticker = st.sidebar.selectbox("Ticker wählen", ["AAPL", "NVDA", "TSLA", "GC=F"])

ticker_cleaned = ticker.split(" ")[0]

# --- TIMEFRAME & ZEITRAUM LOGIK ---
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Zeiteinheit (Timeframe)")
timeframe = st.sidebar.selectbox("Intervall wählen", ["1 Tag (1d)", "4 Stunden (4h)", "1 Stunde (1h)"])

if timeframe == "1 Tag (1d)":
    interval = "1d"
    period = st.sidebar.selectbox("Historischer Zeitraum", ["6mo", "1y", "2y"], index=1)
elif timeframe == "4 Stunden (4h)":
    interval = "4h"
    period = st.sidebar.selectbox("Historischer Zeitraum", ["1mo", "3mo", "6mo"], index=1)
else:
    interval = "1h"
    period = st.sidebar.selectbox("Historischer Zeitraum", ["1wk", "1mo", "3mo"], index=1)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Wellen-Erkennung")
pivot_distance = st.sidebar.slider("Pivot-Sensitivität (Kerzen)", min_value=3, max_value=30, value=10)

# --- DATEN BEREITSTELLEN ---
@st.cache_data(ttl=300)
def load_data(symbol, per, inter):
    df = yf.download(symbol, period=per, interval=inter)
    return df

data = load_data(ticker_cleaned, period, interval)

if data.empty:
    st.error("Keine Daten vom Server empfangen. Bitte anderen Zeitraum oder Ticker wählen.")
else:
    # Bereinigung der X-Achse (Datum/Zeit)
    time_index = data.index.strftime('%Y-%m-%d %H:%M').tolist()
    
    # Konvertierung der Preise in reine Python-Listen (behebt alle Plotly-Bugs)
    close_prices = data['Close'].squeeze().fillna(method='ffill').tolist()
    high_prices = data['High'].squeeze().fillna(method='ffill').tolist()
    low_prices = data['Low'].squeeze().fillna(method='ffill').tolist()
    open_prices = data['Open'].squeeze().fillna(method='ffill').tolist()

    # --- SQUEEZE MOMENTUM INDIKATOR BERECHNUNG ---
    close_series = pd.Series(close_prices)
    high_series = pd.Series(high_prices)
    low_series = pd.Series(low_prices)
    
    length = 20
    mult = 2.0
    length_kc = 20
    mult_kc = 1.5

    basis = close_series.rolling(window=length).mean()
    dev = mult * close_series.rolling(window=length).std()
    upper_bb = basis + dev
    lower_bb = basis - dev

    tr = pd.concat([high_series - low_series, 
                    (high_series - close_series.shift()).abs(), 
                    (low_series - close_series.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=length_kc).mean()
    upper_kc = basis + (mult_kc * atr)
    lower_kc = basis - (mult_kc * atr)

    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    
    highest_high = high_series.rolling(window=length).max()
    lowest_low = low_series.rolling(window=length).min()
    avg_hl = (highest_high + lowest_low) / 2
    avg_all = (avg_hl + basis) / 2
    val = close_series - avg_all
    val_smooth = val.rolling(window=length).mean()

    # --- DYNAMISCHER PIVOT / WELLEN ALGORITHMUS ---
    peaks_high, _ = find_peaks(np.array(high_prices), distance=pivot_distance)
    peaks_low, _ = find_peaks(-np.array(low_prices), distance=pivot_distance)

    # Letzte markante Wendepunkte ermitteln
    last_high_idx = peaks_high[-1] if len(peaks_high) > 0 else 0
    last_low_idx = peaks_low[-1] if len(peaks_low) > 0 else 0

    p_high = high_prices[last_high_idx]
    p_low = low_prices[last_low_idx]

    # --- HKCM FIBONACCI BOX & INVALIDIERUNGS-LOGIK ---
    if last_high_idx > last_low_idx:
        diff = p_high - p_low
        zone_top = p_high - (0.50 * diff)       
        zone_bottom = p_high - (0.786 * diff)   
        invalid_level = p_low                  
        zone_color = "rgba(0, 230, 110, 0.18)"
        line_color = "rgba(0, 230, 110, 0.6)"
        zone_name = "HKCM Kaufbereich (Welle 2)"
        probability = "68%"
    else:
        diff = p_high - p_low
        zone_top = p_low + (0.786 * diff)       
        zone_bottom = p_low + (0.50 * diff)     
        invalid_level = p_high                 
        zone_color = "rgba(255, 40, 80, 0.18)"
        line_color = "rgba(255, 40, 80, 0.6)"
        zone_name = "HKCM Zielzone (Short)"
        probability = "62%"

    start_date_box = time_index[min(last_high_idx, last_low_idx)]
    end_date_box = time_index[-1]

    # --- CHART ERSTELLUNG ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])

    # 1. Candlestick Chart
    fig.add_trace(go.Candlestick(
        x=time_index, open=open_prices, high=high_prices, low=low_prices, close=close_prices,
        name=ticker_cleaned
    ), row=1, col=1)

    # Elliott-Wellen Labels einzeichnen (Hier lagen die Probleme - jetzt absolut sicher gelöst)
    if len(peaks_high) > 0:
        fig.add_trace(go.Scatter(
            x=[time_index[p] for p in peaks_high], 
            y=[high_prices[p] for p in peaks_high],
            mode='markers+text', text=["(1)" if i == len(peaks_high)-1 else "" for i in range(len(peaks_high))],
            textposition="top center", font=dict(color="cyan", size=14, family="Arial Black"),
            marker=dict(color='cyan', size=8, symbol='triangle-down'), name='Wellen-Hoch'
        ), row=1, col=1)
        
    if len(peaks_low) > 0:
        fig.add_trace(go.Scatter(
            x=[time_index[p] for p in peaks_low], 
            y=[low_prices[p] for p in peaks_low],
            mode='markers+text', text=["(A)" if i == len(peaks_low)-1 else "" for i in range(len(peaks_low))],
            textposition="bottom center", font=dict(color="magenta", size=14, family="Arial Black"),
            marker=dict(color='magenta', size=8, symbol='triangle-up'), name='Wellen-Tief'
        ), row=1, col=1)

    # HKCM Fibonacci-Box einzeichnen
    fig.add_shape(
        type="rect", x0=start_date_box, y0=zone_bottom, x1=end_date_box, y1=zone_top,
        fillcolor=zone_color, line=dict(color=line_color, width=2), row=1, col=1
    )

    # Textbeschriftung an der Box mit Wahrscheinlichkeit
    fig.add_annotation(
        x=start_date_box, y=zone_top, text=f" {zone_name} ({probability})",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color="white", size=12, family="Arial Black"), row=1, col=1
    )

    # Invalidierungs-Level einzeichnen (Rote gestrichelte Linie)
    fig.add_shape(
        type="line", x0=start_date_box, y0=invalid_level, x1=end_date_box, y1=invalid_level,
        line=dict(color="red", width=2, dash="dash"), row=1, col=1
    )
    fig.add_annotation(
        x=end_date_box, y=invalid_level, text="Invalidierung ",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(color="red", size=11), row=1, col=1
    )

    # 2. Squeeze Momentum Chart
    colors = []
    for i in range(len(val_smooth)):
        if val_smooth.iloc[i] > 0:
            colors.append("#00ff00" if val_smooth.iloc[i] > val_smooth.shift(1).iloc[i] else "#006400")
        else:
            colors.append("#ff0000" if val_smooth.iloc[i] < val_smooth.shift(1).iloc[i] else "#8b0000")

    fig.add_trace(go.Bar(x=time_index, y=val_smooth.tolist(), marker_color=colors, name="Momentum"), row=2, col=1)

    sqz_colors = ["#ffff00" if val else "#000000" for val in sqz_on.tolist()]
    fig.add_trace(go.Scatter(x=time_index, y=[0]*len(data), mode="markers", 
                             marker=dict(color=sqz_colors, size=5), name="Squeeze Punkte"), row=2, col=1)

    fig.update_layout(height=780, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=50, r=50, t=30, b=30))
    fig.update_yaxes(title_text="Preis in USD", row=1, col=1)
    fig.update_yaxes(title_text="Squeeze Mo.", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # --- METRIKEN ---
    st.markdown("### 📊 Aktuelle Zonen-Berechnung & Risikomanagement")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Aktueller Kurs", f"${close_prices[-1]:.2f}")
    c2.metric("Zone Obergrenze (50.0%)", f"${zone_top:.2f}")
    c3.metric("Zone Untergrenze (78.6%)", f"${zone_bottom:.2f}")
    c4.metric("🛑 Invalidierungs-Level", f"${invalid_level:.2f}")

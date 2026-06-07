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
st.write("Dynamische Pivot-Wellenerkennung & Squeeze Momentum Indikator.")

# --- SEITENLEISTE ---
st.sidebar.header("⚙️ Setup & Parameter")
asset_type = st.sidebar.selectbox("Asset-Klasse", ["Krypto", "Aktien & Rohstoffe"])

if asset_type == "Krypto":
    ticker = st.sidebar.selectbox("Ticker wählen", ["BTC-USD", "ETH-USD", "SOL-USD"])
else:
    ticker = st.sidebar.selectbox("Ticker wählen", ["AAPL", "NVDA", "TSLA", "GC=F"])

ticker_cleaned = ticker.split(" ")[0]
period = st.sidebar.selectbox("Historischer Zeitraum", ["3mo", "6mo", "1y", "2y"], index=1)

# Empfindlichkeit für die Wellenerkennung (Pivots)
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Wellen-Erkennung")
pivot_distance = st.sidebar.slider("Pivot-Sensitivität (Tage)", min_value=3, max_value=20, value=7)

# --- DATEN BEREITSTELLEN ---
@st.cache_data(ttl=600)
def load_data(symbol, per):
    df = yf.download(symbol, period=per, interval="1d")
    return df

data = load_data(ticker_cleaned, period)

if data.empty:
    st.error("Keine Daten empfangen.")
else:
    # Multi-Index Bereinigung für neuere yfinance Versionen
    close_prices = data['Close'].squeeze()
    high_prices = data['High'].squeeze()
    low_prices = data['Low'].squeeze()
    open_prices = data['Open'].squeeze()

    # --- SQUEEZE MOMENTUM BERECHNUNG ---
    length = 20
    mult = 2.0
    length_kc = 20
    mult_kc = 1.5

    # Bollinger Bänder
    basis = close_prices.rolling(window=length).mean()
    dev = mult * close_prices.rolling(window=length).std()
    upper_bb = basis + dev
    lower_bb = basis - dev

    # Keltner Kanäle
    tr = pd.concat([high_prices - low_prices, 
                    (high_prices - close_prices.shift()).abs(), 
                    (low_prices - close_prices.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=length_kc).mean()
    upper_kc = basis + (mult_kc * atr)
    lower_kc = basis - (mult_kc * atr)

    # Ist ein Squeeze aktiv?
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    
    # Momentum (Linearer Regressions-Oszillator)
    highest_high = high_prices.rolling(window=length).max()
    lowest_low = low_prices.rolling(window=length).min()
    avg_hl = (highest_high + lowest_low) / 2
    avg_all = (avg_hl + basis) / 2
    val = close_prices - avg_all
    
    # Vereinfachte Annäherung des KC Momentum-Deltas
    val_smooth = val.rolling(window=length).mean()

    # --- PIVOT / WELLEN ALGORITHMUS ---
    # Findet lokale Hochs und Tiefs im Chart
    peaks_high, _ = find_peaks(high_prices.values, distance=pivot_distance)
    peaks_low, _ = find_peaks(-low_prices.values, distance=pivot_distance)

    # Letzten Wendepunkt bestimmen, um relevante HKCM Box zu bauen
    last_high_idx = peaks_high[-1] if len(peaks_high) > 0 else 0
    last_low_idx = peaks_low[-1] if len(peaks_low) > 0 else 0

    p_high = high_prices.iloc[last_high_idx]
    p_low = low_prices.iloc[last_low_idx]

    # Bestimme, ob wir aus einer Aufwärts- oder Abwärtswelle korrigieren
    if last_high_idx > last_low_idx:
        # Aufwärtswelle korrigiert nach unten -> Kaufbereich berechnen
        diff = p_high - p_low
        zone_top = p_high - (0.50 * diff)
        zone_bottom = p_high - (0.786 * diff)
        zone_color = "rgba(0, 200, 100, 0.15)"
        line_color = "rgba(0, 200, 100, 0.5)"
        zone_name = "Kaufzone"
    else:
        # Abwärtswelle korrigiert nach oben -> Verkaufsbereich
        diff = p_high - p_low
        zone_top = p_low + (0.786 * diff)
        zone_bottom = p_low + (0.50 * diff)
        zone_color = "rgba(255, 0, 50, 0.15)"
        line_color = "rgba(255, 0, 50, 0.5)"
        zone_name = "Verkaufszone"

    # --- PLOTLY INTERACTIVE CHART ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, row_heights=[0.7, 0.3])

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=data.index, open=open_prices, high=high_prices, low=low_prices, close=close_prices,
        name=ticker_cleaned
    ), row=1, col=1)

    # Pivots im Chart als Punkte markieren (Elliott-Wellen-Style)
    if len(peaks_high) > 0:
        fig.add_trace(go.Scatter(x=data.index[peaks_high], y=high_prices.iloc[peaks_high],
                                 mode='markers', marker=dict(color='cyan', size=8, symbol='triangle-down'),
                                 name='Lokales Hoch'), row=1, col=1)
    if len(peaks_low) > 0:
        fig.add_trace(go.Scatter(x=data.index[peaks_low], y=low_prices.iloc[peaks_low],
                                 mode='markers', marker=dict(color='magenta', size=8, symbol='triangle-up'),
                                 name='Lokales Tief'), row=1, col=1)

    # Zielzone einzeichnen
    fig.add_shape(
        type="rect", x0=data.index[min(last_high_idx, last_low_idx)], y0=zone_bottom,
        x1=data.index[-1], y1=zone_top,
        fillcolor=zone_color, line=dict(color=line_color, width=1),
        row=1, col=1
    )

    # Squeeze Momentum Balken zeichnen
    colors = []
    for i in range(len(val_smooth)):
        if val_smooth.iloc[i] > 0:
            colors.append("lime" if val_smooth.iloc[i] > val_smooth.shift(1).iloc[i] else "darkgreen")
        else:
            colors.append("red" if val_smooth.iloc[i] < val_smooth.shift(1).iloc[i] else "maroon")

    fig.add_trace(go.Bar(x=data.index, y=val_smooth, marker_color=colors, name="Momentum"), row=2, col=1)

    # Squeeze Punkte (Gelb = Squeeze an, Schwarz = Ausbruch/Aus)
    sqz_colors = ["yellow" if val else "black" for val in sqz_on]
    fig.add_trace(go.Scatter(x=data.index, y=[0]*len(data), mode="markers", 
                             marker=dict(color=sqz_colors, size=4), name="Squeeze Status"), row=2, col=1)

    # Design optimieren
    fig.update_layout(height=750, template="plotly_dark", xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Preis", row=1, col=1)
    fig.update_yaxes(title_text="Squeeze Mo.", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Metriken anzeigen
    c1, c2, c3 = st.columns(3)
    c1.metric("Aktueller Kurs", f"{close_prices.iloc[-1]:.2f}")
    c2.metric(f"{zone_name} (Oben)", f"{zone_top:.2f}")
    c3.metric(f"{zone_name} (Unten)", f"{zone_bottom:.2f}")
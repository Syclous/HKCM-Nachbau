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
st.write("Erweiterte Pivot-Erkennung, präzise Fibonacci-Zonen & Squeeze Momentum.")

# --- SEITENLEISTE (AUSWAHL) ---
st.sidebar.header("⚙️ Setup & Parameter")
asset_type = st.sidebar.selectbox("Asset-Klasse", ["Krypto", "Aktien & Rohstoffe"])

# Hier sind alle deine Wunsch-Coins fix eingebaut!
if asset_type == "Krypto":
    ticker = st.sidebar.selectbox("Ticker wählen", [
        "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "XLM-USD", "LINK-USD", "ADA-USD"
    ])
else:
    ticker = st.sidebar.selectbox("Ticker wählen", ["AAPL", "NVDA", "TSLA", "GC=F"])

ticker_cleaned = ticker.split(" ")[0]
period = st.sidebar.selectbox("Historischer Zeitraum", ["3mo", "6mo", "1y", "2y"], index=1)

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
    st.error("Keine Daten vom Server empfangen. Bitte anderen Zeitraum oder Ticker wählen.")
else:
    # Bereinigung der Kursdaten (Squeeze entfernt eventuelle Multi-Indizes)
    close_prices = data['Close'].squeeze()
    high_prices = data['High'].squeeze()
    low_prices = data['Low'].squeeze()
    open_prices = data['Open'].squeeze()

    # --- SQUEEZE MOMENTUM INDIKATOR BERECHNUNG ---
    length = 20
    mult = 2.0
    length_kc = 20
    mult_kc = 1.5

    # 1. Bollinger Bänder
    basis = close_prices.rolling(window=length).mean()
    dev = mult * close_prices.rolling(window=length).std()
    upper_bb = basis + dev
    lower_bb = basis - dev

    # 2. Keltner Kanäle (True Range Annäherung)
    tr = pd.concat([high_prices - low_prices, 
                    (high_prices - close_prices.shift()).abs(), 
                    (low_prices - close_prices.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=length_kc).mean()
    upper_kc = basis + (mult_kc * atr)
    lower_kc = basis - (mult_kc * atr)

    # Ist ein Squeeze aktiv? (BB innerhalb von KC)
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    
    # Linearer Regressions-Oszillator für das Momentum
    highest_high = high_prices.rolling(window=length).max()
    lowest_low = low_prices.rolling(window=length).min()
    avg_hl = (highest_high + lowest_low) / 2
    avg_all = (avg_hl + basis) / 2
    val = close_prices - avg_all
    val_smooth = val.rolling(window=length).mean()

    # --- DYNAMISCHER PIVOT / WELLEN ALGORITHMUS ---
    peaks_high, _ = find_peaks(high_prices.values, distance=pivot_distance)
    peaks_low, _ = find_peaks(-low_prices.values, distance=pivot_distance)

    # Letzte markante Wendepunkte ermitteln
    last_high_idx = peaks_high[-1] if len(peaks_high) > 0 else 0
    last_low_idx = peaks_low[-1] if len(peaks_low) > 0 else 0

    p_high = high_prices.iloc[last_high_idx]
    p_low = low_prices.iloc[last_low_idx]

    # --- HKCM FIBONACCI BOX LOGIK ---
    if last_high_idx > last_low_idx:
        # Letzte Bewegung war aufwärts -> Korrektur nach unten erwartet (Kaufbereich)
        diff = p_high - p_low
        zone_top = p_high - (0.50 * diff)       # 50% Fibonacci Level
        zone_bottom = p_high - (0.786 * diff)   # 78.6% Fibonacci Level
        zone_color = "rgba(0, 230, 110, 0.18)"
        line_color = "rgba(0, 230, 110, 0.6)"
        zone_name = "HKCM Kaufbereich"
    else:
        # Letzte Bewegung war abwärts -> Korrektur nach oben erwartet (Verkaufsbereich)
        diff = p_high - p_low
        zone_top = p_low + (0.786 * diff)       # 78.6% Fibonacci Level
        zone_bottom = p_low + (0.50 * diff)     # 50% Fibonacci Level
        zone_color = "rgba(255, 40, 80, 0.18)"
        line_color = "rgba(255, 40, 80, 0.6)"
        zone_name = "HKCM Zielzone (Short)"

    # Startpunkt der Box im Chart festlegen (beim letzten Drehpunkt)
    start_date_box = data.index[min(last_high_idx, last_low_idx)]
    end_date_box = data.index[-1]

    # --- CHART ERSTELLUNG (SUBPLOTS) ---
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.75, 0.25])

    # 1. Candlestick Chart (Hauptchart)
    fig.add_trace(go.Candlestick(
        x=data.index, open=open_prices, high=high_prices, low=low_prices, close=close_prices,
        name=ticker_cleaned
    ), row=1, col=1)

    # Wendepunkte einzeichnen (Cyan = Hochs, Magenta = Tiefs)
    if len(peaks_high) > 0:
        fig.add_trace(go.Scatter(x=data.index[peaks_high], y=high_prices.iloc[peaks_high],
                                 mode='markers', marker=dict(color='cyan', size=9, symbol='triangle-down'),
                                 name='Wellen-Hoch'), row=1, col=1)
    if len(peaks_low) > 0:
        fig.add_trace(go.Scatter(x=data.index[peaks_low], y=low_prices.iloc[peaks_low],
                                 mode='markers', marker=dict(color='magenta', size=9, symbol='triangle-up'),
                                 name='Wellen-Tief'), row=1, col=1)

    # Exakte HKCM Fibonacci-Box einzeichnen
    fig.add_shape(
        type="rect", x0=start_date_box, y0=zone_bottom, x1=end_date_box, y1=zone_top,
        fillcolor=zone_color, line=dict(color=line_color, width=2),
        row=1, col=1
    )

    # Textbeschriftung direkt an der Box platzieren
    fig.add_annotation(
        x=start_date_box, y=zone_top, text=f" {zone_name} (50% - 78.6%)",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color="white", size=11), row=1, col=1
    )

    # 2. Squeeze Momentum Chart (Unterer Chart)
    colors = []
    for i in range(len(val_smooth)):
        if val_smooth.iloc[i] > 0:
            colors.append("#00ff00" if val_smooth.iloc[i] > val_smooth.shift(1).iloc[i] else "#006400")
        else:
            colors.append("#ff0000" if val_smooth.iloc[i] < val_smooth.shift(1).iloc[i] else "#8b0000")

    fig.add_trace(go.Bar(x=data.index, y=val_smooth, marker_color=colors, name="Momentum"), row=2, col=1)

    # Squeeze Status-Punkte (Gelb = Energie staut sich / Schwarz = Freigabe)
    sqz_colors = ["#ffff00" if val else "#000000" for val in sqz_on]
    fig.add_trace(go.Scatter(x=data.index, y=[0]*len(data), mode="markers", 
                             marker=dict(color=sqz_colors, size=5), name="Squeeze Punkte"), row=2, col=1)

    # Styling & Dunkles Design (HKCM-Pro-Look)
    fig.update_layout(height=780, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=50, r=50, t=30, b=30))
    fig.update_yaxes(title_text="Preis in USD", row=1, col=1)
    fig.update_yaxes(title_text="Squeeze Mo.", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # --- METRIKEN (UNTER DEM CHART) ---
    st.markdown("### 📊 Aktuelle Zonen-Berechnung")
    c1, c2, c3 = st.columns(3)
    c1.metric("Aktueller Kurs", f"${close_prices.iloc[-1]:.2f}")
    c2.metric("Zone Obergrenze (50.0%)", f"${zone_top:.2f}")
    c3.metric("Zone Untergrenze (78.6%)", f"${zone_bottom:.2f}")

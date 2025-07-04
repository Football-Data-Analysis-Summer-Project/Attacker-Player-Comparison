import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler



# ✅ Page config
st.set_page_config(page_title="🥅 Attacker Radar Chart", layout="wide")

# ✅ Custom CSS for gradient + animation
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #ffecd2 0%, #fcb69f 100%);
        padding: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #4a148c;
        text-align: center;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Title and intro
st.title("🥅 Striker Radar Comparison")
st.markdown("### Visual comparison of Striker stats with normalized radar plots.")

# ✅ Load CSV and clean column names
df = pd.read_csv("Attacker.csv")


features = [
    'Age', 'Club Level', 'Minutes Played', 'Goals', 'xG',
    'xAG', 'shots on target per 90', 'goals per shot', 'SCA', 'Transfer Value'
]

inverse_features = ['Games Missed']
df[features + inverse_features] = df[features + inverse_features].apply(pd.to_numeric, errors='coerce')

# ✅ Normalize data
df_scaled = df.copy()
for feature in features:
    values = df_scaled[feature]
    if feature in inverse_features:
        values = values.max() - values
    if values.max() > 1000:
        values = np.log1p(values)
    df_scaled[feature] = MinMaxScaler().fit_transform(values.values.reshape(-1, 1))

# ✅ Player selection
players = df['Name'].unique()
col1, col2 = st.columns(2)
with col1:
    player1 = st.selectbox("Select First Attacker", players, index=0)
with col2:
    player2 = st.selectbox("Select Second Attacker", players, index=1)

# ✅ Prepare stats for radar chart
p1_stats = df_scaled[df_scaled['Name'] == player1][features].values.flatten()
p2_stats = df_scaled[df_scaled['Name'] == player2][features].values.flatten()

labels = features
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]
p1_stats = np.concatenate((p1_stats, [p1_stats[0]]))
p2_stats = np.concatenate((p2_stats, [p2_stats[0]]))

# ✅ Radar plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# Style
p1_color = "#1e88e5"
p2_color = "#e53935"
ax.plot(angles, p1_stats, color=p1_color, linewidth=2, label=player1)
ax.fill(angles, p1_stats, color=p1_color, alpha=0.3)
ax.plot(angles, p2_stats, color=p2_color, linewidth=2, label=player2)
ax.fill(angles, p2_stats, color=p2_color, alpha=0.3)

ax.set_facecolor('#fefefe')
ax.grid(color="gray", linestyle="dotted", linewidth=0.7)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
plt.yticks([0.2, 0.4, 0.6, 0.8], ["20", "40", "60", "80"], color="gray", size=8)
ax.set_title("Radar Chart: Defender Performance", size=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))

# ✅ Show chart
st.pyplot(fig)

"""
SmartPlate — Home / Landing Page
Navigasi ke halaman lain via sidebar Streamlit.
"""

import streamlit as st

st.set_page_config(
    page_title="SmartPlate | Sistem Nutrisi Indonesia",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main > div { padding-top: 1.5rem; }

.sp-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.6rem 2rem 1.2rem;
    background: linear-gradient(135deg, #1a3a1a 0%, #2d5a1b 60%, #3d7a25 100%);
    border-radius: 16px; margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(45,90,27,0.25);
}
.sp-header-icon { font-size: 2.8rem; }
.sp-header-text h1 {
    font-family: 'DM Serif Display', serif;
    color: #f0f7e8; margin: 0; font-size: 2rem; letter-spacing: -0.5px;
}
.sp-header-text p { color: #a8d87e; margin: 0; font-size: 0.88rem; font-weight: 500; letter-spacing: 0.5px; }

.project-card {
    border-radius: 16px; padding: 28px 24px; height: 100%;
    box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    transition: transform 0.2s;
}
.project-card:hover { transform: translateY(-3px); }
.year1 { background: rgba(45,122,27,0.09); border: 2px solid rgba(45,122,27,0.35); }
.year2 { background: rgba(64,64,160,0.09); border: 2px solid rgba(64,64,160,0.35); }
.card-year { font-size: 0.72rem; font-weight: 700; letter-spacing: 1.5px;
             text-transform: uppercase; margin-bottom: 8px; }
.year1 .card-year { color: #5aaa3a; }
.year2 .card-year { color: #7070cc; }
.card-icon { font-size: 2.6rem; margin-bottom: 10px; }
.card-title {
    font-family: 'DM Serif Display', serif; font-size: 1.5rem;
    margin: 0 0 6px; color: inherit;
}
.card-sub { font-size: 0.84rem; color: inherit; opacity: 0.7; margin-bottom: 14px; }
.feature-list { list-style: none; padding: 0; margin: 0; }
.feature-list li { font-size: 0.82rem; color: inherit; opacity: 0.82; padding: 4px 0;
                   display: flex; align-items: flex-start; gap: 6px; }
.card-nav {
    margin-top: 18px; display: inline-block;
    padding: 8px 18px; border-radius: 8px; font-size: 0.84rem; font-weight: 600;
}
.year1 .card-nav { background: rgba(45,122,27,0.85); color: white; }
.year2 .card-nav { background: rgba(64,64,160,0.85); color: white; }

.flow-banner {
    background: rgba(128,128,200,0.07);
    border: 1.5px solid rgba(128,128,200,0.2); border-radius: 14px;
    padding: 18px 24px; margin: 1.5rem 0;
    display: flex; align-items: center; justify-content: center;
    gap: 16px; flex-wrap: wrap;
}
.flow-step {
    text-align: center; font-size: 0.82rem; color: inherit; opacity: 0.85;
}
.flow-step .step-icon { font-size: 1.6rem; display: block; margin-bottom: 4px; }
.flow-arrow { font-size: 1.4rem; color: inherit; opacity: 0.35; }

.sp-footer {
    text-align: center; padding: 1.5rem; margin-top: 2rem;
    border-top: 1px solid rgba(128,128,128,0.2); color: inherit;
    opacity: 0.55; font-size: 0.78rem; line-height: 1.8;
}
.sp-footer strong { opacity: 1; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sp-header">
    <div class="sp-header-icon">🍽️</div>
    <div class="sp-header-text">
        <h1>SmartPlate</h1>
        <p>SISTEM ANALISIS NUTRISI & OPTIMASI MENU MAKANAN INDONESIA</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Intro ─────────────────────────────────────────────────────────────────────
st.markdown("""
<p style="font-size:0.95rem;color:#444;max-width:760px;line-height:1.7;">
SmartPlate adalah platform end-to-end untuk mendukung keseimbangan gizi masyarakat Indonesia.
Dimulai dari <strong>deteksi dan analisis nutrisi makanan</strong> berbasis foto (Year 1),
dilanjutkan dengan <strong>optimasi menu harian</strong> menggunakan algoritma komputasional (Year 2).
Navigasi ke halaman yang diinginkan melalui sidebar di sebelah kiri.
</p>
""", unsafe_allow_html=True)

# ── Flow banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="flow-banner">
    <div class="flow-step">
        <span class="step-icon">📸</span>Foto Makanan
    </div>
    <div class="flow-arrow">→</div>
    <div class="flow-step">
        <span class="step-icon">🤖</span>YOLOv8 Deteksi
    </div>
    <div class="flow-arrow">→</div>
    <div class="flow-step">
        <span class="step-icon">📊</span>Analisis Nutrisi
    </div>
    <div class="flow-arrow">→</div>
    <div class="flow-step">
        <span class="step-icon">🧬</span>Optimasi Menu
    </div>
    <div class="flow-arrow">→</div>
    <div class="flow-step">
        <span class="step-icon">✅</span>Menu Seimbang
    </div>
</div>
""", unsafe_allow_html=True)

# ── Project Cards ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="project-card year1">
        <div class="card-year">📌 Year 1 Project</div>
        <div class="card-icon">🍽️</div>
        <div class="card-title">SmartPlate Analyzer</div>
        <div class="card-sub">Deteksi & Analisis Nutrisi via YOLOv8 Instance Segmentation</div>
        <ul class="feature-list">
            <li>📸 Upload foto atau gunakan kamera langsung</li>
            <li>🤖 Deteksi 5 kategori: Buah, Karbo, Protein, Sayur, Minuman</li>
            <li>⚖️ Estimasi berat porsi via kalibrasi piring 22 cm</li>
            <li>🧪 Kalkulasi kalori & nutrisi (Energi, Protein, Karbo, Lemak, Serat)</li>
            <li>📊 Perbandingan vs AKG 2019 (Permenkes No. 28 Tahun 2019)</li>
            <li>🥧 Evaluasi komposisi piring vs standar Isi Piringku</li>
        </ul>
        <div class="card-nav">→ Buka di sidebar: SmartPlate</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="project-card year2">
        <div class="card-year">📌 Year 2 Project</div>
        <div class="card-icon">🥗</div>
        <div class="card-title">Optimasi Menu</div>
        <div class="card-sub">Optimasi Menu Harian via Genetic Algorithm & Particle Swarm Optimization</div>
        <ul class="feature-list">
            <li>🧬 Perbandingan GA vs PSO untuk optimasi menu 1 hari</li>
            <li>📅 Generate menu seimbang untuk 7 hari</li>
            <li>📊 Analisis statistik 30 run dengan Paired T-Test</li>
            <li>💰 Optimasi biaya dalam budget Rp 50.000/hari</li>
            <li>🎯 Target AKG: Kalori 1800–2200 kkal, Protein ≥50g</li>
            <li>📈 Kurva konvergensi & confidence interval 95%</li>
        </ul>
        <div class="card-nav">→ Buka di sidebar: Optimasi Menu</div>
    </div>
    """, unsafe_allow_html=True)

# ── Info box ──────────────────────────────────────────────────────────────────
st.markdown("")
st.info("💡 **Cara navigasi:** Gunakan menu di sidebar kiri untuk berpindah antar halaman. "
        "Klik **SmartPlate** untuk analisis foto makanan, atau **Optimasi Menu** untuk generate menu optimal.", icon=None)

# ── Tech stack ────────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.markdown("**🤖 Model**\nYOLOv8l-seg\n(Instance Segmentation)")
c2.markdown("**🧬 Algoritma**\nGenetic Algorithm\nParticle Swarm Opt.")
c3.markdown("**📚 Referensi Gizi**\nAKG 2019 · Isi Piringku\nFatSecret Indonesia")
c4.markdown("**🛠️ Tech Stack**\nPython · Streamlit\nOpenCV · PyTorch")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sp-footer">
    <strong>SmartPlate</strong> — Sistem Analisis Nutrisi & Optimasi Menu Indonesia<br>
    © 2026 Mochamad Faisal Akbar · Universitas Sebelas Maret<br>
    <em>Untuk kebutuhan medis atau diet klinis, konsultasikan dengan ahli gizi profesional.</em>
</div>
""", unsafe_allow_html=True)
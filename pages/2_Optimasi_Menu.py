"""
SmartPlate — Year 2: Optimasi Menu Makanan
Hybrid GA-PSO:
  Phase 1 — GA  : Optimasi KOMBINASI makanan (ruang diskret/kombinatorial)
  Phase 2 — PSO : Optimasi PORSI/JUMLAH untuk kombinasi terpilih (ruang kontinu)

Standalone GA & PSO dipertahankan sebagai pembanding.
Semua fungsi kalkulasi gizi TIDAK diubah dari versi original.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import time
from typing import Dict

st.set_page_config(
    page_title="Optimasi Menu – SmartPlate Year 2",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CSS — konsisten dengan SmartPlate Year 1 + elemen Hybrid baru
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main > div { padding-top: 1.5rem; }

.sp-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.6rem 2rem 1.2rem;
    background: linear-gradient(135deg, #1a1a3e 0%, #2d2d7a 40%, #1a3a1a 80%, #2d5a1b 100%);
    border-radius: 16px; margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(45,45,120,0.28);
}
.sp-header-icon { font-size: 2.8rem; }
.sp-header-text h1 {
    font-family: 'DM Serif Display', serif;
    color: #f0f0ff; margin: 0; font-size: 2rem; letter-spacing: -0.5px;
}
.sp-header-text p { color: #b0b0ff; margin: 0; font-size: 0.88rem; font-weight: 500; letter-spacing: 0.5px; }

.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem; color: #7070dd; margin: 1.2rem 0 0.6rem;
    padding-bottom: 6px; border-bottom: 2px solid rgba(112,112,220,0.4);
}
.result-card {
    background: rgba(255,255,255,0.04); border: 1.5px solid rgba(128,128,200,0.2);
    border-radius: 14px; padding: 18px 20px; margin: 0.6rem 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
}
.winner-badge {
    display: inline-block; background: rgba(45,122,27,0.8); color: white;
    border-radius: 20px; padding: 3px 14px; font-size: 0.8rem; font-weight: 600;
}
.loser-badge {
    display: inline-block; background: rgba(100,100,100,0.6); color: white;
    border-radius: 20px; padding: 3px 14px; font-size: 0.8rem; font-weight: 600;
}
/* ── Hybrid badge & phase cards ─────────────────────────────────────────── */
.hybrid-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(45,45,160,0.85) 0%, rgba(45,122,27,0.85) 100%);
    color: white; border-radius: 20px; padding: 3px 14px;
    font-size: 0.8rem; font-weight: 600;
}
.phase-ga {
    background: rgba(63,63,160,0.09); border-left: 4px solid #5555bb;
    border-radius: 10px; padding: 10px 14px; margin: 8px 0;
}
.phase-pso {
    background: rgba(45,122,27,0.09); border-left: 4px solid #2d7a1b;
    border-radius: 10px; padding: 10px 14px; margin: 8px 0;
}
.phase-label {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 4px; display: block;
}
.phase-ga .phase-label  { color: #7070dd; }
.phase-pso .phase-label { color: #3d8f28; }
.phase-desc { font-size: 0.82rem; color: inherit; opacity: 0.85; }

.info-box {
    background: rgba(64,64,160,0.1); border: 1.5px solid rgba(100,100,200,0.25);
    border-left: 4px solid #5555bb; border-radius: 10px;
    padding: 12px 16px; font-size: 0.83rem; color: inherit; margin: 0.6rem 0; opacity: 0.9;
}
.warn-box {
    background: rgba(245,195,50,0.1); border: 1.5px solid rgba(245,195,50,0.28);
    border-left: 4px solid #d4a017; border-radius: 10px;
    padding: 12px 16px; font-size: 0.83rem; color: inherit; margin: 0.6rem 0; opacity: 0.9;
}
.sidebar-label {
    font-size: 0.75rem; color: inherit; opacity: 0.55; text-transform: uppercase;
    letter-spacing: 0.6px; font-weight: 600; margin-bottom: 4px;
}
.sp-footer {
    text-align: center; padding: 1.5rem; margin-top: 2rem;
    border-top: 1px solid rgba(128,128,128,0.2); color: inherit;
    opacity: 0.55; font-size: 0.78rem; line-height: 1.8;
}
.sp-footer strong { opacity: 1; }

@media (prefers-color-scheme: dark) {
    .section-title { color: #9090ff; border-bottom-color: #5050aa; }
    .result-card { background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.1); }
    .info-box { background: rgba(64,64,160,0.15); border-color: rgba(100,100,200,0.3); color: #c0c0ff; }
    .warn-box { background: rgba(245,195,50,0.1); border-color: rgba(245,195,50,0.3); color: #e0c060; }
    .phase-ga  { background: rgba(63,63,160,0.15); }
    .phase-pso { background: rgba(45,122,27,0.15); }
}

div[data-testid="stExpander"] { border: 1.5px solid #c0c0ee !important; border-radius: 10px !important; }
.stButton > button {
    background: linear-gradient(135deg, #2d2d7a 0%, #2d5a1b 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 0.95rem !important; padding: 0.55rem 0 !important;
    box-shadow: 0 4px 14px rgba(45,45,120,0.3) !important; transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# ▼▼▼  KODE KALKULASI ORIGINAL — TIDAK DIUBAH  ▼▼▼
# ==============================================================================

FOOD_DATABASE = {
    'buah': [
        {'nama': 'Pisang',    'kalori': 89,  'protein': 1.1, 'karbo': 22.8, 'harga': 8000},
        {'nama': 'Apel',      'kalori': 52,  'protein': 0.3, 'karbo': 14,   'harga': 25000},
        {'nama': 'Jeruk',     'kalori': 47,  'protein': 0.9, 'karbo': 12,   'harga': 12000},
        {'nama': 'Mangga',    'kalori': 60,  'protein': 0.8, 'karbo': 15,   'harga': 15000},
        {'nama': 'Pepaya',    'kalori': 43,  'protein': 0.5, 'karbo': 11,   'harga': 8000},
        {'nama': 'Semangka',  'kalori': 30,  'protein': 0.6, 'karbo': 8,    'harga': 6000},
        {'nama': 'Anggur',    'kalori': 69,  'protein': 0.7, 'karbo': 18,   'harga': 35000},
        {'nama': 'Melon',     'kalori': 34,  'protein': 0.8, 'karbo': 8,    'harga': 10000},
        {'nama': 'Pir',       'kalori': 57,  'protein': 0.4, 'karbo': 15,   'harga': 30000},
        {'nama': 'Nanas',     'kalori': 50,  'protein': 0.5, 'karbo': 13,   'harga': 8000},
    ],
    'karbohidrat': [
        {'nama': 'Nasi Putih',  'kalori': 130, 'protein': 2.7, 'karbo': 28, 'harga': 13000},
        {'nama': 'Roti Tawar',  'kalori': 265, 'protein': 9,   'karbo': 49, 'harga': 18000},
        {'nama': 'Mie Instant', 'kalori': 188, 'protein': 4.5, 'karbo': 27, 'harga': 12000},
        {'nama': 'Kentang',     'kalori': 77,  'protein': 2,   'karbo': 17, 'harga': 10000},
        {'nama': 'Singkong',    'kalori': 160, 'protein': 1.4, 'karbo': 38, 'harga': 6000},
        {'nama': 'Jagung',      'kalori': 86,  'protein': 3.3, 'karbo': 19, 'harga': 8000},
        {'nama': 'Ubi',         'kalori': 86,  'protein': 1.6, 'karbo': 20, 'harga': 9000},
        {'nama': 'Pasta',       'kalori': 158, 'protein': 5.8, 'karbo': 31, 'harga': 22000},
        {'nama': 'Oatmeal',     'kalori': 68,  'protein': 2.4, 'karbo': 12, 'harga': 30000},
        {'nama': 'Roti Gandum', 'kalori': 247, 'protein': 13,  'karbo': 41, 'harga': 25000},
    ],
    'protein': [
        {'nama': 'Ayam',         'kalori': 165, 'protein': 31,  'karbo': 0,    'harga': 38000},
        {'nama': 'Telur',        'kalori': 155, 'protein': 13,  'karbo': 1.1,  'harga': 28000},
        {'nama': 'Tempe',        'kalori': 193, 'protein': 19,  'karbo': 9,    'harga': 10000},
        {'nama': 'Tahu',         'kalori': 76,  'protein': 8,   'karbo': 1.9,  'harga': 8000},
        {'nama': 'Ikan Lele',    'kalori': 168, 'protein': 26,  'karbo': 0,    'harga': 30000},
        {'nama': 'Daging Sapi',  'kalori': 250, 'protein': 26,  'karbo': 0,    'harga': 130000},
        {'nama': 'Ikan Tongkol', 'kalori': 144, 'protein': 23,  'karbo': 0,    'harga': 35000},
        {'nama': 'Udang',        'kalori': 99,  'protein': 24,  'karbo': 0.2,  'harga': 90000},
        {'nama': 'Kacang Merah', 'kalori': 127, 'protein': 8.7, 'karbo': 23,   'harga': 18000},
        {'nama': 'Kacang Hijau', 'kalori': 347, 'protein': 24,  'karbo': 63,   'harga': 22000},
    ],
    'sayur': [
        {'nama': 'Bayam',     'kalori': 23, 'protein': 2.9, 'karbo': 3.6, 'harga': 7000},
        {'nama': 'Kangkung',  'kalori': 19, 'protein': 3,   'karbo': 3,   'harga': 6000},
        {'nama': 'Wortel',    'kalori': 41, 'protein': 0.9, 'karbo': 10,  'harga': 10000},
        {'nama': 'Brokoli',   'kalori': 34, 'protein': 2.8, 'karbo': 7,   'harga': 18000},
        {'nama': 'Kol',       'kalori': 25, 'protein': 1.3, 'karbo': 6,   'harga': 8000},
        {'nama': 'Tomat',     'kalori': 18, 'protein': 0.9, 'karbo': 3.9, 'harga': 12000},
        {'nama': 'Timun',     'kalori': 15, 'protein': 0.7, 'karbo': 3.6, 'harga': 7000},
        {'nama': 'Terong',    'kalori': 25, 'protein': 1,   'karbo': 6,   'harga': 9000},
        {'nama': 'Buncis',    'kalori': 31, 'protein': 1.8, 'karbo': 7,   'harga': 12000},
        {'nama': 'Labu Siam', 'kalori': 19, 'protein': 0.8, 'karbo': 4.5, 'harga': 7000},
    ],
    'minuman': [
        {'nama': 'Susu Sapi',    'kalori': 61,  'protein': 3.2, 'karbo': 4.8, 'harga': 20000},
        {'nama': 'Teh Manis',    'kalori': 30,  'protein': 0,   'karbo': 8,   'harga': 10000},
        {'nama': 'Jus Jeruk',    'kalori': 45,  'protein': 0.7, 'karbo': 10,  'harga': 8000},
        {'nama': 'Air Kelapa',   'kalori': 19,  'protein': 0.7, 'karbo': 3.7, 'harga': 7000},
        {'nama': 'Susu Kedelai', 'kalori': 54,  'protein': 3.3, 'karbo': 6,   'harga': 12000},
        {'nama': 'Yogurt',       'kalori': 59,  'protein': 3.5, 'karbo': 4.7, 'harga': 25000},
        {'nama': 'Kopi Susu',    'kalori': 38,  'protein': 2,   'karbo': 5,   'harga': 8000},
        {'nama': 'Jus Alpukat',  'kalori': 160, 'protein': 2,   'karbo': 8.5, 'harga': 15000},
        {'nama': 'Air Putih',    'kalori': 0,   'protein': 0,   'karbo': 0,   'harga': 4000},
        {'nama': 'Jus Tomat',    'kalori': 17,  'protein': 0.8, 'karbo': 3.9, 'harga': 10000},
    ],
}

ALL_FOODS      = []
CATEGORY_START = {}
idx = 0
for category, foods in FOOD_DATABASE.items():
    CATEGORY_START[category] = idx
    for food in foods:
        food['category'] = category
        ALL_FOODS.append(food)
    idx += len(foods)
NUM_FOODS = len(ALL_FOODS)

TARGETS = {
    'kalori':  {'min': 1800, 'max': 2200, 'ideal': 2000},
    'protein': {'min': 50,   'max': 80,   'ideal': 60},
    'karbo':   {'min': 250,  'max': 350,  'ideal': 300},
}
CATEGORY_MIN_PORTIONS  = {'buah': 150, 'karbohidrat': 300, 'protein': 150, 'sayur': 200, 'minuman': 600}
CATEGORY_MAX_PORTIONS  = {'buah': 400, 'karbohidrat': 500, 'protein': 300, 'sayur': 500, 'minuman': 900}
MAX_BUDGET             = 50000
MIN_PORTION_PER_FOOD   = 50
MAX_ITEMS_PER_CATEGORY = {'buah': 3, 'karbohidrat': 2, 'protein': 3, 'sayur': 4, 'minuman': 2}
STAPLE_FOOD_INDICES    = [10, 11]
MIN_STAPLE_PORTION     = 200


# ── Calculation functions (unchanged) ─────────────────────────────────────────

def calculate_nutrition(portions: np.ndarray) -> Dict:
    total_kalori = total_protein = total_karbo = total_cost = 0
    for i, portion in enumerate(portions):
        food = ALL_FOODS[i]
        total_kalori  += food['kalori']  * (portion / 100)
        total_protein += food['protein'] * (portion / 100)
        total_karbo   += food['karbo']   * (portion / 100)
        total_cost    += food['harga']   * (portion / 1000)
    return {'kalori': total_kalori, 'protein': total_protein,
            'karbo': total_karbo, 'cost': total_cost}


def calculate_penalty(portions: np.ndarray, nutrition: Dict) -> float:
    penalty = 0
    if nutrition['kalori'] < TARGETS['kalori']['min']:
        penalty += (TARGETS['kalori']['min'] - nutrition['kalori']) ** 2 * 1.0
    if nutrition['kalori'] > TARGETS['kalori']['max']:
        penalty += (nutrition['kalori'] - TARGETS['kalori']['max']) ** 2 * 1.0
    if nutrition['protein'] < TARGETS['protein']['min']:
        penalty += (TARGETS['protein']['min'] - nutrition['protein']) ** 2 * 2
    if nutrition['karbo'] < TARGETS['karbo']['min']:
        penalty += (TARGETS['karbo']['min'] - nutrition['karbo']) ** 2 * 0.3
    if nutrition['karbo'] > TARGETS['karbo']['max']:
        penalty += (nutrition['karbo'] - TARGETS['karbo']['max']) ** 2 * 0.3
    if nutrition['cost'] > MAX_BUDGET:
        penalty += (nutrition['cost'] - MAX_BUDGET) ** 2 * 0.01
    for category, min_p in CATEGORY_MIN_PORTIONS.items():
        s = CATEGORY_START[category]; e = s + len(FOOD_DATABASE[category])
        tot = np.sum(portions[s:e])
        if tot < min_p:
            w = 2.0 if category == 'minuman' else 0.8
            penalty += (min_p - tot) ** 2 * w
    for category, max_p in CATEGORY_MAX_PORTIONS.items():
        s = CATEGORY_START[category]; e = s + len(FOOD_DATABASE[category])
        tot = np.sum(portions[s:e])
        if tot > max_p:
            w = 2.0 if category == 'minuman' else 0.5
            penalty += (tot - max_p) ** 2 * w
    for i, portion in enumerate(portions):
        if 0 < portion < MIN_PORTION_PER_FOOD:
            penalty += (MIN_PORTION_PER_FOOD - portion) ** 2 * 0.1
    for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
        s = CATEGORY_START[category]; e = s + len(FOOD_DATABASE[category])
        n = np.sum(portions[s:e] >= MIN_PORTION_PER_FOOD)
        if n > max_items:
            penalty += (n - max_items) ** 2 * 10
    staple_total = sum(portions[i] for i in STAPLE_FOOD_INDICES)
    if staple_total < MIN_STAPLE_PORTION:
        penalty += (MIN_STAPLE_PORTION - staple_total) ** 2 * 1.0
    return penalty


def fitness_function(portions: np.ndarray) -> float:
    nutrition = calculate_nutrition(portions)
    penalty   = calculate_penalty(portions, nutrition)
    return 1000 / (nutrition['cost'] + penalty + 1)


def validate_final_solution(solution: np.ndarray) -> Dict:
    violations = []
    awkward = [(i, p) for i, p in enumerate(solution) if 0 < p < MIN_PORTION_PER_FOOD]
    if awkward:
        violations.append(f"C6: {len(awkward)} porsi < 50g")
    for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
        s = CATEGORY_START[category]; e = s + len(FOOD_DATABASE[category])
        n = np.sum(solution[s:e] >= MIN_PORTION_PER_FOOD)
        if n > max_items:
            violations.append(f"C7: {category} = {n} item (maks {max_items})")
    staple = sum(solution[i] for i in STAPLE_FOOD_INDICES)
    if staple < MIN_STAPLE_PORTION:
        violations.append(f"C8: Staple food {staple:.1f}g (min 200g)")
    ms = CATEGORY_START['minuman']; me = ms + len(FOOD_DATABASE['minuman'])
    mt = np.sum(solution[ms:me])
    if mt < 600 or mt > 900:
        violations.append(f"C5B: Minuman {mt:.1f}ml (target 600–900ml)")
    nutr = calculate_nutrition(solution)
    if not (TARGETS['kalori']['min'] <= nutr['kalori'] <= TARGETS['kalori']['max']):
        violations.append(f"C1: Kalori {nutr['kalori']:.0f} (target 1800–2200)")
    if nutr['protein'] < TARGETS['protein']['min']:
        violations.append(f"C2: Protein {nutr['protein']:.1f}g (min 50g)")
    if not (TARGETS['karbo']['min'] <= nutr['karbo'] <= TARGETS['karbo']['max']):
        violations.append(f"C3: Karbo {nutr['karbo']:.1f}g (target 250–350)")
    if nutr['cost'] > MAX_BUDGET:
        violations.append(f"C4: Biaya Rp{nutr['cost']:,.0f} (maks Rp50.000)")
    return {'all_constraints_met': len(violations) == 0,
            'violations': violations, 'num_violations': len(violations),
            'nutrition': nutr}


def aggressive_repair_solution(individual: np.ndarray, max_iterations: int = 5) -> np.ndarray:
    repaired = individual.copy()
    ms = CATEGORY_START['minuman']; me = ms + len(FOOD_DATABASE['minuman'])
    for _ in range(max_iterations):
        changed = False
        for i in range(NUM_FOODS):
            if 0 < repaired[i] < MIN_PORTION_PER_FOOD:
                repaired[i] = 0; changed = True
        for category, max_items in MAX_ITEMS_PER_CATEGORY.items():
            s = CATEGORY_START[category]; e = s + len(FOOD_DATABASE[category])
            active = [(i, repaired[i]) for i in range(s, e) if repaired[i] >= MIN_PORTION_PER_FOOD]
            while len(active) > max_items:
                active.sort(key=lambda x: x[1])
                idx_rem, rem_p = active[0]
                repaired[idx_rem] = 0; changed = True
                remaining = active[1:]
                tot_rem = sum(p for _, p in remaining)
                for idx2, p2 in remaining:
                    if tot_rem > 0:
                        repaired[idx2] += rem_p * (p2 / tot_rem)
                active = [(i, repaired[i]) for i in range(s, e) if repaired[i] >= MIN_PORTION_PER_FOOD]
        staple = sum(repaired[i] for i in STAPLE_FOOD_INDICES)
        if staple < MIN_STAPLE_PORTION:
            repaired[10] += MIN_STAPLE_PORTION - staple; changed = True
        mt = np.sum(repaired[ms:me])
        if mt < 600:
            repaired[ms + 8] += 600 - mt; changed = True
        elif mt > 900:
            excess = mt - 900
            items = sorted([(i, repaired[i]) for i in range(ms, me) if repaired[i] >= MIN_PORTION_PER_FOOD],
                           key=lambda x: -x[1])
            for idx2, p2 in items:
                if excess <= 0: break
                red = min(p2 - MIN_PORTION_PER_FOOD, excess)
                if red > 0: repaired[idx2] -= red; excess -= red; changed = True
        n = calculate_nutrition(repaired)
        if n['kalori'] < TARGETS['kalori']['min']:
            diff = TARGETS['kalori']['min'] - n['kalori']
            repaired[10] += (diff / ALL_FOODS[10]['kalori']) * 100 * 1.02; changed = True
        elif n['kalori'] > TARGETS['kalori']['max']:
            excess = n['kalori'] - TARGETS['kalori']['max']
            hc = sorted([(i, ALL_FOODS[i]['kalori'] * (repaired[i]/100), repaired[i])
                         for i in range(NUM_FOODS)
                         if repaired[i] >= MIN_PORTION_PER_FOOD and i not in STAPLE_FOOD_INDICES
                         and ALL_FOODS[i]['kalori'] * (repaired[i]/100) > 50],
                        key=lambda x: -x[1])
            for idx2, kcal, p2 in hc:
                if excess <= 0: break
                kpg = ALL_FOODS[idx2]['kalori'] / 100
                red = min(excess / kpg, p2 - MIN_PORTION_PER_FOOD)
                if red > 0: repaired[idx2] -= red; excess -= red * kpg; changed = True
        if n['protein'] < TARGETS['protein']['min']:
            diff = TARGETS['protein']['min'] - n['protein']
            repaired[12] += (diff / ALL_FOODS[12]['protein']) * 100 * 1.03; changed = True
        if n['karbo'] < TARGETS['karbo']['min']:
            diff = TARGETS['karbo']['min'] - n['karbo']
            repaired[10] += (diff / ALL_FOODS[10]['karbo']) * 100 * 1.02; changed = True
        elif n['karbo'] > TARGETS['karbo']['max']:
            excess = n['karbo'] - TARGETS['karbo']['max']
            ks = CATEGORY_START['karbohidrat']; ke = ks + len(FOOD_DATABASE['karbohidrat'])
            ki = sorted([(i, ALL_FOODS[i]['karbo']*(repaired[i]/100), repaired[i])
                         for i in range(ks, ke)
                         if repaired[i] >= MIN_PORTION_PER_FOOD and i not in STAPLE_FOOD_INDICES],
                        key=lambda x: -x[1])
            for idx2, kc, p2 in ki:
                if excess <= 0: break
                kpg = ALL_FOODS[idx2]['karbo'] / 100
                red = min(excess / kpg, p2 - MIN_PORTION_PER_FOOD)
                if red > 0: repaired[idx2] -= red; excess -= red * kpg; changed = True
        if not changed:
            break
        mt2 = np.sum(repaired[ms:me])
        if mt2 < 600: repaired[ms+8] += 600 - mt2
        elif mt2 > 900:
            excess2 = mt2 - 900
            for i in range(ms, me):
                if repaired[i] >= MIN_PORTION_PER_FOOD and excess2 > 0:
                    red = min(repaired[i] - MIN_PORTION_PER_FOOD, excess2)
                    repaired[i] -= red; excess2 -= red
    return repaired


# ==============================================================================
# GENETIC ALGORITHM (unchanged from original)
# ==============================================================================

class GeneticAlgorithm:
    def __init__(self, pop_size=30, generations=50, pc=0.8, pm=0.2):
        self.pop_size = pop_size; self.generations = generations
        self.pc = pc; self.pm = pm
        self.best_fitness_history = []; self.avg_fitness_history = []

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            ind = np.zeros(NUM_FOODS)
            for category in FOOD_DATABASE:
                s = CATEGORY_START[category]; csz = len(FOOD_DATABASE[category])
                max_i = MAX_ITEMS_PER_CATEGORY[category]
                n_i = np.random.randint(1, min(max_i + 1, 3))
                sel = np.random.choice(csz, n_i, replace=False)
                totals = {'karbohidrat': (250,400), 'protein': (100,200),
                          'sayur': (150,300), 'buah': (150,250), 'minuman': (600,900)}
                lo, hi = totals.get(category, (200,400))
                total = np.random.uniform(lo, hi)
                portions = np.random.dirichlet(np.ones(n_i)) * total
                for sidx, p in zip(sel, portions):
                    ind[s + sidx] = max(p, MIN_PORTION_PER_FOOD)
            population.append(ind)
        return population

    def tournament_selection(self, population, fitnesses, k=3):
        cands = np.random.choice(len(population), k, replace=False)
        return population[cands[np.argmax([fitnesses[i] for i in cands])]].copy()

    def crossover(self, p1, p2):
        if np.random.rand() > self.pc:
            return p1.copy(), p2.copy()
        pt = np.random.randint(1, NUM_FOODS - 1)
        return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])

    def mutate(self, individual):
        m = individual.copy()
        for i in range(NUM_FOODS):
            if np.random.rand() < self.pm:
                if m[i] >= MIN_PORTION_PER_FOOD:
                    m[i] += np.random.normal(0, m[i] * 0.15)
                    m[i] = 0 if m[i] < MIN_PORTION_PER_FOOD else np.clip(m[i], MIN_PORTION_PER_FOOD, 500)
                elif m[i] == 0 and np.random.rand() < 0.15:
                    m[i] = np.random.uniform(MIN_PORTION_PER_FOOD, 150)
        return m

    def _basic_repair(self, ind):
        r = ind.copy()
        for i in range(NUM_FOODS):
            if 0 < r[i] < MIN_PORTION_PER_FOOD: r[i] = 0
        for cat, mx in MAX_ITEMS_PER_CATEGORY.items():
            s = CATEGORY_START[cat]; e = s + len(FOOD_DATABASE[cat])
            items = sorted([(i, r[i]) for i in range(s, e) if r[i] >= MIN_PORTION_PER_FOOD], key=lambda x: x[1])
            for i in range(max(0, len(items) - mx)): r[items[i][0]] = 0
        staple = sum(r[i] for i in STAPLE_FOOD_INDICES)
        if staple < MIN_STAPLE_PORTION: r[10] += MIN_STAPLE_PORTION - staple
        ms = CATEGORY_START['minuman']; me = ms + len(FOOD_DATABASE['minuman'])
        mt = np.sum(r[ms:me])
        if mt < 600: r[ms+8] += 600 - mt
        elif mt > 900:
            excess = mt - 900
            for i in sorted(range(ms, me), key=lambda x: -r[x]):
                if excess <= 0: break
                red = min(r[i] - MIN_PORTION_PER_FOOD, excess)
                if red > 0: r[i] -= red; excess -= red
        return r

    def evolve(self, verbose=False):
        self.best_fitness_history = []; self.avg_fitness_history = []
        population = self.initialize_population()
        best_solution = None; best_fitness = -np.inf
        start_time = time.time()
        for gen in range(self.generations):
            fitnesses = [fitness_function(ind) for ind in population]
            gi = np.argmax(fitnesses)
            if fitnesses[gi] > best_fitness:
                best_fitness = fitnesses[gi]; best_solution = population[gi].copy()
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))
            elite_n   = max(2, int(0.1 * self.pop_size))
            elites    = [population[i].copy() for i in np.argsort(fitnesses)[-elite_n:]]
            new_pop   = elites[:]
            while len(new_pop) < self.pop_size:
                c1, c2 = self.crossover(self.tournament_selection(population, fitnesses),
                                         self.tournament_selection(population, fitnesses))
                new_pop.append(self._basic_repair(self.mutate(c1)))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._basic_repair(self.mutate(c2)))
            population = new_pop[:self.pop_size]
        best_solution = aggressive_repair_solution(best_solution, max_iterations=5)
        return best_solution, fitness_function(best_solution), {
            'best_history': self.best_fitness_history,
            'avg_history':  self.avg_fitness_history,
            'time':         time.time() - start_time,
            'validation':   validate_final_solution(best_solution),
        }


# ==============================================================================
# PARTICLE SWARM OPTIMIZATION (unchanged from original)
# ==============================================================================

class ParticleSwarmOptimization:
    def __init__(self, n_particles=30, iterations=50, w=0.7, c1=1.5, c2=1.5):
        self.n_particles = n_particles; self.iterations = iterations
        self.w = w; self.c1 = c1; self.c2 = c2
        self.best_fitness_history = []; self.avg_fitness_history = []

    def _basic_repair(self, ind):
        r = ind.copy()
        for i in range(NUM_FOODS):
            if 0 < r[i] < MIN_PORTION_PER_FOOD: r[i] = 0
        for cat, mx in MAX_ITEMS_PER_CATEGORY.items():
            s = CATEGORY_START[cat]; e = s + len(FOOD_DATABASE[cat])
            items = sorted([(i, r[i]) for i in range(s, e) if r[i] >= MIN_PORTION_PER_FOOD], key=lambda x: x[1])
            for i in range(max(0, len(items) - mx)): r[items[i][0]] = 0
        staple = sum(r[i] for i in STAPLE_FOOD_INDICES)
        if staple < MIN_STAPLE_PORTION: r[10] += MIN_STAPLE_PORTION - staple
        ms = CATEGORY_START['minuman']; me = ms + len(FOOD_DATABASE['minuman'])
        mt = np.sum(r[ms:me])
        if mt < 600: r[ms+8] += 600 - mt
        elif mt > 900:
            excess = mt - 900
            for i in sorted(range(ms, me), key=lambda x: -r[x]):
                if excess <= 0: break
                red = min(r[i] - MIN_PORTION_PER_FOOD, excess)
                if red > 0: r[i] -= red; excess -= red
        return r

    def optimize(self, verbose=False):
        self.best_fitness_history = []; self.avg_fitness_history = []
        positions  = np.random.uniform(0, 300, (self.n_particles, NUM_FOODS))
        velocities = np.random.uniform(-50, 50, (self.n_particles, NUM_FOODS))
        pbest_pos  = positions.copy()
        pbest_fit  = np.array([fitness_function(p) for p in positions])
        gi         = np.argmax(pbest_fit)
        gbest_pos  = pbest_pos[gi].copy(); gbest_fit = pbest_fit[gi]
        start_time = time.time()
        for _ in range(self.iterations):
            for i in range(self.n_particles):
                fit = fitness_function(positions[i])
                if fit > pbest_fit[i]: pbest_fit[i] = fit; pbest_pos[i] = positions[i].copy()
                if fit > gbest_fit:    gbest_fit = fit;    gbest_pos  = positions[i].copy()
                r1, r2 = np.random.rand(NUM_FOODS), np.random.rand(NUM_FOODS)
                velocities[i] = (self.w * velocities[i]
                                 + self.c1 * r1 * (pbest_pos[i] - positions[i])
                                 + self.c2 * r2 * (gbest_pos   - positions[i]))
                positions[i] = np.clip(positions[i] + velocities[i], 0, 500)
                positions[i] = self._basic_repair(positions[i])
            self.best_fitness_history.append(gbest_fit)
            self.avg_fitness_history.append(np.mean(pbest_fit))
        gbest_pos = aggressive_repair_solution(gbest_pos, max_iterations=5)
        return gbest_pos, fitness_function(gbest_pos), {
            'best_history': self.best_fitness_history,
            'avg_history':  self.avg_fitness_history,
            'time':         time.time() - start_time,
            'validation':   validate_final_solution(gbest_pos),
        }


# ==============================================================================
# ▲▲▲  AKHIR KODE KALKULASI ORIGINAL  ▲▲▲
# ==============================================================================


# ==============================================================================
# ▼▼▼  HYBRID GA-PSO — BARU  ▼▼▼
# ==============================================================================

def _hybrid_repair_fast(sol: np.ndarray) -> np.ndarray:
    """
    Lightweight repair untuk iterasi PSO pada Hybrid.
    Hanya menangani constraint kritis (minuman, staple food, minimum portion).
    Constraint kombinasi per-kategori TIDAK dicek di sini karena sudah
    dikunci oleh GA pada Phase 1.
    """
    r = sol.copy()
    # Zero-kan porsi di bawah minimum
    r[r < MIN_PORTION_PER_FOOD] = 0
    # Pastikan staple food cukup
    staple = sum(r[i] for i in STAPLE_FOOD_INDICES)
    if staple < MIN_STAPLE_PORTION:
        r[10] += MIN_STAPLE_PORTION - staple
    # Pastikan minuman dalam rentang 600–900 ml
    ms = CATEGORY_START['minuman']; me = ms + len(FOOD_DATABASE['minuman'])
    mt = np.sum(r[ms:me])
    if mt < 600:
        r[ms + 8] += 600 - mt
    elif mt > 900:
        excess = mt - 900
        for i in sorted(range(ms, me), key=lambda x: -r[x]):
            if excess <= 0: break
            red = min(max(r[i] - MIN_PORTION_PER_FOOD, 0), excess)
            if red > 0: r[i] -= red; excess -= red
    return r


class HybridGAPSO:
    """
    Hybrid GA-PSO dengan pemisahan objective yang jelas:

    ┌─────────────────────────────────────────────────────────────────┐
    │  PHASE 1 — Genetic Algorithm: Optimasi KOMBINASI Makanan        │
    │    • GA mengeksplorasi ruang kombinatorial (makanan apa saja    │
    │      yang dipilih, per kategori).                               │
    │    • Kromosom = vektor porsi; aktif/tidaknya item (≥50g / 0g)  │
    │      merepresentasikan keputusan kombinasi.                     │
    │    • Output: best chromosome → food combination (selected mask) │
    ├─────────────────────────────────────────────────────────────────┤
    │  PHASE 2 — Particle Swarm Optimization: Optimasi PORSI/JUMLAH  │
    │    • PSO menerima kombinasi makanan terpilih dari GA.           │
    │    • Hanya mengoptimasi jumlah/porsi (nilai kontinu) pada item  │
    │      yang sudah dipilih GA — ruang pencarian jauh lebih kecil.  │
    │    • Inertia menurun linear (w_start → 0.4) untuk konvergensi  │
    │      yang lebih presisi.                                        │
    │    • Particle pertama di-seed dari solusi GA sebagai warm-start.│
    └─────────────────────────────────────────────────────────────────┘
    """

    def __init__(self,
                 ga_pop_size: int = 30,   ga_generations: int = 25,
                 ga_pc: float = 0.8,      ga_pm: float = 0.2,
                 pso_particles: int = 30, pso_iterations: int = 25,
                 pso_w: float = 0.7,      pso_c1: float = 1.5, pso_c2: float = 1.5):
        self.ga_pop_size    = ga_pop_size
        self.ga_generations = ga_generations
        self.ga_pc          = ga_pc
        self.ga_pm          = ga_pm
        self.pso_particles  = pso_particles
        self.pso_iterations = pso_iterations
        self.pso_w          = pso_w
        self.pso_c1         = pso_c1
        self.pso_c2         = pso_c2
        # History per-phase
        self.ga_best_history:  list = []
        self.ga_avg_history:   list = []
        self.pso_best_history: list = []
        self.pso_avg_history:  list = []

    # ------------------------------------------------------------------
    def run(self, verbose: bool = False):
        start_time = time.time()

        # ══════════════════════════════════════════════════════════════
        # PHASE 1 — GA: Mencari Kombinasi Makanan Terbaik
        # ══════════════════════════════════════════════════════════════
        ga = GeneticAlgorithm(
            pop_size=self.ga_pop_size, generations=self.ga_generations,
            pc=self.ga_pc, pm=self.ga_pm
        )
        ga_solution, ga_fitness, ga_history = ga.evolve(verbose=verbose)
        self.ga_best_history = ga_history['best_history']
        self.ga_avg_history  = ga_history['avg_history']

        # Ekstrak "kombinasi" terpilih dari solusi GA terbaik
        # Food item dianggap terpilih jika porsinya ≥ MIN_PORTION_PER_FOOD
        selected_mask    = ga_solution >= MIN_PORTION_PER_FOOD
        selected_indices = np.where(selected_mask)[0]
        n_selected       = len(selected_indices)

        # Edge case: tidak ada makanan aktif (seharusnya tidak terjadi setelah repair)
        if n_selected == 0:
            best_sol = aggressive_repair_solution(ga_solution)
            bf       = fitness_function(best_sol)
            self.pso_best_history = [bf] * self.pso_iterations
            self.pso_avg_history  = [bf] * self.pso_iterations
            return best_sol, bf, self._build_history(start_time, best_sol, selected_indices)

        # ══════════════════════════════════════════════════════════════
        # PHASE 2 — PSO: Mengoptimasi Porsi pada Kombinasi Terpilih
        # Dimensi pencarian PSO = n_selected (bukan NUM_FOODS penuh),
        # sehingga jauh lebih efisien dan presisi.
        # ══════════════════════════════════════════════════════════════
        ga_portions = ga_solution[selected_indices]          # (n_selected,)

        # Batas bawah & atas porsi per item (berdasarkan kategori)
        lo_bounds = np.full(n_selected, float(MIN_PORTION_PER_FOOD))
        hi_bounds = np.array([
            float(CATEGORY_MAX_PORTIONS.get(ALL_FOODS[i]['category'], 500))
            for i in selected_indices
        ])

        # ── Inisialisasi Particle ──────────────────────────────────────
        positions  = np.zeros((self.pso_particles, n_selected))
        velocities = np.random.uniform(-40, 40, (self.pso_particles, n_selected))
        # Particle ke-0: warm-start dari solusi GA (memastikan setidaknya satu solusi baik)
        positions[0] = np.clip(ga_portions, lo_bounds, hi_bounds)
        # Particle lain: perturbasi Gaussian di sekitar solusi GA
        for p in range(1, self.pso_particles):
            noise      = np.random.normal(0, ga_portions * 0.25)
            positions[p] = np.clip(ga_portions + noise, lo_bounds, hi_bounds)

        # ── Helper: rekonstruksi solusi penuh dari vektor porsi tereduksi ──
        def _make_full(reduced: np.ndarray) -> np.ndarray:
            sol = np.zeros(NUM_FOODS)
            sol[selected_indices] = np.clip(reduced, lo_bounds, hi_bounds)
            return _hybrid_repair_fast(sol)

        def _pso_fit(reduced: np.ndarray) -> float:
            return fitness_function(_make_full(reduced))

        # ── Inisialisasi personal best & global best ───────────────────
        pbest_pos = positions.copy()
        pbest_fit = np.array([_pso_fit(p) for p in positions])
        gi        = np.argmax(pbest_fit)
        gbest_pos = pbest_pos[gi].copy()
        gbest_fit = float(pbest_fit[gi])

        self.pso_best_history = []
        self.pso_avg_history  = []

        # ── Main PSO Loop (inertia menurun linear: w_start → 0.4) ─────
        w_start = self.pso_w
        w_end   = 0.4

        for it in range(self.pso_iterations):
            # Inertia weight decay — eksplorasi di awal, eksploitasi di akhir
            w_it = w_start - (w_start - w_end) * (it / max(self.pso_iterations - 1, 1))

            for i in range(self.pso_particles):
                r1 = np.random.rand(n_selected)
                r2 = np.random.rand(n_selected)
                velocities[i] = (
                    w_it * velocities[i]
                    + self.pso_c1 * r1 * (pbest_pos[i] - positions[i])
                    + self.pso_c2 * r2 * (gbest_pos   - positions[i])
                )
                positions[i] = np.clip(positions[i] + velocities[i], lo_bounds, hi_bounds)
                fit_i = _pso_fit(positions[i])
                if fit_i > pbest_fit[i]:
                    pbest_fit[i] = fit_i
                    pbest_pos[i] = positions[i].copy()
                if fit_i > gbest_fit:
                    gbest_fit = fit_i
                    gbest_pos = positions[i].copy()

            self.pso_best_history.append(float(gbest_fit))
            self.pso_avg_history.append(float(np.mean(pbest_fit)))

        # ── Rekonstruksi & aggressive repair solusi akhir ─────────────
        best_solution = _make_full(gbest_pos)
        best_solution = aggressive_repair_solution(best_solution, max_iterations=5)
        final_fitness = fitness_function(best_solution)

        return best_solution, final_fitness, self._build_history(start_time, best_solution, selected_indices)

    def _build_history(self, start_time: float, solution: np.ndarray, selected_indices: np.ndarray) -> Dict:
        return {
            'ga_best_history':  self.ga_best_history,
            'ga_avg_history':   self.ga_avg_history,
            'pso_best_history': self.pso_best_history,
            'pso_avg_history':  self.pso_avg_history,
            'time':             time.time() - start_time,
            'validation':       validate_final_solution(solution),
            'n_selected':       int(len(selected_indices)),
            'ga_generations':   self.ga_generations,
            'pso_iterations':   self.pso_iterations,
        }

# ==============================================================================
# ▲▲▲  AKHIR HYBRID GA-PSO  ▲▲▲
# ==============================================================================


# ==============================================================================
# DISPLAY HELPERS
# ==============================================================================

CATEGORY_EMOJI = {'buah':'🍎','karbohidrat':'🍚','protein':'🍗','sayur':'🥗','minuman':'🥤'}
CATEGORY_COLOR = {'buah':'#E8544A','karbohidrat':'#F5A623','protein':'#C0392B','sayur':'#27AE60','minuman':'#4A90D9'}


def _check_target(val, lo, hi=None):
    if hi is None: return "✅" if val >= lo else "❌"
    return "✅" if lo <= val <= hi else "❌"


def display_menu_st(solution: np.ndarray, label: str = ""):
    """Tampilkan detail menu + metrik nutrisi."""
    nutrition  = calculate_nutrition(solution)
    validation = validate_final_solution(solution)
    if validation['all_constraints_met']:
        st.success("✅ Semua constraint terpenuhi")
    else:
        for v in validation['violations']:
            st.warning(f"⚠️ {v}")
    c1, c2, c3, c4 = st.columns(4)
    kal_ok = TARGETS['kalori']['min'] <= nutrition['kalori'] <= TARGETS['kalori']['max']
    pro_ok = nutrition['protein'] >= TARGETS['protein']['min']
    kar_ok = TARGETS['karbo']['min'] <= nutrition['karbo'] <= TARGETS['karbo']['max']
    bud_ok = nutrition['cost'] <= MAX_BUDGET
    c1.metric("🔥 Kalori",      f"{nutrition['kalori']:.0f} kkal",
              delta="Sesuai" if kal_ok else "Di luar target", delta_color="normal" if kal_ok else "inverse")
    c2.metric("💪 Protein",     f"{nutrition['protein']:.1f} g",
              delta="≥50g ✓" if pro_ok else "<50g ✗",  delta_color="normal" if pro_ok else "inverse")
    c3.metric("🌾 Karbohidrat", f"{nutrition['karbo']:.1f} g",
              delta="Sesuai" if kar_ok else "Di luar target", delta_color="normal" if kar_ok else "inverse")
    c4.metric("💰 Biaya",       f"Rp {nutrition['cost']:,.0f}",
              delta="Dalam budget" if bud_ok else "Melebihi budget", delta_color="normal" if bud_ok else "inverse")
    rows = []
    for category, foods in FOOD_DATABASE.items():
        s = CATEGORY_START[category]
        for i, food in enumerate(foods):
            portion = solution[s + i]
            if portion >= MIN_PORTION_PER_FOOD:
                rows.append({
                    'Kategori':       f"{CATEGORY_EMOJI[category]} {category.capitalize()}",
                    'Makanan':        food['nama'],
                    'Porsi (g/ml)':   round(portion, 1),
                    'Biaya (Rp)':     int(food['harga'] * portion / 1000),
                    'Kalori (kkal)':  round(food['kalori'] * portion / 100, 1),
                })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Tidak ada makanan yang dipilih.")


def display_three_way_comparison_st(ga_res: Dict, pso_res: Dict, hybrid_res: Dict):
    """Tabel perbandingan 3 algoritma: GA vs PSO vs Hybrid GA-PSO."""
    st.markdown('<div class="section-title">📊 Perbandingan GA vs PSO vs Hybrid GA-PSO</div>',
                unsafe_allow_html=True)
    ga_n = ga_res['nutrition']; pso_n = pso_res['nutrition']; hyb_n = hybrid_res['nutrition']

    def best3(a, b, c, higher=True):
        vals = {'GA': a, 'PSO': b, 'Hybrid': c}
        w = max(vals, key=vals.get) if higher else min(vals, key=vals.get)
        return f"🏆 {w}"

    rows = [
        {"Metrik": "Best Fitness",
         "GA": f"{ga_res['fitness']:.6f}", "PSO": f"{pso_res['fitness']:.6f}",
         "Hybrid GA-PSO": f"{hybrid_res['fitness']:.6f}",
         "Terbaik": best3(ga_res['fitness'], pso_res['fitness'], hybrid_res['fitness'], True)},
        {"Metrik": "Biaya Total (Rp)",
         "GA": f"{ga_n['cost']:,.0f}", "PSO": f"{pso_n['cost']:,.0f}",
         "Hybrid GA-PSO": f"{hyb_n['cost']:,.0f}",
         "Terbaik": best3(ga_n['cost'], pso_n['cost'], hyb_n['cost'], False)},
        {"Metrik": "Waktu Komputasi (s)",
         "GA": f"{ga_res['time']:.2f}", "PSO": f"{pso_res['time']:.2f}",
         "Hybrid GA-PSO": f"{hybrid_res['time']:.2f}",
         "Terbaik": best3(ga_res['time'], pso_res['time'], hybrid_res['time'], False)},
        {"Metrik": "Kalori (kkal)",
         "GA": f"{ga_n['kalori']:.1f}", "PSO": f"{pso_n['kalori']:.1f}",
         "Hybrid GA-PSO": f"{hyb_n['kalori']:.1f}", "Terbaik": "—"},
        {"Metrik": "Protein (g)",
         "GA": f"{ga_n['protein']:.1f}", "PSO": f"{pso_n['protein']:.1f}",
         "Hybrid GA-PSO": f"{hyb_n['protein']:.1f}", "Terbaik": "—"},
        {"Metrik": "Karbohidrat (g)",
         "GA": f"{ga_n['karbo']:.1f}", "PSO": f"{pso_n['karbo']:.1f}",
         "Hybrid GA-PSO": f"{hyb_n['karbo']:.1f}", "Terbaik": "—"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def display_convergence_all_st(ga_hist: Dict, pso_hist: Dict, hybrid_hist: Dict):
    """
    Kurva konvergensi 3-way:
    Kiri  — GA vs PSO vs Hybrid (gabungan, untuk perbandingan langsung)
    Kanan — Hybrid two-phase breakdown (GA phase | PSO phase)
    """
    plt.rcParams.update({
        'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#f8f8f8',
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='white')

    # ── Subplot kiri: Perbandingan langsung ───────────────────────────────────
    ax1 = axes[0]
    ax1.plot(ga_hist['best_history'],  label='GA',  lw=2, color='#3a3aaa',
             marker='o', ms=3, markevery=max(1, len(ga_hist['best_history'])//10))
    ax1.plot(pso_hist['best_history'], label='PSO', lw=2, color='#2d7a1b',
             marker='s', ms=3, markevery=max(1, len(pso_hist['best_history'])//10))

    hyb_combined = hybrid_hist['ga_best_history'] + hybrid_hist['pso_best_history']
    ax1.plot(hyb_combined, label='Hybrid GA-PSO', lw=2.5, color='#c05000',
             marker='^', ms=3, markevery=max(1, len(hyb_combined)//10), linestyle='--')
    n_ga_steps = len(hybrid_hist['ga_best_history'])
    ax1.axvline(x=n_ga_steps, color='#c05000', linestyle=':', alpha=0.5, lw=1.5,
                label=f'PSO phase mulai (step {n_ga_steps})')

    ax1.set_xlabel('Step (Generasi / Iterasi)', fontsize=11, color='#1a1a1a')
    ax1.set_ylabel('Fitness', fontsize=11, color='#1a1a1a')
    ax1.set_title('Best Fitness: GA vs PSO vs Hybrid', fontsize=12, fontweight='bold', color='#1a1a1a')
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)
    for sp in ax1.spines.values(): sp.set_edgecolor('#ccc')

    # ── Subplot kanan: Hybrid two-phase breakdown ─────────────────────────────
    ax2 = axes[1]
    n_ga  = len(hybrid_hist['ga_best_history'])
    n_pso = len(hybrid_hist['pso_best_history'])
    x_ga  = list(range(n_ga))
    x_pso = list(range(n_ga, n_ga + n_pso))

    # GA Phase
    ax2.plot(x_ga, hybrid_hist['ga_best_history'], label='GA Phase — Best (Kombinasi)',
             lw=2, color='#3a3aaa', marker='o', ms=4,
             markevery=max(1, n_ga//8))
    ax2.plot(x_ga, hybrid_hist['ga_avg_history'],  label='GA Phase — Avg',
             lw=1.5, color='#3a3aaa', alpha=0.4, linestyle='--')

    # PSO Phase
    if n_pso > 0:
        ax2.plot(x_pso, hybrid_hist['pso_best_history'], label='PSO Phase — Best (Porsi)',
                 lw=2, color='#2d7a1b', marker='s', ms=4,
                 markevery=max(1, n_pso//8))
        ax2.plot(x_pso, hybrid_hist['pso_avg_history'],  label='PSO Phase — Avg',
                 lw=1.5, color='#2d7a1b', alpha=0.4, linestyle='--')

    # Divider garis transisi
    ax2.axvline(x=n_ga, color='gray', linestyle='--', alpha=0.7, lw=2)

    # Shading per phase
    ymin_, ymax_ = ax2.get_ylim()
    ax2.axvspan(0,    n_ga,         alpha=0.05, color='#3a3aaa')
    ax2.axvspan(n_ga, n_ga + n_pso, alpha=0.05, color='#2d7a1b')

    # Anotasi phase
    ax2.text(n_ga * 0.5,  ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else 0,
             'Phase 1\nGA', ha='center', va='bottom', fontsize=8, color='#3a3aaa', alpha=0.7)
    if n_pso > 0:
        ax2.text(n_ga + n_pso * 0.5, ax2.get_ylim()[0] if ax2.get_ylim()[0] != 0 else 0,
                 'Phase 2\nPSO', ha='center', va='bottom', fontsize=8, color='#2d7a1b', alpha=0.7)

    ax2.set_xlabel('Step', fontsize=11, color='#1a1a1a')
    ax2.set_ylabel('Fitness', fontsize=11, color='#1a1a1a')
    ax2.set_title('Hybrid: Two-Phase Convergence Breakdown', fontsize=12, fontweight='bold', color='#1a1a1a')
    ax2.legend(fontsize=8, loc='lower right'); ax2.grid(True, alpha=0.3)
    for sp in ax2.spines.values(): sp.set_edgecolor('#ccc')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    plt.rcdefaults()


def display_convergence_st(ga_hist: Dict, pso_hist: Dict):
    """Kurva konvergensi standar GA vs PSO (digunakan pada weekly & stats)."""
    plt.rcParams.update({
        'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#f8f8f8',
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')
    for ax, key, title in [
        (axes[0], 'best_history', 'Best Fitness Convergence'),
        (axes[1], 'avg_history',  'Average Fitness Convergence'),
    ]:
        ax.plot(ga_hist[key],  label='GA',  lw=2, color='#3a3aaa', marker='o', ms=3, markevery=5)
        ax.plot(pso_hist[key], label='PSO', lw=2, color='#2d7a1b', marker='s', ms=3, markevery=5)
        ax.set_xlabel('Generation / Iteration', fontsize=11, color='#1a1a1a')
        ax.set_ylabel('Fitness', fontsize=11, color='#1a1a1a')
        ax.set_title(title, fontsize=13, fontweight='bold', color='#1a1a1a')
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        for spine in ax.spines.values(): spine.set_edgecolor('#ccc')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    plt.rcdefaults()


# ── Session state init ────────────────────────────────────────────────────────
for key in ('single_result', 'weekly_result', 'stats_result'):
    if key not in st.session_state:
        st.session_state[key] = None


# ==============================================================================
# SIDEBAR
# ==============================================================================

def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.8rem 0 0.4rem;">
            <div style="font-size:2rem;">🥗</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.15rem;
                        background:linear-gradient(90deg,#5555cc,#2d7a1b);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:bold;">
                Optimasi Menu
            </div>
            <div style="font-size:0.7rem;color:#888;letter-spacing:0.5px;">HYBRID GA-PSO · SMARTPLATE YEAR 2</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        # ── Parameter GA ──────────────────────────────────────────────
        st.markdown("**🧬 Parameter GA** *(Standalone & Hybrid Phase 1)*")
        ga_pop = st.slider("Population Size", 10, 50, 30, 5)
        ga_gen = st.slider("Generations",     20, 100, 50, 10)
        ga_pc  = st.slider("Crossover Rate",  0.5, 1.0, 0.8, 0.05)
        ga_pm  = st.slider("Mutation Rate",   0.05, 0.5, 0.2, 0.05)

        st.divider()

        # ── Parameter PSO ─────────────────────────────────────────────
        st.markdown("**🐝 Parameter PSO** *(Standalone & Hybrid Phase 2)*")
        pso_n  = st.slider("Jumlah Particle", 10, 50, 30, 5)
        pso_it = st.slider("Iterations",      20, 100, 50, 10)
        pso_w  = st.slider("Inertia (w)",     0.3, 1.0, 0.7, 0.05)
        pso_c1 = st.slider("Cognitive (c1)",  0.5, 2.5, 1.5, 0.1)
        pso_c2 = st.slider("Social (c2)",     0.5, 2.5, 1.5, 0.1)

        st.divider()

        # ── Parameter Hybrid ─────────────────────────────────────────
        st.markdown("**⚡ Hybrid GA-PSO — Budget per Phase**")
        st.markdown("""
        <div class="info-box" style="font-size:0.75rem;padding:8px 12px;margin:4px 0 8px;">
        Atur berapa generasi/iterasi yang dialokasikan ke masing-masing phase.
        Total budget ≈ Phase 1 + Phase 2 (setara dengan standalone GA atau PSO).
        </div>
        """, unsafe_allow_html=True)
        hybrid_ga_gen = st.slider("Phase 1 — GA Generations",  5, 80, 25, 5, key='h_ga_gen',
                                   help="Generasi GA untuk eksplorasi kombinasi makanan")
        hybrid_pso_it = st.slider("Phase 2 — PSO Iterations",  5, 80, 25, 5, key='h_pso_it',
                                   help="Iterasi PSO untuk fine-tune porsi makanan")
        st.caption(f"Total budget Hybrid: **{hybrid_ga_gen + hybrid_pso_it} steps** "
                   f"(GA {hybrid_ga_gen} + PSO {hybrid_pso_it})")

        st.divider()
        seed = st.number_input("Random Seed", value=42, step=1)

        st.divider()
        st.markdown('<div class="sidebar-label">Target Nutrisi Harian</div>', unsafe_allow_html=True)
        for lbl, val in [("Kalori","1800–2200 kkal"),("Protein","≥ 50 g"),
                          ("Karbohidrat","250–350 g"),("Budget","≤ Rp 50.000")]:
            a, b = st.columns(2); a.caption(lbl); b.markdown(f"**{val}**")

        st.divider()
        st.markdown('<div class="sidebar-label">Referensi</div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-size:0.72rem;color:#aaa;line-height:1.6;">
        Permenkes No. 28/2019 · TKPI 2017 ·
        Holland (1975) · Kennedy & Eberhart (1995) ·
        Kao & Zahara (2008) — Hybrid GA-PSO
        </div>""", unsafe_allow_html=True)

    ga_p = dict(pop_size=ga_pop, generations=ga_gen, pc=ga_pc, pm=ga_pm)
    pso_p = dict(n_particles=pso_n, iterations=pso_it, w=pso_w, c1=pso_c1, c2=pso_c2)
    hybrid_p = dict(
        ga_pop_size=ga_pop,   ga_generations=hybrid_ga_gen,
        ga_pc=ga_pc,          ga_pm=ga_pm,
        pso_particles=pso_n,  pso_iterations=hybrid_pso_it,
        pso_w=pso_w,          pso_c1=pso_c1, pso_c2=pso_c2,
    )
    return ga_p, pso_p, hybrid_p, int(seed)


# ==============================================================================
# TAB 1 — Optimasi 1 Hari (GA + PSO + Hybrid)
# ==============================================================================

def tab_single_day(ga_p, pso_p, hybrid_p, seed):
    # Penjelasan arsitektur Hybrid
    st.markdown("""
    <div class="info-box">
    <strong>⚡ Hybrid GA-PSO</strong> — Dua fase dengan objective berbeda:
    </div>
    """, unsafe_allow_html=True)
    col_ph1, col_ph2 = st.columns(2)
    with col_ph1:
        st.markdown("""
        <div class="phase-ga">
        <span class="phase-label">🧬 Phase 1 — GA: Kombinasi Makanan</span>
        <span class="phase-desc">GA mengeksplorasi ruang kombinatorial — <em>makanan apa saja</em>
        yang masuk ke menu. Setiap kromosom merepresentasikan keputusan seleksi item.
        Crossover & mutasi memungkinkan eksplorasi kombinasi yang beragam.</span>
        </div>
        """, unsafe_allow_html=True)
    with col_ph2:
        st.markdown("""
        <div class="phase-pso">
        <span class="phase-label">🐝 Phase 2 — PSO: Optimasi Porsi</span>
        <span class="phase-desc">PSO menerima kombinasi terbaik dari GA, lalu mengoptimasi
        <em>berapa banyak</em> porsi tiap item secara kontinu. Ruang pencarian jauh lebih
        kecil (hanya item aktif), sehingga konvergensi lebih presisi.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("🚀  Jalankan Optimasi 1 Hari (GA + PSO + Hybrid)", use_container_width=True):
        np.random.seed(seed)
        placeholder = st.empty()

        with placeholder.container():
            with st.spinner("🧬 Menjalankan Genetic Algorithm…"):
                ga  = GeneticAlgorithm(**ga_p)
                gs, gf, gh = ga.evolve()
            with st.spinner("🐝 Menjalankan Particle Swarm Optimization…"):
                pso = ParticleSwarmOptimization(**pso_p)
                ps, pf, ph = pso.optimize()
            with st.spinner("⚡ Menjalankan Hybrid GA-PSO (Phase 1: GA Kombinasi → Phase 2: PSO Porsi)…"):
                hyb = HybridGAPSO(**hybrid_p)
                hs, hf, hh = hyb.run()

        placeholder.empty()
        st.session_state.single_result = {
            'ga':  {'solution': gs, 'fitness': gf, 'history': gh,
                    'nutrition': calculate_nutrition(gs), 'time': gh['time']},
            'pso': {'solution': ps, 'fitness': pf, 'history': ph,
                    'nutrition': calculate_nutrition(ps), 'time': ph['time']},
            'hyb': {'solution': hs, 'fitness': hf, 'history': hh,
                    'nutrition': calculate_nutrition(hs), 'time': hh['time'],
                    'n_selected': hh.get('n_selected', 0)},
        }

    res = st.session_state.single_result
    if res is None:
        return

    # ── 3-way comparison table ───────────────────────────────────────────────
    display_three_way_comparison_st(res['ga'], res['pso'], res['hyb'])

    # Info item yang dipilih GA pada Hybrid
    n_sel = res['hyb'].get('n_selected', 0)
    st.caption(f"ℹ️ Hybrid Phase 1 (GA) memilih **{n_sel} item makanan** — "
               f"Phase 2 (PSO) mengoptimasi porsi hanya pada {n_sel} dimensi tersebut.")

    st.divider()

    # ── Convergence charts ───────────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Kurva Konvergensi</div>', unsafe_allow_html=True)
    display_convergence_all_st(res['ga']['history'], res['pso']['history'], res['hyb']['history'])

    st.divider()

    # ── Detail menu per algoritma ────────────────────────────────────────────
    st.markdown('<div class="section-title">🍽️ Detail Menu</div>', unsafe_allow_html=True)
    t_ga, t_pso, t_hyb = st.tabs(["🧬 Genetic Algorithm", "🐝 Particle Swarm Opt.",
                                    "⚡ Hybrid GA-PSO"])
    with t_ga:  display_menu_st(res['ga']['solution'],  "GA")
    with t_pso: display_menu_st(res['pso']['solution'], "PSO")
    with t_hyb: display_menu_st(res['hyb']['solution'], "Hybrid GA-PSO")


# ==============================================================================
# TAB 2 — Menu 7 Hari
# ==============================================================================

def tab_weekly(ga_p, pso_p, hybrid_p, seed):
    DAYS = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

    st.markdown("""
    <div class="info-box">
    Generate menu seimbang untuk <strong>7 hari</strong> menggunakan GA, PSO, atau Hybrid.
    Setiap hari dioptimasi secara independen dengan seed berbeda untuk variasi menu.
    Estimasi waktu: <strong>GA/PSO</strong> ~1–3 menit · <strong>Hybrid</strong> ~2–4 menit · <strong>Semua</strong> ~4–8 menit.
    </div>
    """, unsafe_allow_html=True)

    algo = st.radio("Algoritma untuk 7 Hari",
                    ["GA", "PSO", "Hybrid GA-PSO", "GA + PSO", "Semua (GA + PSO + Hybrid)"],
                    horizontal=True, index=3)

    run_ga     = algo in ("GA", "GA + PSO", "Semua (GA + PSO + Hybrid)")
    run_pso    = algo in ("PSO", "GA + PSO", "Semua (GA + PSO + Hybrid)")
    run_hybrid = algo in ("Hybrid GA-PSO", "Semua (GA + PSO + Hybrid)")

    if st.button("📅  Generate Menu 7 Hari", use_container_width=True):
        results  = {'GA': [], 'PSO': [], 'Hybrid': []}
        progress = st.progress(0)
        status   = st.empty()

        for d_idx, day in enumerate(DAYS):
            np.random.seed((seed + d_idx) * 7)
            status.markdown(f"⏳ Hari {d_idx+1}/7 — **{day}**…")

            if run_ga:
                ga_ = GeneticAlgorithm(**ga_p)
                gs_, gf_, gh_ = ga_.evolve()
                results['GA'].append({'day': day, 'solution': gs_, 'fitness': gf_,
                                      'nutrition': calculate_nutrition(gs_)})
            if run_pso:
                pso_ = ParticleSwarmOptimization(**pso_p)
                ps_, pf_, ph_ = pso_.optimize()
                results['PSO'].append({'day': day, 'solution': ps_, 'fitness': pf_,
                                       'nutrition': calculate_nutrition(ps_)})
            if run_hybrid:
                hyb_ = HybridGAPSO(**hybrid_p)
                hs_, hf_, hh_ = hyb_.run()
                results['Hybrid'].append({'day': day, 'solution': hs_, 'fitness': hf_,
                                          'nutrition': calculate_nutrition(hs_)})
            progress.progress((d_idx + 1) / 7)

        progress.empty(); status.empty()
        st.session_state.weekly_result = {'results': results, 'algo': algo}

    res = st.session_state.weekly_result
    if res is None:
        return

    results = res['results']
    st.divider()

    def make_summary_df(day_list):
        return pd.DataFrame([{
            'Hari': r['day'],
            'Fitness': f"{r['fitness']:.5f}",
            'Kalori (kkal)': f"{r['nutrition']['kalori']:.0f}",
            'Protein (g)':   f"{r['nutrition']['protein']:.1f}",
            'Karbo (g)':     f"{r['nutrition']['karbo']:.1f}",
            'Biaya (Rp)':    f"{r['nutrition']['cost']:,.0f}",
        } for r in day_list])

    lbl_map = [('GA', '🧬 Ringkasan 7 Hari — GA'),
               ('PSO', '🐝 Ringkasan 7 Hari — PSO'),
               ('Hybrid', '⚡ Ringkasan 7 Hari — Hybrid GA-PSO')]
    for key, title in lbl_map:
        if results[key]:
            st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
            st.dataframe(make_summary_df(results[key]), use_container_width=True, hide_index=True)
            total = sum(r['nutrition']['cost'] for r in results[key])
            st.caption(f"Total biaya 7 hari: **Rp {total:,.0f}** · Rata-rata: **Rp {total/7:,.0f}/hari**")

    st.divider()
    st.markdown('<div class="section-title">🍽️ Detail Menu per Hari</div>', unsafe_allow_html=True)

    exp_map = [('GA', '📋 Detail Harian — GA'),
               ('PSO', '📋 Detail Harian — PSO'),
               ('Hybrid', '📋 Detail Harian — Hybrid GA-PSO')]
    for key, exp_label in exp_map:
        if results[key]:
            with st.expander(exp_label):
                day_tabs = st.tabs([r['day'] for r in results[key]])
                for i, dtab in enumerate(day_tabs):
                    with dtab:
                        display_menu_st(results[key][i]['solution'])


# ==============================================================================
# TAB 3 — Analisis Statistik (3-way: GA vs PSO vs Hybrid)
# ==============================================================================

def tab_stats(ga_p, pso_p, hybrid_p, seed):
    st.markdown("""
    <div class="info-box">
    Jalankan GA, PSO, dan <strong>Hybrid GA-PSO</strong> sebanyak <em>n</em> kali untuk analisis statistik:
    <strong>Paired T-Test</strong> (pairwise 3-way), <strong>95% Confidence Interval</strong>, dan distribusi fitness.
    </div>
    <div class="warn-box">
    ⏱️ <strong>Perkiraan waktu per run:</strong> ±2–6 detik (GA) + ±2–6 detik (PSO) + ±2–5 detik (Hybrid).
    Mulai dengan 5–10 run untuk uji coba.
    </div>
    """, unsafe_allow_html=True)

    n_runs = st.slider("Jumlah Run", min_value=5, max_value=30, value=10, step=5)

    if st.button(f"📊  Jalankan Analisis Statistik 3-way ({n_runs} run × 3 algoritma)",
                 use_container_width=True):
        ga_fit,  ga_cost,  ga_time  = [], [], []
        pso_fit, pso_cost, pso_time = [], [], []
        hyb_fit, hyb_cost, hyb_time = [], [], []

        progress = st.progress(0)
        status   = st.empty()

        for run in range(n_runs):
            np.random.seed(seed + run * 123)
            status.markdown(f"⏳ Run **{run+1}/{n_runs}** — GA → PSO → Hybrid…")

            _ga  = GeneticAlgorithm(**ga_p)
            gs_, gf_, gh_ = _ga.evolve()
            gn_ = calculate_nutrition(gs_)
            ga_fit.append(gf_); ga_cost.append(gn_['cost']); ga_time.append(gh_['time'])

            _pso = ParticleSwarmOptimization(**pso_p)
            ps_, pf_, ph_ = _pso.optimize()
            pn_ = calculate_nutrition(ps_)
            pso_fit.append(pf_); pso_cost.append(pn_['cost']); pso_time.append(ph_['time'])

            _hyb = HybridGAPSO(**hybrid_p)
            hs_, hf_, hh_ = _hyb.run()
            hn_ = calculate_nutrition(hs_)
            hyb_fit.append(hf_); hyb_cost.append(hn_['cost']); hyb_time.append(hh_['time'])

            progress.progress((run + 1) / n_runs)

        progress.empty(); status.empty()

        ga_fa  = np.array(ga_fit);  ga_ca  = np.array(ga_cost);  ga_ta  = np.array(ga_time)
        pso_fa = np.array(pso_fit); pso_ca = np.array(pso_cost); pso_ta = np.array(pso_time)
        hyb_fa = np.array(hyb_fit); hyb_ca = np.array(hyb_cost); hyb_ta = np.array(hyb_time)

        # Pairwise Paired T-Tests (3 pasang × 3 metrik)
        def _ttest(a, b): return stats.ttest_rel(a, b)

        df_    = n_runs - 1
        t_crit = stats.t.ppf(0.975, df_)
        def ci(arr): return t_crit * (np.std(arr, ddof=1) / np.sqrt(n_runs))

        st.session_state.stats_result = {
            'n_runs': n_runs,
            'ga':  {'fit': ga_fa,  'cost': ga_ca,  'time': ga_ta},
            'pso': {'fit': pso_fa, 'cost': pso_ca, 'time': pso_ta},
            'hyb': {'fit': hyb_fa, 'cost': hyb_ca, 'time': hyb_ta},
            'ttest_pairs': {
                'GA vs PSO':    {k: _ttest(ga_fa  if k=='fit' else ga_ca  if k=='cost' else ga_ta,
                                           pso_fa if k=='fit' else pso_ca if k=='cost' else pso_ta)
                                 for k in ('fit','cost','time')},
                'GA vs Hybrid': {k: _ttest(ga_fa  if k=='fit' else ga_ca  if k=='cost' else ga_ta,
                                           hyb_fa if k=='fit' else hyb_ca if k=='cost' else hyb_ta)
                                 for k in ('fit','cost','time')},
                'PSO vs Hybrid':{k: _ttest(pso_fa if k=='fit' else pso_ca if k=='cost' else pso_ta,
                                           hyb_fa if k=='fit' else hyb_ca if k=='cost' else hyb_ta)
                                 for k in ('fit','cost','time')},
            },
            'ci': {
                'ga':  {'fit': ci(ga_fa),  'cost': ci(ga_ca),  'time': ci(ga_ta)},
                'pso': {'fit': ci(pso_fa), 'cost': ci(pso_ca), 'time': ci(pso_ta)},
                'hyb': {'fit': ci(hyb_fa), 'cost': ci(hyb_ca), 'time': ci(hyb_ta)},
            },
        }

    r = st.session_state.stats_result
    if r is None:
        return

    n   = r['n_runs']
    ga  = r['ga']; pso = r['pso']; hyb = r['hyb']
    ci  = r['ci']; pairs = r['ttest_pairs']
    st.divider()

    # ── Descriptive stats ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Statistik Deskriptif</div>', unsafe_allow_html=True)
    desc_rows = []
    for algo_lbl, d in [("GA", ga), ("PSO", pso), ("Hybrid", hyb)]:
        for metric, key in [("Fitness","fit"),("Biaya (Rp)","cost"),("Waktu (s)","time")]:
            arr = d[key]
            desc_rows.append({
                'Algoritma': algo_lbl, 'Metrik': metric,
                'Mean': f"{np.mean(arr):.4f}" if key=='fit' else f"{np.mean(arr):,.2f}",
                'Std':  f"{np.std(arr):.4f}"  if key=='fit' else f"{np.std(arr):,.2f}",
                'Min':  f"{np.min(arr):.4f}"  if key=='fit' else f"{np.min(arr):,.2f}",
                'Max':  f"{np.max(arr):.4f}"  if key=='fit' else f"{np.max(arr):,.2f}",
            })
    st.dataframe(pd.DataFrame(desc_rows), use_container_width=True, hide_index=True)

    # ── Pairwise Paired T-Tests ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🔬 Paired T-Test Pairwise (α = 0.05)</div>',
                unsafe_allow_html=True)
    ttest_rows = []
    metric_map = [("Fitness","fit",True),("Biaya (Rp)","cost",False),("Waktu (s)","time",False)]
    for pair_label, pair_data in pairs.items():
        for metric_lbl, metric_key, higher_is_better in metric_map:
            ts, pv = pair_data[metric_key]
            ttest_rows.append({
                'Pasangan': pair_label,
                'Metrik': metric_lbl,
                'T-Statistic': f"{ts:.4f}",
                'P-Value': f"{pv:.6f}",
                'Signifikan': "✅ Ya (p < 0.05)" if pv < 0.05 else "❌ Tidak",
            })
    st.dataframe(pd.DataFrame(ttest_rows), use_container_width=True, hide_index=True)

    # ── 95% CI ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📏 95% Confidence Interval</div>', unsafe_allow_html=True)
    ci_rows = []
    for algo_lbl, d_arr, ci_d in [("GA",ga,ci['ga']),("PSO",pso,ci['pso']),("Hybrid",hyb,ci['hyb'])]:
        ci_rows.append({
            'Algoritma': algo_lbl,
            'Fitness CI':    f"{np.mean(d_arr['fit']):.5f} ± {ci_d['fit']:.5f}",
            'Biaya CI (Rp)': f"{np.mean(d_arr['cost']):,.0f} ± {ci_d['cost']:,.0f}",
            'Waktu CI (s)':  f"{np.mean(d_arr['time']):.2f} ± {ci_d['time']:.2f}",
        })
    st.dataframe(pd.DataFrame(ci_rows), use_container_width=True, hide_index=True)

    # ── Boxplots 3-way ───────────────────────────────────────────────────────
    st.markdown(f'<div class="section-title">📦 Distribusi ({n} Run)</div>',
                unsafe_allow_html=True)

    plt.rcParams.update({
        'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#f8f8f8',
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
    colors = ['#a0a0f0', '#a0e0a0', '#f0b070']   # GA=blue, PSO=green, Hybrid=orange

    for ax, arr_ga, arr_pso, arr_hyb, title, ylabel in [
        (axes[0], ga['fit'],  pso['fit'],  hyb['fit'],  'Fitness Distribution',   'Fitness'),
        (axes[1], ga['cost'], pso['cost'], hyb['cost'], 'Cost Distribution (Rp)', 'Rp'),
        (axes[2], ga['time'], pso['time'], hyb['time'], 'Time Distribution (s)',  'Seconds'),
    ]:
        bp = ax.boxplot([arr_ga, arr_pso, arr_hyb], labels=['GA','PSO','Hybrid'],
                        patch_artist=True)
        for box, col in zip(bp['boxes'], colors):
            box.set_facecolor(col)
        ax.plot([1,2,3], [np.mean(arr_ga), np.mean(arr_pso), np.mean(arr_hyb)],
                'ro', ms=8, label='Mean', zorder=5)
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1a1a1a')
        ax.set_ylabel(ylabel, fontsize=10, color='#1a1a1a')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
        for sp in ax.spines.values(): sp.set_edgecolor('#ccc')

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    plt.rcdefaults()

    # ── Ranking keseluruhan ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Ranking Keseluruhan</div>', unsafe_allow_html=True)
    means = {
        'GA':     {'fit': np.mean(ga['fit']),  'cost': np.mean(ga['cost']),  'time': np.mean(ga['time'])},
        'PSO':    {'fit': np.mean(pso['fit']), 'cost': np.mean(pso['cost']), 'time': np.mean(pso['time'])},
        'Hybrid': {'fit': np.mean(hyb['fit']), 'cost': np.mean(hyb['cost']), 'time': np.mean(hyb['time'])},
    }
    best_fit  = max(means, key=lambda x: means[x]['fit'])
    best_cost = min(means, key=lambda x: means[x]['cost'])
    best_time = min(means, key=lambda x: means[x]['time'])
    rank_rows = [
        {'Kategori': '🎯 Fitness Tertinggi (rata-rata)',   'Pemenang': f"🏆 {best_fit}",
         'Nilai': f"{means[best_fit]['fit']:.5f}"},
        {'Kategori': '💰 Biaya Terendah (rata-rata)',      'Pemenang': f"🏆 {best_cost}",
         'Nilai': f"Rp {means[best_cost]['cost']:,.0f}"},
        {'Kategori': '⚡ Waktu Tercepat (rata-rata)',      'Pemenang': f"🏆 {best_time}",
         'Nilai': f"{means[best_time]['time']:.2f}s"},
    ]
    st.dataframe(pd.DataFrame(rank_rows), use_container_width=True, hide_index=True)


# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.markdown("""
    <div class="sp-header">
        <div class="sp-header-icon">🥗</div>
        <div class="sp-header-text">
            <h1>Optimasi Menu Makanan</h1>
            <p>HYBRID GA-PSO · GENETIC ALGORITHM · PARTICLE SWARM OPTIMIZATION · SMARTPLATE YEAR 2</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    ga_p, pso_p, hybrid_p, seed = build_sidebar()

    tab1, tab2, tab3 = st.tabs([
        "🔬  Optimasi 1 Hari",
        "📅  Menu 7 Hari",
        "📊  Analisis Statistik",
    ])
    with tab1: tab_single_day(ga_p, pso_p, hybrid_p, seed)
    with tab2: tab_weekly(ga_p, pso_p, hybrid_p, seed)
    with tab3: tab_stats(ga_p, pso_p, hybrid_p, seed)

    st.markdown("""
    <div class="sp-footer">
        <strong>SmartPlate Year 2</strong> — Optimasi Menu via Hybrid GA-PSO, GA, & PSO<br>
        © 2026 Mochamad Faisal Akbar · Kecerdasan Komputasional · Universitas Sebelas Maret<br>
        <em>Permenkes No. 28/2019 · TKPI 2017 · Holland (1975) · Kennedy & Eberhart (1995) ·
        Kao & Zahara (2008) — Hybrid GA-PSO</em>
    </div>
    """, unsafe_allow_html=True)


main()
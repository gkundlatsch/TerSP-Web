from flask import Flask, render_template, request
from joblib import load
from scipy.stats import entropy
import pandas as pd

app = Flask(__name__)

# Load trained model
model_data = load("terminator_strength_predictor.joblib")
model = model_data["model"]

def calculate_features(a_tract, u_tract, stem, loop):
    """Calculate all 17 features from:
       - a_tract: 8 nt A-tract
       - u_tract: 12 nt U-tract
       - stem: concatenation of first_half + second_half (no loop)
       - loop: the intervening loop sequence
    """
    features = {}

    # Helper functions
    def nt_percent(seq, nt, length):
        return seq.count(nt) / length if length > 0 else 0

    def count_changes(seq):
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y)

    # 1. A-tract (8 nt)
    a_sub = a_tract[2:]  # for 6-nt window
    features.update({
        'A%_total_A_tract':    nt_percent(a_tract, 'A', 8),
        'C%_A_tract':          nt_percent(a_tract, 'C', 8),
        'G%_A_tract':          nt_percent(a_tract, 'G', 8),
        'U%_A_tract':          nt_percent(a_tract, 'U', 8),
        'A%_6_A_tract':        nt_percent(a_sub,  'A', 6),
        'C%_6_A_tract':        nt_percent(a_sub,  'C', 6),
        'A_Tract_state-change': count_changes(a_tract)
    })

    # 2. U-tract (12 nt)
    features.update({
        'U%_Total_U_tract':     nt_percent(u_tract, 'U', 12),
        'G%_U_tract':           nt_percent(u_tract, 'G', 12),
        'A%_U_tract':           nt_percent(u_tract, 'A', 12),
        'C%_U_tract':           nt_percent(u_tract, 'C', 12),
        'A%_6_U_tract':         nt_percent(u_tract[:6],  'A', 6),
        'U%_6_U_tract':         nt_percent(u_tract[:6],  'U', 6),
        'U%_10_U_tract':        nt_percent(u_tract[:10], 'U',10),
        'U_Tract_state-change': count_changes(u_tract)
    })

    # 3. Loop
    loop_len = len(loop)
    features.update({
        'Tamanho Loop': loop_len,
        '%GC_Loop':    (loop.count('G') + loop.count('C')) / loop_len
    })

    # 4. Stem (hairpin without loop)
    hp_len = len(stem)
    features.update({
        'Tamanho Hairpin sem Loop': hp_len,
        '%G_HP': stem.count('G') / hp_len,
        '%C_HP': stem.count('C') / hp_len,
        '%A_HP': stem.count('A') / hp_len,
        '%U_HP': stem.count('U') / hp_len,
        'HP_S_Loop_state_change': sum(1 for x, y in zip(stem, stem[1:]) if x != y),
        'GC_Inicial_Hairpin': next((i for i, c in enumerate(stem) if c not in ('G','C')), hp_len)
    })

    # 5. Entropy
    features['Entropia_A_tract']    = entropy([
        features['A%_total_A_tract'],
        features['C%_A_tract'],
        features['G%_A_tract'],
        features['U%_A_tract']
    ], base=2)

    features['Entropia_U_tract']    = entropy([
        features['U%_Total_U_tract'],
        features['C%_U_tract'],
        features['G%_U_tract'],
        features['A%_U_tract']
    ], base=2)

    features['Entropia_HP_S_Loop']  = entropy([
        features['%G_HP'],
        features['%C_HP'],
        features['%A_HP'],
        features['%U_HP']
    ], base=2)

    # 6. Normalize to [0,1] (using original training bounds)
    normalized = {
        'Tamanho Loop':               (features['Tamanho Loop'] - 3) / (16 - 3),
        'A%_6_A_tract':               features['A%_6_A_tract'],
        'C%_6_A_tract':               features['C%_6_A_tract'],
        'U%_10_U_tract':              features['U%_10_U_tract'],
        'U%_6_U_tract':               features['U%_6_U_tract'],
        'A%_6_U_tract':               features['A%_6_U_tract'],
        'C%_U_tract':                 features['C%_U_tract'],
        'Tamanho Hairpin sem Loop':   (features['Tamanho Hairpin sem Loop'] - 6) / (49 - 6),
        '%GC_Loop':                   features['%GC_Loop'],
        'Entropia_A_tract':           features['Entropia_A_tract']     / 2,
        'Entropia_U_tract':           features['Entropia_U_tract']     / 2,
        'Entropia_HP_S_Loop':         (features['Entropia_HP_S_Loop']- 0.998000884)  / (2.0 - 0.998000884),
        'A_Tract_state-change':       features['A_Tract_state-change'] / 7,
        'U_Tract_state-change':       features['U_Tract_state-change'] / 11,
        'HP_S_Loop_state_change':     (features['HP_S_Loop_state_change'] - 2) / (34 - 2),
        'GC_Inicial_Hairpin':         features['GC_Inicial_Hairpin']   / 12
    }
    # ── CLIP every normalized value into [0,1] ──
    for k, v in normalized.items():
        normalized[k] = min(max(v, 0.0), 1.0)
    return normalized

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            a_tract     = request.form['a_tract'].upper()
            u_tract     = request.form['u_tract'].upper()
            first_half  = request.form['first_half'].upper()
            loop        = request.form['loop'].upper()
            second_half = request.form['second_half'].upper()

            # Validate lengths
            if len(a_tract) != 8 or len(u_tract) != 12:
                return "Error: A-tract must be 8 nt and U-tract 12 nt"
            if not (3 <= len(first_half) <= 24):
                return "Error: First half must be 3–24 nt"
            if not (3 <= len(loop) <= 16):
                return "Error: Loop must be 3–16 nt"


            # Build stem and compute features
            stem = first_half + second_half
            feats = calculate_features(a_tract, u_tract, stem, loop)
            df    = pd.DataFrame([feats])
            strength = model.predict(df)[0]
            result   = "STRONG" if strength >= 40 else "WEAK"

            return render_template(
                'result.html',
                strength=f"{strength:.2f}",
                result=result,
                a_tract=a_tract,
                u_tract=u_tract,
                first_half=first_half,
                loop=loop,
                second_half=second_half
            )
        except Exception as e:
            return f"Error: {e}"

    return render_template('predictor.html')

if __name__ == '__main__':
    app.run(debug=True)

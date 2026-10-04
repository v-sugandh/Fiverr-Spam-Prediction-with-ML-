import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

st.set_page_config(page_title="Fiverr Spammer Radar", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

# ====================================================================== styling
CSS = """
<style>
.block-container {padding-top: 1.3rem; max-width: 1200px;}
.hero {padding: 28px 34px; border-radius: 20px; color: #fff; margin-bottom: 20px;
       background: linear-gradient(120deg, #141e30 0%, #243b55 55%, #3a6073 100%);
       box-shadow: 0 10px 28px rgba(0,0,0,.20);}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; letter-spacing: .01em;}
.hero p  {margin: 8px 0 0; opacity: .88; font-size: 1.02rem;}
.pill {display: inline-block; padding: 4px 14px; border-radius: 999px; font-size: .78rem;
       background: rgba(255,255,255,.16); margin: 14px 8px 0 0;}
.kpi {border-radius: 14px; padding: 16px 18px; background: rgba(128,128,128,.10);
      border: 1px solid rgba(128,128,128,.22); border-left: 6px solid var(--c, #4C9BE8);
      height: 100%;}
.kpi .l {font-size: .74rem; opacity: .72; text-transform: uppercase; letter-spacing: .06em;}
.kpi .v {font-size: 1.85rem; font-weight: 700; line-height: 1.25;}
.kpi .s {font-size: .78rem; opacity: .65;}
.result {border-radius: 18px; padding: 26px 20px; text-align: center; color: #fff;
         box-shadow: 0 8px 24px rgba(0,0,0,.20);}
.result .big {font-size: 3.4rem; font-weight: 800; line-height: 1.1;}
.result .tag {font-size: 1.15rem; font-weight: 700; letter-spacing: .08em; margin-top: 6px;}
.result .sub {font-size: .85rem; opacity: .9; margin-top: 6px;}
.riskwrap {position: relative; margin: 30px 6px 6px;}
.riskbar {height: 16px; border-radius: 999px;}
.marker {position: absolute; top: -7px; width: 6px; height: 30px; border-radius: 4px;
         background: #fff; border: 2px solid #111;}
.thrline {position: absolute; top: -12px; height: 40px; border-left: 2px dashed #888;}
.scale {display: flex; justify-content: space-between; font-size: .72rem; opacity: .65; margin: 6px 6px 0;}
.step {border-radius: 14px; padding: 14px 12px; text-align: center; font-size: .86rem;
       background: rgba(128,128,128,.10); border: 1px solid rgba(128,128,128,.22); height: 100%;}
.step b {display: block; font-size: 1.5rem;}
.note {font-size: .82rem; opacity: .7;}
div[data-testid="stTabs"] button[role="tab"] {font-size: 1.02rem; font-weight: 600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

GREEN, AMBER, RED, BLUE = "#2ecc71", "#f5a623", "#ff4b6e", "#4C9BE8"


def kpi(label, value, sub="", color=BLUE):
    return (f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def style_ax(fig, ax):
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color("#888")
    ax.tick_params(colors="#888")
    ax.xaxis.label.set_color("#888"); ax.yaxis.label.set_color("#888")
    ax.title.set_color("#888")


# ====================================================================== artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("lightgbm_spammer_model.pkl")
    meta = json.load(open("model_metadata.json"))
    info = json.load(open("feature_defaults.json"))
    return model, meta, info


try:
    model, meta, info = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files. Run the app from the folder that contains "
             f"lightgbm_spammer_model.pkl, model_metadata.json and feature_defaults.json.\n\n{e}")
    st.stop()

FEATURES = meta["features"]
RAW = info["raw_columns"]
DEF, RNG = info["defaults"], info["ranges"]
KEY = [c for c in ["X19", "X6", "X2", "X4", "X1", "X3", "X9", "X7", "X22", "X5", "X8",
                   "X25", "X17", "X16", "X21", "X11", "X15"] if c in RAW]
BEST_THR = float(meta["threshold"])


def prepare(df):
    missing = [c for c in RAW if c not in df.columns]
    if missing:
        return None, missing
    d = df[RAW].apply(pd.to_numeric, errors="coerce")
    d["X13"] = d["X13"].fillna(DEF["X13"])
    d["X10_is_1"] = (d["X10"] == 1).astype(int)
    return d[FEATURES], []


def risk_level(p, thr):
    if p >= thr:
        return "HIGH RISK", RED
    if p >= thr * 0.5:
        return "ELEVATED", AMBER
    return "LOW RISK", GREEN


# ====================================================================== sidebar
st.sidebar.markdown("## 🛡️ Spammer Radar")
st.sidebar.caption("Choose how strict the model should be.")

PRESETS = {
    "Balanced (best F1)": (BEST_THR, "Precision 0.81 · Recall 0.69"),
    "Wide net (review queue)": (0.10, "Precision 0.55 · Recall 0.81"),
    "Strict (auto-block)": (0.80, "Precision 0.93 · Recall 0.48"),
    "Custom": (None, ""),
}
mode = st.sidebar.radio("Operating mode", list(PRESETS.keys()))
if mode == "Custom":
    thr = st.sidebar.slider("Decision threshold", 0.05, 0.95, BEST_THR, 0.01)
else:
    thr = PRESETS[mode][0]
    st.sidebar.info(f"Threshold **{thr:.3f}**\n\n{PRESETS[mode][1]} (test set)")

st.sidebar.markdown("---")
st.sidebar.markdown("**Risk levels**")
st.sidebar.markdown(f"🟢 Low: below {thr*0.5:.2f}  \n🟠 Elevated: {thr*0.5:.2f} to {thr:.2f}  \n"
                    f"🔴 High: {thr:.2f} and above")
st.sidebar.caption("Lower threshold = catch more spammers but more false alarms. "
                   "Higher = fewer false alarms but more spammers missed.")

# ====================================================================== hero
st.markdown(f"""
<div class="hero">
  <h1>🛡️ Fiverr Spammer Radar</h1>
  <p>Predict which new users are likely to become spammers from their early behaviour.</p>
  <span class="pill">LightGBM (tuned)</span>
  <span class="pill">Test PR-AUC {meta['test_PR_AUC']:.3f}</span>
  <span class="pill">Test ROC-AUC {meta['test_ROC_AUC']:.3f}</span>
  <span class="pill">458,798 users · 2.7% spammers</span>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Single user", "📂 Batch scoring", "📊 About the model"])

# ====================================================================== tab 1
with tab1:
    st.markdown("#### Check one user")
    st.markdown('<div class="note">Enter the behaviour values that matter most to the model. '
                'The other features are set to typical (median) values.</div>', unsafe_allow_html=True)
    with st.form("single_user"):
        vals = {}
        cols = st.columns(4)
        for i, c in enumerate(KEY):
            lo, hi = RNG[c]
            vals[c] = cols[i % 4].number_input(c, min_value=float(lo), max_value=float(hi),
                                               value=float(DEF[c]), step=1.0)
        go = st.form_submit_button("🔍  Predict risk", type="primary")

    if go:
        row = {c: DEF[c] for c in RAW}
        row.update(vals)
        X, _ = prepare(pd.DataFrame([row]))
        p = float(model.predict_proba(X)[:, 1][0])
        label, color = risk_level(p, thr)

        left, right = st.columns([1, 1.4], gap="large")
        with left:
            verdict = "Likely spammer" if p >= thr else "Likely genuine user"
            st.markdown(f"""
            <div class="result" style="background:linear-gradient(135deg,{color},{color}cc)">
              <div class="sub">Spam probability</div>
              <div class="big">{p:.1%}</div>
              <div class="tag">{label}</div>
              <div class="sub">{verdict} at threshold {thr:.2f}</div>
            </div>""", unsafe_allow_html=True)

            pos = min(max(p, 0), 1) * 100
            tpos = thr * 100
            st.markdown(f"""
            <div class="riskwrap">
              <div class="riskbar" style="background:linear-gradient(90deg,{GREEN} 0%,{AMBER} {tpos:.0f}%,{RED} {min(tpos+20,100):.0f}%)"></div>
              <div class="thrline" style="left:{tpos}%"></div>
              <div class="marker" style="left:calc({pos}% - 3px)"></div>
            </div>
            <div class="scale"><span>0%</span><span>threshold {thr:.2f} (dashed)</span><span>100%</span></div>
            """, unsafe_allow_html=True)

        with right:
            try:
                contrib = pd.Series(model.predict_proba(X, pred_contrib=True)[0][:-1], index=FEATURES)
                top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
                fig, ax = plt.subplots(figsize=(6, 3.6))
                ax.barh(top.index, top.values, color=[RED if v > 0 else GREEN for v in top.values])
                ax.axvline(0, color="#888", lw=1)
                ax.set_xlabel("Effect on spam score (log-odds)")
                ax.set_title("What drives this prediction")
                style_ax(fig, ax)
                st.pyplot(fig)
                st.markdown('<div class="note">🔴 pushes the score up (more spam-like) · '
                            '🟢 pushes it down. Features not entered are held at typical values.</div>',
                            unsafe_allow_html=True)
            except Exception:
                st.info("Per-prediction explanation is not available for this model version.")

# ====================================================================== tab 2
with tab2:
    st.markdown("#### Score many users at once")
    st.markdown(f'<div class="note">Upload a CSV or Excel file with the {len(RAW)} feature columns '
                '(X1 to X51 without the 9 dropped columns). <b>user_id</b> and <b>label</b> are optional. '
                'Try the included <b>sample_users.csv</b>.</div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload file", type=["csv", "xlsx"], label_visibility="collapsed")

    if up is None:
        st.info("⬆️ Upload a file to see predictions, risk breakdown and a downloadable results file.")
    else:
        raw = pd.read_excel(up) if up.name.endswith("xlsx") else pd.read_csv(up)
        X, missing = prepare(raw)
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= thr).astype(int)
            level = np.select([prob >= thr, prob >= thr * 0.5], ["High", "Elevated"], "Low")
            out = raw.copy()
            out["spam_probability"] = prob
            out["risk_level"] = level
            out["predicted_spammer"] = pred

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(kpi("Users scored", f"{len(out):,}", "rows in file"), unsafe_allow_html=True)
            k2.markdown(kpi("Flagged", f"{pred.sum():,}", f"at threshold {thr:.2f}", RED), unsafe_allow_html=True)
            k3.markdown(kpi("Share flagged", f"{pred.mean():.1%}", "of all users", AMBER), unsafe_allow_html=True)
            k4.markdown(kpi("Average risk", f"{prob.mean():.1%}", "mean probability", GREEN), unsafe_allow_html=True)

            has_labels = "label" in raw.columns and raw["label"].isin([0, 1]).all()
            if has_labels:
                y = raw["label"].astype(int)
                st.markdown("##### Performance on this file")
                m1, m2, m3, m4 = st.columns(4)
                m1.markdown(kpi("PR-AUC", f"{average_precision_score(y, prob):.3f}", "ranking quality"), unsafe_allow_html=True)
                m2.markdown(kpi("Precision", f"{precision_score(y, pred, zero_division=0):.3f}", "flagged that are spam", GREEN), unsafe_allow_html=True)
                m3.markdown(kpi("Recall", f"{recall_score(y, pred, zero_division=0):.3f}", "spammers caught", AMBER), unsafe_allow_html=True)
                m4.markdown(kpi("F1", f"{f1_score(y, pred, zero_division=0):.3f}", "balance", RED), unsafe_allow_html=True)

            st.markdown("")
            c1, c2 = st.columns(2, gap="large")
            with c1:
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.hist(prob, bins=40, color=BLUE, alpha=.9)
                ax.axvline(thr, color=RED, ls="--", lw=2, label=f"threshold {thr:.2f}")
                ax.set_yscale("log"); ax.set_xlabel("Spam probability"); ax.set_ylabel("Users (log)")
                ax.set_title("Score distribution"); ax.legend(frameon=False, labelcolor="#888")
                style_ax(fig, ax); st.pyplot(fig)
            with c2:
                if has_labels:
                    cm = confusion_matrix(y, pred, labels=[0, 1])
                    fig, ax = plt.subplots(figsize=(6, 3.4))
                    ax.imshow(cm, cmap="Blues")
                    for (i, j), v in np.ndenumerate(cm):
                        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=14,
                                color="white" if v > cm.max() / 2 else "#222")
                    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred genuine", "Pred spam"])
                    ax.set_yticks([0, 1]); ax.set_yticklabels(["True genuine", "True spam"])
                    ax.set_title("Confusion matrix"); style_ax(fig, ax); st.pyplot(fig)
                else:
                    counts = pd.Series(level).value_counts().reindex(["Low", "Elevated", "High"]).fillna(0)
                    fig, ax = plt.subplots(figsize=(6, 3.4))
                    ax.bar(counts.index, counts.values, color=[GREEN, AMBER, RED])
                    ax.set_title("Users by risk level"); ax.set_ylabel("Users")
                    style_ax(fig, ax); st.pyplot(fig)

            st.markdown("##### Highest-risk users")
            show = (["user_id"] if "user_id" in out.columns else []) + ["spam_probability", "risk_level", "predicted_spammer"]
            top = out.sort_values("spam_probability", ascending=False)[show].head(25)
            st.dataframe(top, hide_index=True, column_config={
                "spam_probability": st.column_config.ProgressColumn(
                    "Spam probability", min_value=0.0, max_value=1.0, format="%.3f")})
            st.download_button("⬇️  Download scored file (CSV)", out.to_csv(index=False).encode("utf-8"),
                               "scored_users.csv", "text/csv", type="primary")

# ====================================================================== tab 3
with tab3:
    st.markdown("#### Model performance (held-out test set, 68,820 users)")
    a, b, c3, d = st.columns(4)
    a.markdown(kpi("PR-AUC", f"{meta['test_PR_AUC']:.3f}", "random model ≈ 0.027", BLUE), unsafe_allow_html=True)
    b.markdown(kpi("ROC-AUC", f"{meta['test_ROC_AUC']:.3f}", "ranking of all users", BLUE), unsafe_allow_html=True)
    c3.markdown(kpi("Precision", "0.813", "flagged users that are spam", GREEN), unsafe_allow_html=True)
    d.markdown(kpi("Recall", "0.691", "of all spammers caught", AMBER), unsafe_allow_html=True)

    st.markdown("")
    l, r = st.columns([1.1, 1], gap="large")
    with l:
        imp = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=FEATURES)
        imp = (imp / imp.sum()).sort_values(ascending=False).head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(imp.index, imp.values * 100, color=BLUE)
        ax.set_xlabel("Share of total gain (%)"); ax.set_title("Top 15 features")
        style_ax(fig, ax); st.pyplot(fig)
    with r:
        st.markdown("**Threshold trade-off (test set)**")
        st.dataframe(pd.DataFrame({
            "Threshold": [0.10, 0.30, 0.376, 0.50, 0.80],
            "Flagged": [2712, 1723, 1569, 1363, 965],
            "Precision": [0.552, 0.773, 0.813, 0.862, 0.928],
            "Recall": [0.811, 0.721, 0.691, 0.636, 0.484],
            "F1": [0.657, 0.746, 0.747, 0.732, 0.636],
        }), hide_index=True)
        st.markdown('<div class="note">Importance shows what the model relies on, not why users spam. '
                    'Columns are anonymised. X19 alone carries about a third of the signal.</div>',
                    unsafe_allow_html=True)

    st.markdown("#### How the model was built")
    s = st.columns(5)
    steps = [("🧹", "Clean", "duplicates, constants, missing"), ("📈", "Explore", "EDA and key signals"),
             ("🔧", "Transform", "log1p, caps, scaling, encoding"), ("🤖", "Compare", "11 algorithms"),
             ("🎯", "Tune and test", "LightGBM, one-time test")]
    for col, (ic, t, d_) in zip(s, steps):
        col.markdown(f'<div class="step"><b>{ic}</b><strong>{t}</strong><br>{d_}</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Fiverr Potential Spammer Predictor · Kaggle: Predict Potential Spammers on Fiverr · "
           "Built with LightGBM and Streamlit")

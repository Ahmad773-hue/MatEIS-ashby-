"""MatEIS - Materials Selection & Ashby Plotter (Streamlit). Run: streamlit run app.py"""
import io
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import engine as en

COL = {"Metal": "#1f77b4", "Ceramic": "#d62728", "Polymer": "#2ca02c", "Composite": "#9467bd"}
lab = lambda k: f"{en.PROPS[k][0]} ({en.PROPS[k][1]})"

st.set_page_config("MatEIS – Ashby Plotter", "🔬", layout="wide")
st.title("🔬 MatEIS · Materials Selection & Ashby Plotter")
st.caption("CSC-214 · B.Sc. Metallurgy & Materials Engineering · University of the Punjab, Lahore")


@st.cache_data
def get_db():
    return en.load()


df = get_db()
with st.sidebar:
    st.header("Constraints")
    fams = st.multiselect("Material families", en.FAMILIES, default=en.FAMILIES)
    limits = {}
    for k in en.PROPS:
        with st.expander(lab(k), expanded=k == "sigma_y"):
            lo = st.number_input("Minimum", 0.0, float(df[k].max()) * 10, 200.0 if k == "sigma_y" else 0.0, key=k + "lo")
            hi = st.number_input("Maximum (0 = no limit)", 0.0, float(df[k].max()) * 10, 0.0, key=k + "hi")
            limits[k] = (lo or None, hi or None)
try:
    ok = en.filter_materials(df, limits, fams)
except ValueError as e:
    st.sidebar.error(str(e)); ok = df.iloc[0:0]

c1, c2, c3 = st.columns(3)
c1.metric("Materials in database", len(df)); c2.metric("Pass all constraints", len(ok))
c3.metric("Rejected", len(df) - len(ok))

t_plot, t_rank, t_data = st.tabs(["📈 Ashby plot", "🏆 Ranking", "🗄️ Database"])

with t_plot:
    a, b, c = st.columns(3)
    x = a.selectbox("X axis", list(en.PROPS), 0, format_func=lab)
    y = b.selectbox("Y axis", list(en.PROPS), 2, format_func=lab)
    idx = c.selectbox("Material index guide line", ["None"] + list(en.INDICES))
    o1, o2, o3 = st.columns(3)
    log = o1.checkbox("Log–log axes", True); front = o2.checkbox("Show Pareto front", True)
    labels = o3.checkbox("Label materials", True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for fam, g in df.groupby("family"):
        g_ok = g[g.index.isin(ok.index)]
        ax.scatter(g.drop(g_ok.index)[x], g.drop(g_ok.index)[y], s=35, c=COL[fam], alpha=.2)
        ax.scatter(g_ok[x], g_ok[y], s=95, c=COL[fam], edgecolors="k", label=fam, zorder=3)
    if labels:
        for _, r in ok.iterrows():
            ax.annotate(r["material"], (r[x], r[y]), xytext=(5, 4), textcoords="offset points", fontsize=7.5)
    if front and x != y and len(ok) > 1:
        f = en.pareto_front(ok, x, y)
        ax.plot(f[x], f[y], "k--", lw=1, label="Pareto front", zorder=2)
    if idx != "None" and len(ok):
        prop, ex, wc = en.INDICES[idx]
        if log and not wc and x == "density" and y == prop:
            best = en.rank(ok, idx).iloc[0]
            xs = np.geomspace(df[x].min() * .7, df[x].max() * 1.4, 50)
            ax.plot(xs, best[y] * (xs / best[x]) ** (1 / ex), color="orange", lw=2, label="Index guide line")
        else:
            st.info("Guide lines are drawn on log–log plots of Density (x) vs the index's strength/stiffness property (y), for non-cost indices.")
    if log: ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(lab(x)); ax.set_ylabel(lab(y)); ax.grid(True, which="both", alpha=.25)
    ax.legend(fontsize=8); ax.set_title("Ashby chart – faded points fail the constraints")
    st.pyplot(fig)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    st.download_button("⬇️ Download plot (PNG)", buf.getvalue(), "ashby_plot.png", "image/png")

with t_rank:
    key = st.selectbox("Performance index", list(en.INDICES), key="rk")
    if ok.empty:
        st.warning("No material satisfies the current constraints. Relax a limit in the sidebar.")
    else:
        r = en.rank(ok, key)
        st.success(f"Best candidate: **{r.loc[0, 'material']}**")
        st.dataframe(r.rename(columns={k: lab(k) for k in en.PROPS}), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export ranking (CSV)", r.to_csv(index=False), "ranking.csv", "text/csv")
        st.caption("Indices use ρ in kg/m³, E in GPa, σy in MPa, cost in USD/kg; use them for relative ranking only.")

with t_data:
    st.dataframe(df.rename(columns={k: lab(k) for k in en.PROPS}), use_container_width=True, hide_index=True)
    with st.form("add"):
        st.subheader("Add a material")
        n, f = st.columns(2)
        name = n.text_input("Name"); fam = f.selectbox("Family", en.FAMILIES)
        cols = st.columns(4)
        vals = {k: cols[i].number_input(lab(k), 0.0, format="%.2f") for i, k in enumerate(en.PROPS)}
        if st.form_submit_button("Save to database"):
            try:
                en.add_material({"material": name, "family": fam, **vals})
                get_db.clear(); st.success("Saved."); st.rerun()
            except ValueError as e:
                st.error(str(e))
    st.caption("Values are typical handbook figures (annealed/typical temper; ceramic strength = flexural). Verify against ASM Handbook / Granta before design use.")

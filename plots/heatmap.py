import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import os
from settings import ROOT_DIR

OUTPUT_DIR = os.path.join(ROOT_DIR, "figures")

# Embed fonts in PDF
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

def load_with_type(path, mapping_type):
    df = pd.read_csv(path)
    df["mapping_type"] = mapping_type
    return df

paths = {
    "exploit":   os.path.join(ROOT_DIR, "data/post_processed/tables_data/exp_classwise.csv"),
    "primary":   os.path.join(ROOT_DIR, "data/post_processed/tables_data/prim_classwise.csv"),
    "secondary": os.path.join(ROOT_DIR, "data/post_processed/tables_data/sec_classwise.csv")
}

dfs = [load_with_type(p, mtype) for mtype, p in paths.items() if Path(p).exists()]
if not dfs:
    raise FileNotFoundError("No input CSVs found.")
raw = pd.concat(dfs, ignore_index=True)

# IDs only on the y-axis
ids = raw["attack-id"].apply(lambda x: str(x).strip() if pd.notna(x) else None)
raw = raw.assign(technique_label=ids)
raw = raw[raw["technique_label"].notna()]
raw = raw[raw["technique_label"].str.lower() != "nan"]

# ---------- Identify metric columns ----------
lower = {c.lower(): c for c in raw.columns}
def pick(*aliases):
    for a in aliases:
        if a in lower: return lower[a]
    return None

col_n_train = pick("n-true-train","n_true_train","ntrain","n_true_tr")
col_n_test  = pick("n-true-test","n_true_test","ntest","n_true_te")
col_r_train = pick("recall-at-ten-train","recall_at_ten_train","r@10-train","recall10_train")
col_r_test  = pick("recall-at-ten-test","recall_at_ten_test","r@10-test","recall10_test")

for c in (col_n_train,col_n_test,col_r_train,col_r_test):
    if c is not None:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")

# ---------- Build aligned matrices ----------
pairs = [("exploit","train"),("exploit","test"),
         ("primary","train"),("primary","test"),
         ("secondary","train"),("secondary","test")]

def build_matrix(col_train, col_test):
    parts = []
    if col_train is not None:
        parts.append(raw[["technique_label","mapping_type",col_train]]
                     .rename(columns={col_train:"value"}).assign(split="train"))
    if col_test is not None:
        parts.append(raw[["technique_label","mapping_type",col_test]]
                     .rename(columns={col_test:"value"}).assign(split="test"))
    long_df = pd.concat(parts, ignore_index=True)
    return (long_df.pivot_table(index="technique_label",
                                columns=["mapping_type","split"],
                                values="value",
                                aggfunc="first")
                   .reindex(columns=pd.MultiIndex.from_tuples(pairs, names=["type","split"])))

M_freq = build_matrix(col_n_train, col_n_test)
M_reca = build_matrix(col_r_train, col_r_test).clip(lower=0, upper=1)

# ---------- Cluster rows by presence (type first), then total freq ----------
type_presence = pd.DataFrame(index=M_freq.index, columns=["exploit","primary","secondary"], dtype=int)
for t in ["exploit","primary","secondary"]:
    sub = M_freq[[(t,"train"),(t,"test")]].fillna(0)
    type_presence[t] = (sub.sum(axis=1) > 0).astype(int)

split_presence = (M_freq.fillna(0) > 0).astype(int)
total_freq = M_freq.fillna(0).sum(axis=1)

# order presence patterns preferring exploit -> primary -> secondary
patterns = [(e,p,s) for e in (1,0) for p in (1,0) for s in (1,0)]
pattern_rank = {pat:i for i, pat in enumerate(patterns)}

def row_key(idx):
    e,p,s = type_presence.loc[idx, ["exploit","primary","secondary"]].astype(int).tolist()
    return (
        pattern_rank[(e,p,s)],
        -total_freq.loc[idx],
        tuple(-M_freq.fillna(0).loc[idx, pairs]),
        idx
    )

row_order = sorted(M_freq.index.tolist(), key=row_key)
M_freq = M_freq.loc[row_order]
M_reca = M_reca.loc[row_order]

# ---------- Compose interleaved RGBA image (Recall first, then #) ----------
n_rows = len(M_freq)
n_cols = len(pairs) * 2
img = np.ones((n_rows, n_cols, 4), dtype=float)

cmap_freq = plt.get_cmap("Blues")
cmap_reca = plt.get_cmap("YlGn")
vmax_freq = float(np.nanmax(M_freq.to_numpy())) if np.isfinite(M_freq.to_numpy()).any() else 1.0
norm_freq = Normalize(vmin=0, vmax=vmax_freq)
norm_reca = Normalize(vmin=0, vmax=1.0)

# xticklabels: ONLY the leaf labels per column
xticklabels = []
cj = 0
for t, s in pairs:
    # recall first
    rv = M_reca[(t, s)].to_numpy()
    for i, v in enumerate(rv):
        if np.isnan(v):
            img[i, cj, 0:3] = 1.0   # keep RGB white
            img[i, cj, 3]   = 0.0   # alpha=0 → transparent
        else:
            img[i, cj, :] = cmap_reca(norm_reca(v))
    xticklabels.append("R@10"); cj += 1

    # frequency second
    fv = M_freq[(t, s)].to_numpy()
    for i, v in enumerate(fv):
        if np.isnan(v):
            img[i, cj, 0:3] = 1.0
            img[i, cj, 3]   = 0.0
        else:
            img[i, cj, :] = cmap_freq(norm_freq(v))
    xticklabels.append("freq"); cj += 1
# ---------- Figure geometry (bricks; height < width) ----------
base_cell_w = 0.40
left, right, top, bottom = 1.6, 3.0, 0.4, 0.6
data_w = n_cols * base_cell_w
data_h = n_rows * (base_cell_w * 0.60)

fig_w = left + data_w + right
fig_h = top + data_h + bottom
fig = plt.figure(figsize=(fig_w, fig_h))

ax = fig.add_axes([left/fig_w, bottom/fig_h, data_w/fig_w, data_h/fig_h])
ax.grid(True, linewidth=0.5, color="0.9", zorder=0)
for sp in ax.spines.values():
    sp.set_visible(False)

ax.imshow(
    img,
    aspect="auto",
    interpolation="nearest",                         # crisp cell edges
    extent=(-0.5, n_cols - 0.5, n_rows - 0.5, -0.5), # 1.0 unit per column/row
    zorder=2
)

# Immediately constrain the axes to those exact bounds
ax.set_xlim(-0.5, n_cols - 0.5)
ax.set_ylim(n_rows - 0.5, -0.5)

ax.set_yticks(range(n_rows))
ax.set_yticklabels(M_freq.index, fontsize=9)
ax.set_xticks(range(n_cols))
ax.set_xticklabels(xticklabels, fontsize=9)

# Hide tick marks (keep labels)
ax.tick_params(axis='both', which='both', length=0)

# --- measure widest y-label and refine geometry for brick height ---
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
max_lbl_in = max((t.get_window_extent(renderer=renderer).width / fig.dpi) for t in ax.get_yticklabels())

left = max(1.2, max_lbl_in + 0.25)
height_per_width = float(np.clip(0.85 - 0.20 * max_lbl_in, 0.45, 0.85))
cell_h = base_cell_w * height_per_width

data_w = n_cols * base_cell_w
data_h = n_rows * cell_h
fig_w = left + data_w + right
fig_h = top + data_h + bottom
fig.set_size_inches(fig_w, fig_h, forward=True)
ax.set_position([left/fig_w, bottom/fig_h, data_w/fig_w, data_h/fig_h])

# --- Booktabs-style bottom group labels and cmidrules ---
from matplotlib import transforms
extra_bottom_inches = 0
fig_h2 = fig.get_figheight() + extra_bottom_inches
fig.set_size_inches(fig.get_figwidth(), fig_h2, forward=True)
ax.set_position([left/fig_w, (bottom+extra_bottom_inches)/fig_h2, data_w/fig_w, data_h/fig_h2])

trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

def cmidrule(x0, x1, y, shrink=0.15, lw=1.0):
    ax.plot([x0 + shrink, x1 - shrink], [y, y],
            transform=trans, clip_on=False, linewidth=lw, color="black")

y_split_rule  = -0.027
y_split_label = -0.032
y_type_rule   = -0.048
y_type_label  = -0.053

# helper: span in cell-centered coords -> edge coords
def span(c0, c1): return (c0 - 0.5, c1 + 0.5)

# fixed order of types; each has two pairs (train/test) → 4 columns
display_names = {
    "exploit":   "exploitation technique",
    "primary":   "primary impact",
    "secondary": "secondary impact",
}
types = ["exploit","primary","secondary"]
for block_idx, t in enumerate(types):
    p0 = 2 * block_idx      # pair index for (t, train)
    p1 = p0 + 1             # pair index for (t, test)

    # columns for train/test (each pair spans 2 columns: [R@10, #])
    train_c0, train_c1 = 2 * p0, 2 * p0 + 1
    test_c0,  test_c1  = 2 * p1, 2 * p1 + 1

    # draw split-level rules + labels
    x0, x1 = span(train_c0, train_c1)
    cmidrule(x0, x1, y_split_rule)
    ax.text((x0 + x1) / 2, y_split_label, "train", transform=trans,
            ha="center", va="top", fontsize=9)

    x0, x1 = span(test_c0, test_c1)
    cmidrule(x0, x1, y_split_rule)
    ax.text((x0 + x1) / 2, y_split_label, "test", transform=trans,
            ha="center", va="top", fontsize=9)

    # type-level rule across both train+test (four columns)
    x0, x1 = span(2 * p0, 2 * p1 + 1)
    cmidrule(x0, x1, y_type_rule, lw=1)
    ax.text((x0 + x1) / 2, y_type_label, display_names.get(t, t), transform=trans,
            ha="center", va="top", fontsize=9)

# --- Annotate counts inside frequency bricks ---
freq_white_threshold = 70
for pair_idx, (t, s) in enumerate(pairs):
    col_j = 2 * pair_idx + 1  # frequency columns are second in each pair
    for i, v in enumerate(M_freq[(t, s)].to_numpy()):
        if np.isnan(v): continue
        val = int(round(v))
        ax.text(col_j, i, f"{val}",
                ha="center", va="center",
                fontsize=9, color=("white" if val > freq_white_threshold else "black"))

# --- Annotate R@10 values inside green bricks ---
reca_white_threshold = 0.60   # <= 0.60 → black text; > 0.60 → white text
reca_fmt = "{:.2f}"           # change to "{:.1f}" or "{:.0%}" if you prefer

for pair_idx, (t, s) in enumerate(pairs):
    col_j = 2 * pair_idx      # recall columns are FIRST in each (R@10, #) pair
    for i, v in enumerate(M_reca[(t, s)].to_numpy()):
        if np.isnan(v):
            continue
        ax.text(
            col_j, i, reca_fmt.format(float(v)),
            ha="center", va="center",
            fontsize=9,
            color=("white" if v > reca_white_threshold else "black")
        )
        
# --- External colorbars: Frequency (left/outside) and Recall (right/inside) ---
from matplotlib.cm import ScalarMappable
sm_freq = ScalarMappable(cmap=cmap_freq, norm=norm_freq)
sm_reca = ScalarMappable(cmap=cmap_reca, norm=norm_reca)
sm_freq.set_array([]); sm_reca.set_array([])

bar_w = 0.02; pad_w = 0.017; bottom_offset=.5

# LEFT (outer) = Frequency
cax_freq = fig.add_axes([(left/fig_w) - 4*pad_w - bar_w, (bottom+extra_bottom_inches+bottom_offset)/fig_h2, bar_w, data_h/fig_h2*0.9 ])
# RIGHT (inner) = Recall
cax_reca = fig.add_axes([ (left+data_w+0.30)/fig_w - pad_w, (bottom+extra_bottom_inches+bottom_offset)/fig_h2, bar_w, data_h/fig_h2*0.9 ])

cb_freq = fig.colorbar(sm_freq, cax=cax_freq)
cb_reca = fig.colorbar(sm_reca, cax=cax_reca)

cb_freq.locator = MaxNLocator(nbins=6, integer=True); cb_freq.update_ticks()
cb_freq.ax.tick_params(labelsize=8)
cb_freq.ax.yaxis.set_label_position('left'); cb_freq.ax.yaxis.tick_left()
cb_freq.set_label("Frequency (count)", rotation=90, labelpad=-10)

cb_reca.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cb_reca.ax.tick_params(labelsize=8)
cb_reca.ax.yaxis.set_label_position('right'); cb_reca.ax.yaxis.tick_right()
cb_reca.set_label("Recall@10", rotation=270, labelpad=0)

for cax in (cax_freq, cax_reca):
    cax.grid(False)
    for sp in cax.spines.values():
        sp.set_visible(False)

fig.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"), dpi=200, bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "heatmap.pdf"), bbox_inches="tight")
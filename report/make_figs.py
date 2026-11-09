import sys, matplotlib, matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
out = sys.argv[1]
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

# Fig 1: model-only vs graded entry per phase (each phase on its own metric)
ph = ["P1\nBERTScore", "P2\nstrict F1", "P3\nMacro-F1", "P4\nMacro-F1", "P5\nMacro-F1", "P6\nBERTScore", "P7\nMicro-F1@7"]
model = [0.8449, 0.4873, 0.5637, 0.7225, 0.3949, 0.8391, 0.3325]
hyb   = [None,   None,   0.5879, 0.7488, None,   None,   0.4979]
final = [0.8689, 0.6515, None,   0.8767, 0.4632, 0.8606, 0.5157]
fig, ax = plt.subplots(figsize=(6.3, 2.1))
w = 0.27; x = range(len(ph))
def bars(vals, off, lab, c):
    xs = [i + off for i, v in enumerate(vals) if v is not None]
    vs = [v for v in vals if v is not None]
    ax.bar(xs, vs, w, label=lab, color=c)
    for a, b in zip(xs, vs): ax.text(a, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
bars(model, -w, "Phase-1 models only", "#4C72B0")
bars(hyb, 0, "best without corpus lookup (where different)", "#DD8452")
bars(final, w, "graded entry", "#55A868")
ax.set_xticks(list(x)); ax.set_xticklabels(ph); ax.set_ylim(0, 1.08); ax.set_ylabel("test score")
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7)
fig.tight_layout(); fig.savefig(f"{out}/fig_phase_scores.pdf"); plt.close(fig)

# Fig 2: data scaling, Phase 4 (dev Macro-F1 vs training documents)
fig, ax = plt.subplots(1, 2, figsize=(6.3, 1.9))
lex_n = [2.5, 5, 10, 20, 32.3]; lex_f = [0.6252, 0.6409, 0.6498, 0.6579, 0.6748]
ax[0].plot(lex_n, lex_f, "o-", color="#DD8452", label="TF-IDF + LR, judgment tail")
ax[0].plot([10, 24.3], [0.6343, 0.6760], "s-", color="#4C72B0", label="M4 LoRMA, gold only")
ax[0].plot([25, 46.6], [0.6731, 0.6905], "^--", color="#55A868", label="M4 LoRMA, gold + silver")
ax[0].set_xscale("log"); ax[0].set_xlabel("training documents (thousands)"); ax[0].set_ylabel("P4 dev Macro-F1")
ax[0].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax[0].set_xticks([2.5, 5, 10, 20, 46.6]); ax[0].set_xticklabels(["2.5", "5", "10", "20", "46.6"])
ax[0].legend(frameon=False, fontsize=6)
# Phase 1: retrieved count vs leaderboard BERTScore, with marginal quality of each batch
n = [0, 152, 241, 263, 308]; s = [0.8449, 0.8592, 0.8659, 0.8667, 0.8689]
ax[1].plot(n, s, "o-", color="#55A868")
for a, b, q, off in zip(n[1:], s[1:], ["0.939", "0.920", "0.881", "0.894"],
                        [(-6, 6), (-22, 4), (4, -9), (-6, 6)]):
    ax[1].annotate(q, (a, b), textcoords="offset points", xytext=off, fontsize=6)
ax[1].set_xlabel("P1 test prompts answered by retrieval"); ax[1].set_ylabel("P1 test BERTScore")
fig.tight_layout(); fig.savefig(f"{out}/fig_scaling.pdf"); plt.close(fig)
print("ok")

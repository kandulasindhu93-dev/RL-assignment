"""Metrics, CSV export and figures computed from the saved raw results."""
import glob, os, pickle, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import TASK_SHORT

COLORS = {"sequential": "#d1495b", "replay": "#2e86ab"}
LABELS = {"sequential": "Sequential (no replay)", "replay": "Sequential + Experience Replay"}
K = len(TASK_SHORT)


def load(out_dir="results"):
    eps, per, mat = [], [], []
    for f in sorted(glob.glob(f"{out_dir}/raw/*.pkl")):
        d = pickle.load(open(f, "rb"))
        eps += d["episodes"]; per += d["periodic"]; mat += d["matrix"]
    return pd.DataFrame(eps), pd.DataFrame(per), pd.DataFrame(mat)


# ----------------------------------------------------------------- metrics
def per_run_metrics(mat: pd.DataFrame, value="success_rate") -> pd.DataFrame:
    """Forgetting / backward-transfer / average performance for every run."""
    rows = []
    for (m, s), g in mat.groupby(["method", "seed"]):
        R = np.full((K, K), np.nan)                          # R[i,j]: after stage i, on task j
        for _, r in g.iterrows():
            R[int(r.after_stage) - 1, int(r.eval_task_idx) - 1] = r[value]
        # forgetting of task j = best past performance - final performance (tasks 1..K-1)
        forget = [np.max(R[j:K - 1, j]) - R[K - 1, j] for j in range(K - 1)]
        bwt = [R[K - 1, j] - R[j, j] for j in range(K - 1)]
        row = dict(method=m, seed=s, metric=value,
                   avg_final_performance=R[K - 1].mean(),
                   avg_forgetting=float(np.mean(forget)),
                   backward_transfer=float(np.mean(bwt)),
                   **{f"final_{TASK_SHORT[j]}": R[K - 1, j] for j in range(K)},
                   **{f"forgetting_{TASK_SHORT[j]}": forget[j] for j in range(K - 1)})
        rows.append(row)
    return pd.DataFrame(rows)


def summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in metrics.columns if c not in ("method", "seed", "metric")]
    mean = metrics.groupby("method")[cols].mean()
    std = metrics.groupby("method")[cols].std(ddof=1)
    out = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    return out


def mean_matrix(mat, method, value="success_rate"):
    g = mat[mat.method == method].groupby(["after_stage", "eval_task_idx"])[value].mean().unstack()
    return g.values


# ----------------------------------------------------------------- figures
def smooth(x, w):
    x = np.asarray(x, float)
    if len(x) < w: return x
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), c])


def fig_training_reward(ax, eps, stage, w=25):
    """Native training return per episode (moving average) for one stage/task."""
    for m in ["sequential", "replay"]:
        curves = []
        for s, g in eps[(eps.method == m) & (eps.stage == stage)].groupby("seed"):
            curves.append(smooth(g.sort_values("episode_in_stage").ret.values, w))
        n = min(len(c) for c in curves)
        A = np.vstack([c[:n] for c in curves])
        mu, sd = np.nanmean(A, 0), np.nanstd(A, 0)
        x = np.arange(1, n + 1)
        ax.plot(x, mu, color=COLORS[m], label=LABELS[m]); ax.fill_between(x, mu - sd, mu + sd, color=COLORS[m], alpha=.18)
    ax.set_title(f"Training reward - Stage {stage}: {TASK_SHORT[stage-1]}")
    ax.set_xlabel("Training episode (within stage)"); ax.set_ylabel(f"Return (moving avg, w={w})")
    ax.set_ylim(-0.05, 1.0); ax.grid(alpha=.3)


def fig_heatmap(ax, mat, method):
    M = mean_matrix(mat, method)
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdYlGn")
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="black", fontweight="bold" if j <= i else "normal")
    ax.set_xticks(range(K)); ax.set_xticklabels(TASK_SHORT)
    ax.set_yticks(range(K)); ax.set_yticklabels([f"After {i+1}\n({TASK_SHORT[i]})" for i in range(K)])
    ax.set_xlabel("Evaluated task"); ax.set_title(f"Success rate matrix - {LABELS[method]}", fontsize=9)
    return im


def fig_forgetting_bars(ax, metrics):
    x = np.arange(K - 1); wdt = 0.35
    for k, m in enumerate(["sequential", "replay"]):
        g = metrics[metrics.method == m]
        mu = [g[f"forgetting_{TASK_SHORT[j]}"].mean() for j in range(K - 1)]
        sd = [g[f"forgetting_{TASK_SHORT[j]}"].std(ddof=1) for j in range(K - 1)]
        ax.bar(x + (k - .5) * wdt, mu, wdt, yerr=sd, capsize=3, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(x); ax.set_xticklabels(TASK_SHORT[:K - 1])
    ax.set_ylabel("Forgetting (peak - final success rate)"); ax.set_title("Catastrophic forgetting per task")
    ax.axhline(0, color="k", lw=.8); ax.grid(axis="y", alpha=.3)


def fig_periodic(ax, per, method, steps_per_task):
    g = per[per.method == method]
    tcol = ["#2a9d8f", "#e9c46a", "#6a4c93"]
    for j in range(K):
        d = g[g.eval_task_idx == j + 1].groupby("global_step").success_rate.agg(["mean", "std"]).reset_index()
        ax.plot(d.global_step, d["mean"], marker="o", ms=3, color=tcol[j], label=TASK_SHORT[j])
        ax.fill_between(d.global_step, d["mean"] - d["std"], d["mean"] + d["std"], color=tcol[j], alpha=.12)
    for b in range(1, K):
        ax.axvline(b * steps_per_task, color="k", ls="--", lw=1)
    ax.set_ylim(-0.03, 1.03); ax.set_xlabel("Environment steps (all stages)"); ax.set_ylabel("Eval success rate")
    ax.set_title(f"All-task performance during training\n{LABELS[method]}", fontsize=9); ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower left")


def fig_avg_perf(ax, mat):
    """Average success rate over ALL three tasks and over SEEN tasks after each stage."""
    for m in ["sequential", "replay"]:
        g = mat[mat.method == m]
        seen = []; allt = []
        for st in range(1, K + 1):
            gs = g[g.after_stage == st]
            allt.append(gs.groupby("seed").success_rate.mean())
            seen.append(gs[gs.eval_task_idx <= st].groupby("seed").success_rate.mean())
        x = np.arange(1, K + 1)
        ax.errorbar(x, [a.mean() for a in seen], [a.std(ddof=1) for a in seen], color=COLORS[m],
                    marker="o", capsize=3, label=LABELS[m] + " (seen tasks)")
        ax.errorbar(x, [a.mean() for a in allt], [a.std(ddof=1) for a in allt], color=COLORS[m],
                    marker="s", ls="--", capsize=3, alpha=.6, label=LABELS[m] + " (all 3 tasks)")
    ax.set_xticks(range(1, K + 1)); ax.set_xticklabels([f"After\n{TASK_SHORT[i]}" for i in range(K)])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Average success rate"); ax.set_title("Average task performance per stage")
    ax.grid(alpha=.3); ax.legend(fontsize=6.5, loc="lower left")


def make_all_figures(out_dir="results"):
    from config import Config
    spt = Config().steps_per_task
    eps, per, mat = load(out_dir)
    metrics = per_run_metrics(mat)
    os.makedirs(f"{out_dir}/figures", exist_ok=True)

    # ---- composite figure ----
    fig, axs = plt.subplots(3, 3, figsize=(17, 14))
    for s in range(1, K + 1): fig_training_reward(axs[0, s - 1], eps, s)
    axs[0, 0].legend(fontsize=8, loc="lower right")
    im = fig_heatmap(axs[1, 0], mat, "sequential"); fig_heatmap(axs[1, 1], mat, "replay")
    fig_forgetting_bars(axs[1, 2], metrics); axs[1, 2].legend(fontsize=8)
    fig_periodic(axs[2, 0], per, "sequential", spt); fig_periodic(axs[2, 1], per, "replay", spt)
    fig_avg_perf(axs[2, 2], mat)
    fig.suptitle("Lifelong Learning Without Forgetting - MiniGrid (Double DQN, mean ± std over seeds)", fontsize=15, y=0.995)
    fig.tight_layout(); fig.savefig(f"{out_dir}/Learning_Curves.png", dpi=130); plt.close(fig)

    # ---- individual figures (used in the PDF / slides) ----
    def single(name, fn, size=(6.2, 4.4)):
        f, a = plt.subplots(figsize=size); fn(a); f.tight_layout()
        f.savefig(f"{out_dir}/figures/{name}.png", dpi=150); plt.close(f)
    for s in range(1, K + 1):
        single(f"training_reward_stage{s}", lambda a, s=s: fig_training_reward(a, eps, s))
    single("heatmap_sequential", lambda a: fig_heatmap(a, mat, "sequential"), (5.2, 4.2))
    single("heatmap_replay", lambda a: fig_heatmap(a, mat, "replay"), (5.2, 4.2))
    single("forgetting_bars", lambda a: fig_forgetting_bars(a, metrics))
    single("periodic_sequential", lambda a: fig_periodic(a, per, "sequential", spt))
    single("periodic_replay", lambda a: fig_periodic(a, per, "replay", spt))
    single("avg_performance", lambda a: fig_avg_perf(a, mat))
    return eps, per, mat, metrics


def export_csvs(out_dir="results"):
    eps, per, mat, metrics_sr = make_all_figures(out_dir)
    metrics_ret = per_run_metrics(mat, "mean_return")
    # Experiment_Results.csv: end-of-stage performance matrix for every run (raw numbers)
    mat.to_csv(f"{out_dir}/Experiment_Results.csv", index=False)
    eps.to_csv(f"{out_dir}/training_episodes.csv", index=False)
    per.to_csv(f"{out_dir}/periodic_eval.csv", index=False)
    pd.concat([metrics_sr, metrics_ret]).to_csv(f"{out_dir}/metrics_per_run.csv", index=False)
    summ_sr, summ_ret = summary_table(metrics_sr), summary_table(metrics_ret)
    summ_sr.to_csv(f"{out_dir}/metrics_summary_success_rate.csv")
    summ_ret.to_csv(f"{out_dir}/metrics_summary_mean_return.csv")
    for m in ["sequential", "replay"]:
        for v in ["success_rate", "mean_return"]:
            pd.DataFrame(mean_matrix(mat, m, v), index=[f"after_{t}" for t in TASK_SHORT],
                         columns=TASK_SHORT).round(4).to_csv(f"{out_dir}/matrix_{m}_{v}.csv")
    return mat, metrics_sr, metrics_ret, summ_sr, summ_ret


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results"
    mat, msr, mret, ssr, sret = export_csvs(out)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("== success-rate metrics (mean ± std over seeds) ==\n", ssr.T)
    print("\n== mean-return metrics ==\n", sret.T)

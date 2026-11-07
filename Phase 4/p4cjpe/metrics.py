import numpy as np

def binary_report(gold, pred):
    g, p = np.asarray(gold, dtype=np.int64), np.asarray(pred, dtype=np.int64)
    out = {"per_class": {}}
    for k in (0, 1):
        tp = int(((p == k) & (g == k)).sum())
        fp = int(((p == k) & (g != k)).sum())
        fn = int(((p != k) & (g == k)).sum())
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / (tp + fn) if tp + fn else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out["per_class"][k] = {"p": P, "r": R, "f1": F, "gold": int((g == k).sum()), "pred": int((p == k).sum())}
    pc = out["per_class"]
    out.update(macro_f1=(pc[0]["f1"] + pc[1]["f1"]) / 2, macro_p=(pc[0]["p"] + pc[1]["p"]) / 2,
               macro_r=(pc[0]["r"] + pc[1]["r"]) / 2, accuracy=float((g == p).mean()) if len(g) else 0.0,
               n=int(len(g)))
    out["confusion"] = [[int(((g == a) & (p == b)).sum()) for b in (0, 1)] for a in (0, 1)]
    try:
        from sklearn.metrics import f1_score
        out["sklearn_macro_f1"] = float(f1_score(g, p, average="macro", labels=[0, 1], zero_division=0))
    except ImportError:
        pass
    return out

def best_threshold(gold, prob1, lo=0.05, hi=0.95, step=0.01):
    g = np.asarray(gold)
    best = (-1.0, 0.5)
    for t in np.arange(lo, hi + 1e-9, step):
        f = binary_report(g, (np.asarray(prob1) >= t).astype(int))["macro_f1"]
        if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and abs(t - 0.5) < abs(best[1] - 0.5)):
            best = (f, float(round(t, 4)))
    return best[1], best[0]

def format_report(rep, title=""):
    pc = {int(k): v for k, v in rep["per_class"].items()}
    lines = [title] if title else []
    lines.append(f"{'class':8s} {'P':>6s} {'R':>6s} {'F1':>6s} {'gold':>6s} {'pred':>6s}")
    for k in (0, 1):
        lines.append(f"{k:<8d} {pc[k]['p']:6.3f} {pc[k]['r']:6.3f} {pc[k]['f1']:6.3f} {pc[k]['gold']:6d} {pc[k]['pred']:6d}")
    lines.append(f"{'MACRO':8s} {rep['macro_p']:6.3f} {rep['macro_r']:6.3f} {rep['macro_f1']:6.3f}   (accuracy {rep['accuracy']:.3f}, "
                 f"{rep['n']} cases, confusion [[TN FP] [FN TP]] = {rep['confusion']})")
    return "\n".join(lines)

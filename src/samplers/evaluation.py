
import numpy as np

def eval_healthy_and_synthetic(healthy_prop, synth_prop, eps_vec=None, method_string=""):
    """
    Evaluate and print statistics comparing healthy_prop and synth_prop.
    """

    healthy_sums = healthy_prop.sum(axis=1)
    synth_sums = synth_prop.sum(axis=1)

    print(f"[EVAL] Healthy samples: {healthy_prop.shape[0]}, Synthetic samples: {synth_prop.shape[0]}")
    print(f"[EVAL] Healthy sum stats: mean={healthy_sums.mean():.4f}, std={healthy_sums.std():.4f}, min={healthy_sums.min():.4f}, max={healthy_sums.max():.4f}")
    print(f"[EVAL] Synthetic sum stats: mean={synth_sums.mean():.4f}, std={synth_sums.std():.4f}, min={synth_sums.min():.4f}, max={synth_sums.max():.4f}")

    if eps_vec is not None:
        num_below_eps = (synth_prop < eps_vec).sum()
        total_elements = synth_prop.size
        print(f"[EVAL] Number of synthetic elements below eps: {num_below_eps} out of {total_elements} ({100.0 * num_below_eps / total_elements:.2f}%)")
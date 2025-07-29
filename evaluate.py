import numpy as np
import scipy.stats as stats


def get_percentile(score_dic: dict) -> np.ndarray:
    labels = np.array(list(score_dic.values()))
    percentiles = []
    for i in range(5):
        percentiles.append(len(labels[labels == i + 1]) / len(labels) * 100)
    percentiles = np.cumsum(percentiles)
    return percentiles


def get_scoremat_from_sim(
    sim_mat: np.ndarray,
    percentiles: np.ndarray | None,
) -> np.ndarray:
    score_mat = -1 * np.ones(sim_mat.shape)
    for i in range(len(sim_mat[0])):
        sims = sim_mat[:, i]
        sorted_inds = np.argsort(sims)
        constrained_inds = len(sims[sims == -1.0])
        valid_inds = sorted_inds[constrained_inds:]

        if percentiles is not None:
            percentile_inds = percentiles * len(valid_inds) // 100
        else:
            percentile_inds = np.round(np.linspace(0, len(valid_inds), 6)[1:])
        percentile_inds[-1] = len(valid_inds)
        start = 0
        for j, ind in enumerate(percentile_inds):
            end = int(ind)
            score_mat[valid_inds[start:end], i] = j + 1
            start = end
    return score_mat


def get_ranking(
    sim_matrix: np.ndarray,
    score_dic: dict[tuple[int, int], float],
    percentiles: np.ndarray,
    score_mat: np.ndarray,
    output_path: str,
) -> np.ndarray:
    sim_pairs = list(score_dic.keys())
    sim_vals = [sim_matrix[j, v] for j, v in sim_pairs]

    sorted_indices = np.argsort(sim_vals)
    score_ranks = np.zeros(len(sorted_indices))

    if percentiles is not None:
        percentile_inds = percentiles * len(sorted_indices) // 100
    else:
        percentile_inds = np.round(np.linspace(0, len(sorted_indices), 6)[1:])
    percentile_inds[-1] = len(sorted_indices)

    start = 0
    for rank, end in enumerate(percentile_inds, start=1):
        score_ranks[sorted_indices[start : int(end)]] = rank
        start = int(end)

    manual_scores = list(score_dic.values())
    full_sim_scores = [score_mat[j, v] for j, v in sim_pairs]

    tau1, p1 = stats.kendalltau(manual_scores, score_ranks)

    full_sorted = np.argsort(sim_vals)
    raw_ranks = np.empty_like(full_sorted)
    raw_ranks[full_sorted] = np.arange(len(full_sorted))
    tau2, p2 = stats.kendalltau(manual_scores, raw_ranks)

    with open(output_path, "a") as f:
        f.write(
            "Ranking based on manually labeled scores vs similarity-based buckets\n"
        )
        f.write(f"{tau1:.2f}, p={p1:.3f}\n\n")
        f.write("Manual scores vs full similarity rank order\n")
        f.write(f"{tau2:.2f}, p={p2:.3f}\n\n")

        for i, (j, v) in enumerate(sim_pairs):
            f.write(f"Pair: ({j}, {v})\n")
            f.write(
                f"Manual: {manual_scores[i]}, Bucket Rank: {score_ranks[i]}, "
                f"Full Rank: {full_sim_scores[i]}, Rank Index: {raw_ranks[i]}, "
                f"Sim Value: {sim_vals[i]}\n\n"
            )
        f.write("\n\n")
        f.write("----------------------------------------------------")
    return score_ranks

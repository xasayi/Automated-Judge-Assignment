import numpy as np
from scipy.optimize import minimize
from evaluate import get_percentile


def get_rankings_from_sim(data: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    n = len(data)
    ret = np.zeros(n)
    valid_inds = np.argsort(data)
    percentile_inds = percentiles * len(valid_inds) // 100
    percentile_inds[-1] = n
    start = 0
    for j, ind in enumerate(percentile_inds):
        end = int(ind)
        ret[valid_inds[start:end]] = j + 1
        start = end
    return ret


def optimize_similarity(
    filenames: list[str], score_dic: dict[tuple[int, int], float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n_samples = len(score_dic)
    n_features = len(filenames)
    X, y, norm_y = get_input_and_labels(filenames, score_dic)

    assert len(X) == n_samples
    assert len(X[0]) == n_features

    def negative_log_likelihood(
        weights: np.ndarray, X: np.ndarray, y: np.ndarray
    ) -> float:
        residuals = y - X @ weights
        sigma2 = np.var(residuals)
        log_likelihood = (
            -0.5 * len(y) * np.log(2 * np.pi * sigma2)
            - 0.5 * np.sum(residuals**2) / sigma2
        )
        return -log_likelihood

    constraints = {"type": "eq", "fun": lambda beta: np.sum(beta) - 1}
    bounds = [(0, 1) for _ in range(n_features)]
    initial_weights = np.random.uniform(0, 1, size=n_features)
    result = minimize(
        negative_log_likelihood,
        initial_weights,
        args=(X, norm_y),
        constraints=constraints,
        bounds=bounds,
    )

    estimated_weights = result.x
    print(f"Weights: {estimated_weights}")

    manual_scores = list(score_dic.values())
    auto_scores = get_rankings_from_sim(
        np.dot(estimated_weights, X.T), get_percentile(score_dic)
    )
    mean_diff = abs(manual_scores - auto_scores).mean()
    return estimated_weights, X, y, mean_diff


def get_input_and_labels(
    names: list[str], score_dic: dict[tuple[int, int], float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = list(score_dic.keys())
    model_sims = []
    for name in names:
        sim_mats = np.loadtxt(name)
        sims = [sim_mats[j, v] for j, v in pairs]
        normalized_sims = (sims - np.min(sims)) / (np.max(sims) - np.min(sims))
        model_sims.append(normalized_sims)

    X = np.array(model_sims).T
    y = np.array(list(score_dic.values()))
    norm_y = (y - np.min(y)) / (np.max(y) - np.min(y))
    return X, y, norm_y


def get_score_difference(
    filenames: list[str], score_dic: dict[tuple[int, int], float], weights: np.ndarray
) -> float:
    X, _, _ = get_input_and_labels(filenames, score_dic)

    pred_scores = np.dot(weights, X.T)
    percentiles = get_percentile(score_dic)
    pred_ranks = get_rankings_from_sim(pred_scores, percentiles)
    manual_ranks = np.array(list(score_dic.values()))
    return abs(pred_ranks - manual_ranks).mean()

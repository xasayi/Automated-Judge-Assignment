import math
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

from preprocess_data import parse_info, get_parsed_data


def compute_idf(dics: list[dict]) -> dict[str, float]:
    num_documents = sum([len(i) for i in dics])
    for d in dics:
        for document in d:
            d[document] = Counter(d[document])
    doc_count = defaultdict(lambda: 0.0)
    for d in dics:
        for document in d:
            for word in d[document]:
                doc_count[word] += 1.0
    idf = {word: math.log(num_documents / doc_count[word]) for word in doc_count}
    return idf


def compute_tf(count: int, total_count: int) -> float:
    return (1 + count / total_count) / 2


def tfidf(count: int, total_count: int, idf: float) -> float:
    return (compute_tf(count, total_count) * idf) ** 2


def augmented_tfidf(
    count1: int,
    tot_1: int,
    count2: int,
    tot_2: int,
    idf: float,
) -> float:
    return compute_tf(count1, tot_1) * compute_tf(count2, tot_2) * (idf**2)


def get_all_unique_words(dics: list[dict]) -> set[str]:
    all_words = set()
    for dic in dics:
        for key in dic:
            all_words |= set(dic[key])
    return all_words


def tfidf_sim_aug(
    counter1: dict[str, int], counter2: dict[str, int], idf: dict[str, float]
) -> float:
    tot_1, tot_2 = max(counter1.values()), max(counter2.values())
    norm_1, norm_2, sim = 0.0, 0.0, 0.0
    for word in counter1:
        norm_1 += tfidf(counter1[word], tot_1, idf[word])
        if word in counter2:
            sim += augmented_tfidf(
                counter1[word], tot_1, counter2[word], tot_2, idf[word]
            )
    for word in counter2:
        norm_2 += tfidf(counter2[word], tot_2, idf[word])
    return sim / math.sqrt(norm_1) / math.sqrt(norm_2)


def get_aug_tfidf(
    dics: list[dict],
    wiki_folder: str | None,
    stem: bool,
    sanitize: bool,
    lines: bool,
    keep_ui: bool,
) -> tuple[dict[str, float], set]:
    wiki_dic = None
    if wiki_folder is not None:
        wiki_dic = parse_info(
            folder=wiki_folder,
            stem=stem,
            sanitize=sanitize,
            lines=lines,
            keep_ui=keep_ui,
        )
        dics += [wiki_dic]
    idf = compute_idf(dics)
    all_words = get_all_unique_words(dics)
    assert len(all_words) == len(idf)
    return idf, all_words


def compute_tfidf_sim_aug(
    judge_folder: str,
    venture_folder: str,
    wiki_folder: str | None,
    stem: bool,
    sanitize: bool,
    keep_ui: bool,
) -> tuple[np.ndarray, dict, dict]:
    judges, ventures = get_parsed_data(
        judge_folder, venture_folder, stem, sanitize, False, keep_ui
    )
    ind_to_judge = {ind: judge for ind, judge in enumerate(judges.keys())}
    ind_to_venture = {ind: venture for ind, venture in enumerate(ventures.keys())}
    idf, _ = get_aug_tfidf(
        [judges, ventures], wiki_folder, stem, sanitize, False, keep_ui
    )
    similarity_matrix = -np.ones((len(judges), len(ventures)))

    for j_ind, judge in ind_to_judge.items():
        for v_ind, venture in ind_to_venture.items():
            j_counter, v_counter = judges[judge], ventures[str(venture)]

            if len(j_counter) != 0 and len(v_counter) != 0:
                similarity_matrix[j_ind, v_ind] = tfidf_sim_aug(
                    j_counter, v_counter, idf
                )
    return similarity_matrix, ind_to_judge, ind_to_venture


def compute_tfidf_sim(
    judge_folder: str,
    venture_folder: str,
    wiki_folder: str | None,
    stem: bool,
    sanitize: bool,
    keep_ui: bool,
) -> tuple[np.ndarray, dict, dict]:
    count_vectorizer = CountVectorizer(stop_words="english")
    tfidf = TfidfTransformer(
        smooth_idf=True, use_idf=True, norm="l2", sublinear_tf=False
    )

    judges, ventures = get_parsed_data(
        judge_folder, venture_folder, stem, sanitize, True, keep_ui
    )
    ind_to_judge = {ind: judge for ind, judge in enumerate(judges.keys())}
    ind_to_venture = {ind: venture for ind, venture in enumerate(ventures.keys())}

    judge_values = list(judges.values())
    venture_values = list(ventures.values())

    if wiki_folder is not None:
        wiki_dic = parse_info(wiki_folder, stem, sanitize, True, keep_ui)
        wiki_values = list(wiki_dic.values())
        word_count_vec = count_vectorizer.fit_transform(
            judge_values + venture_values + wiki_values
        )
    else:
        word_count_vec = count_vectorizer.fit_transform(judge_values + venture_values)

    tfidf.fit(word_count_vec)

    count_vec = count_vectorizer.transform(judge_values + venture_values)
    tfidf_vec = tfidf.transform(count_vec)
    similarity_matrix = linear_kernel(tfidf_vec, tfidf_vec)[
        : len(judge_values), len(judge_values) :
    ]
    return similarity_matrix, ind_to_judge, ind_to_venture


def get_smoothed_tfidf(
    judge_folder: str,
    venture_folder: str,
    wiki_folder: str | None,
    stem: bool,
    sanitize: bool,
    lines: bool,
    keep_ui: bool,
) -> dict[str, float]:
    count_vectorizer = CountVectorizer(stop_words="english")
    tfidf = TfidfTransformer(
        smooth_idf=True, use_idf=True, norm="l2", sublinear_tf=False
    )

    judges, ventures = get_parsed_data(
        judge_folder, venture_folder, stem, sanitize, lines, keep_ui
    )

    judge_values = list(judges.values())
    venture_values = list(ventures.values())

    if wiki_folder is not None:
        wiki_dic = parse_info(wiki_folder, stem, sanitize, lines, keep_ui)
        wiki_values = list(wiki_dic.values())
        word_count_vec = count_vectorizer.fit_transform(
            judge_values + venture_values + wiki_values
        )
    else:
        word_count_vec = count_vectorizer.fit_transform(judge_values + venture_values)
    tfidf.fit(word_count_vec)
    idf = {
        count_vectorizer.get_feature_names_out()[i]: tfidf.idf_[i]
        for i in range(len(tfidf.idf_))
    }
    return idf

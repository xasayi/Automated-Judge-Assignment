import time
import torch
import pandas as pd
import numpy as np
from torch import Tensor
from similarity.tfidf_sim import get_parsed_data, get_aug_tfidf, get_smoothed_tfidf
from similarity.embed_sim import (
    get_tokenizer_and_model,
    get_bert_embeddings,
    get_auto_embeddings,
    token_similarity,
)
from sklearn.metrics.pairwise import cosine_similarity


def tokenize_hybrid(
    df: pd.DataFrame, modelname: str, tokenizer, model, idf
) -> tuple[Tensor, Tensor, Tensor]:
    texts = df["info"].tolist()
    encoded = tokenizer.batch_encode_plus(
        texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
        return_offsets_mapping=True,
        return_attention_mask=True,
    )

    input_ids = torch.tensor(encoded["input_ids"])
    idf_weights = torch.zeros_like(input_ids, dtype=torch.float)

    for i in range(len(input_ids)):
        word_ids = encoded.word_ids(i)
        token_ids = input_ids[i].tolist()
        word_to_token_idxs = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                word_to_token_idxs.setdefault(word_id, []).append(idx)

        for word_id, token_idxs in word_to_token_idxs.items():
            token_strs = [
                tokenizer.convert_ids_to_tokens(token_ids[t]) for t in token_idxs
            ]
            word_str = tokenizer.convert_tokens_to_string(token_strs).lower()
            word_idf = idf.get(word_str, 1.0)
            for t in token_idxs:
                idf_weights[i, t] = word_idf

    start = time.time()
    with torch.no_grad():
        if (
            modelname == "bert-base-uncased"
            or modelname == "allenai/scibert_scivocab_uncased"
        ):
            attention_mask = torch.tensor(encoded["attention_mask"])
            token_embeddings, input_mask_expanded = get_bert_embeddings(
                model, attention_mask, input_ids
            )

        else:
            token_embeddings, input_mask_expanded = get_auto_embeddings(
                model, texts, input_ids.shape[1]
            )
    idf_weights = idf_weights.unsqueeze(-1).to(token_embeddings.device)
    print(f"Hybrid tokenization took {round(time.time() - start, 2)} seconds")

    token_embeddings *= input_mask_expanded
    return token_embeddings, idf_weights, input_mask_expanded


def get_hybrid_embeddings(
    judge_folder: str,
    venture_folder: str,
    wiki_folder: str | None,
    model: str,
    augmented_idf: bool,
    stem: bool,
    sanitize: bool,
    keep_ui: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, dict, dict]:
    if augmented_idf:
        judges, ventures = get_parsed_data(
            judge_folder,
            venture_folder,
            stem,
            sanitize,
            lines=False,
            keep_ui=keep_ui,
        )
        idf, _ = get_aug_tfidf(
            [judges, ventures],
            wiki_folder,
            stem,
            sanitize,
            lines=False,
            keep_ui=keep_ui,
        )
    else:
        idf = get_smoothed_tfidf(
            judge_folder,
            venture_folder,
            wiki_folder,
            stem,
            sanitize,
            lines=True,
            keep_ui=keep_ui,
        )

    judge_line, venture_line = get_parsed_data(
        judge_folder, venture_folder, stem, sanitize, lines=True, keep_ui=keep_ui
    )

    ind_to_judge = {ind: judge for ind, judge in enumerate(judge_line.keys())}
    ind_to_venture = {ind: venture for ind, venture in enumerate(venture_line.keys())}

    judge_df = pd.DataFrame(
        [{"index": i, "info": text} for i, (_, text) in enumerate(judge_line.items())]
    )
    venture_df = pd.DataFrame(
        [{"index": i, "info": text} for i, (_, text) in enumerate(venture_line.items())]
    )

    tokenizer, pretrained_model = get_tokenizer_and_model(model)

    judge_embeddings, judge_idf_weights, judge_mask = tokenize_hybrid(
        judge_df, model, tokenizer, pretrained_model, idf
    )

    venture_embeddings, venture_idf_weights, venture_mask = tokenize_hybrid(
        venture_df, model, tokenizer, pretrained_model, idf
    )

    return (
        judge_embeddings,
        judge_idf_weights,
        judge_mask,
        venture_embeddings,
        venture_idf_weights,
        venture_mask,
        ind_to_judge,
        ind_to_venture,
    )


def compute_hybrid_sim(
    judge_embeddings: Tensor,
    judge_idf_weights: Tensor,
    judge_mask: Tensor,
    venture_embeddings: Tensor,
    venture_idf_weights: Tensor,
    venture_mask: Tensor,
    token_level: bool,
) -> np.ndarray:
    start = time.time()
    if not token_level:

        def get_mean_embeddings(embed, mask, weights):
            embed *= weights
            sum_embed = torch.sum(embed, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_embed = sum_embed / sum_mask
            return mean_embed

        mean_judge_embed = get_mean_embeddings(
            judge_embeddings, judge_mask, judge_idf_weights
        )
        mean_venture_embed = get_mean_embeddings(
            venture_embeddings, venture_mask, venture_idf_weights
        )
        similarity_matrix = cosine_similarity(
            mean_judge_embed.numpy(), mean_venture_embed.numpy()
        )
    else:
        similarity_matrix = token_similarity(
            judge_embeddings, judge_mask, venture_embeddings, venture_mask
        )
    end = time.time()
    print(f"Calculating similarities took: {round(end - start, 2)} seconds")
    return similarity_matrix

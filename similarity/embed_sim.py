import time
import numpy as np
import torch
from torch import Tensor
import pandas as pd
from transformers import (
    BertTokenizerFast,
    BertModel,
    AutoTokenizer,
    AutoModel,
)

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess_data import get_parsed_data


def get_bert_embeddings(
    model: BertModel, attention_mask: Tensor, input_ids: Tensor
) -> tuple[Tensor, Tensor]:
    outputs = model(input_ids, attention_mask=attention_mask)
    token_embeddings = outputs[2][-1]  # last hidden layer
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return token_embeddings, input_mask_expanded


def get_auto_embeddings(
    model: SentenceTransformer, texts: list[str], max_len: int
) -> tuple[Tensor, Tensor]:
    token_embeddings_list = model.encode(
        texts,
        convert_to_tensor=True,
        output_value="token_embeddings",
    )
    token_embeddings = torch.stack(
        [
            F.pad(embed, (0, 0, 0, max_len - embed.size(0)))
            for embed in token_embeddings_list
        ]
    )
    input_mask_expanded = token_embeddings != 0
    return token_embeddings, input_mask_expanded


def tokenize(
    df: pd.DataFrame,
    modelname: str,
    tokenizer: PreTrainedTokenizerFast,
    model: BertModel | SentenceTransformer,
) -> tuple[Tensor, Tensor, Tensor]:
    texts = df["info"].tolist()
    encoded = tokenizer.batch_encode_plus(
        texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_token_type_ids=False,
    )
    input_ids = torch.tensor(encoded["input_ids"])
    attention_mask = torch.tensor(encoded["attention_mask"])

    start = time.time()
    with torch.no_grad():
        if (
            modelname == "bert-base-uncased"
            or modelname == "allenai/scibert_scivocab_uncased"
        ):
            token_embeddings, input_mask_expanded = get_bert_embeddings(
                model,  # pyright: ignore[reportArgumentType]
                attention_mask,
                input_ids,
            )

        else:
            token_embeddings, input_mask_expanded = get_auto_embeddings(
                model,  # pyright: ignore[reportArgumentType]
                texts,
                input_ids.shape[1],
            )
    print(f"Tokenization took {round(time.time() - start, 2)} seconds")

    masked_embeddings = token_embeddings * input_mask_expanded
    sum_embeddings = torch.sum(masked_embeddings, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask

    return token_embeddings, mean_embeddings, input_mask_expanded


def get_tokenizer_and_model(
    model: str,
) -> tuple[PreTrainedTokenizerFast, BertModel | SentenceTransformer]:
    if model == "bert-base-uncased":
        tokenizer = BertTokenizerFast.from_pretrained(model)
        pretrained_model = BertModel.from_pretrained(model, output_hidden_states=True)
    elif model == "allenai/scibert_scivocab_uncased":
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        pretrained_model = AutoModel.from_pretrained(
            "allenai/scibert_scivocab_uncased", output_hidden_states=True
        )
    else:
        if model == "sentence-transformers/all-distilroberta-v1":
            tokenizer = AutoTokenizer.from_pretrained(
                model, add_prefix_space=True, use_fast=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        pretrained_model = SentenceTransformer(
            model, revision=None, trust_remote_code=True
        )
    tokenizer.pad_token = "[PAD]"
    return tokenizer, pretrained_model


def compute_embedding_sim(
    judge_folder: str,
    venture_folder: str,
    model: str,
    stem: bool,
    sanitize: bool,
    keep_ui: bool,
    token_level: bool,
) -> tuple[np.ndarray, dict, dict]:
    judges, ventures = get_parsed_data(
        judge_folder, venture_folder, stem, sanitize, True, keep_ui
    )
    judge_df = pd.DataFrame(
        [{"index": i, "info": text} for i, (_, text) in enumerate(judges.items())]
    )
    venture_df = pd.DataFrame(
        [{"index": i, "info": text} for i, (_, text) in enumerate(ventures.items())]
    )

    tokenizer, pretrained_model = get_tokenizer_and_model(model)
    judge_embeddings, mean_judge_embed, judge_mask = tokenize(
        judge_df, model, tokenizer, pretrained_model
    )

    venture_embeddings, mean_venture_embed, venture_mask = tokenize(
        venture_df, model, tokenizer, pretrained_model
    )
    start = time.time()
    if not token_level:
        similarity_matrix = cosine_similarity(
            mean_judge_embed.cpu().numpy(), mean_venture_embed.cpu().numpy()
        )
    else:
        similarity_matrix = token_similarity(
            judge_embeddings, judge_mask, venture_embeddings, venture_mask
        )
    end = time.time()
    print(f"Calculating similarity matrix took: {round(end - start, 2)} seconds")

    ind_to_judge = {ind: judge for ind, judge in enumerate(judges.keys())}
    ind_to_venture = {ind: venture for ind, venture in enumerate(ventures.keys())}

    return similarity_matrix, ind_to_judge, ind_to_venture


def token_similarity(
    embeddings1: Tensor, mask1: Tensor, embeddings2: Tensor, mask2: Tensor
) -> np.ndarray:
    n1 = embeddings1.shape[0]
    n2 = embeddings2.shape[0]

    similarity_matrix = np.zeros((n1, n2))
    for i in range(n1):
        valid_emb1 = embeddings1[
            i,
            mask1[i, :, 0].bool(),
        ]
        norm_emb1 = F.normalize(valid_emb1, dim=1)
        for j in range(n2):
            valid_emb2 = embeddings2[
                j,
                mask2[j, :, 0].bool(),
            ]
            norm_emb2 = F.normalize(valid_emb2, dim=1)
            sim_mat = torch.matmul(norm_emb1, norm_emb2.T)
            similarity_matrix[i, j] = sim_mat.mean()
    return similarity_matrix

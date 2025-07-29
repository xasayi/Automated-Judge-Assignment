import re
import unicodedata
from pathlib import Path
from nltk.stem import PorterStemmer

from constants import ACCENT_MAP, UI_WORDS

SPACE_REGEXP = re.compile("[^a-zA-Z]")


def isUIWord(word: str) -> bool:
    return len(word) <= 1 or word in UI_WORDS


def sanitize(line: str) -> str:
    for char, replace_char in ACCENT_MAP.items():
        line = line.replace(char, replace_char)
    return "".join(
        c for c in unicodedata.normalize("NFD", line) if unicodedata.category(c) != "Mn"
    )


def tokenize(line: str, sanitize_bool: bool) -> list[str]:
    line = sanitize(line) if sanitize_bool else line
    return [w for w in SPACE_REGEXP.split(line) if w]


def preprocess(
    text: str, ps: PorterStemmer, stem: bool, sanitize_bool: bool, keep_ui: bool
) -> list[str]:
    words = tokenize(text, sanitize_bool)
    if not keep_ui:
        words = [w for w in words if not isUIWord(w)]
    if stem:
        words = [ps.stem(w) for w in words]
    return words


def parse_info(
    folder: str,
    stem: bool = True,
    sanitize: bool = True,
    lines: bool = False,
    keep_ui: bool = False,
) -> dict:
    folder_path = Path(folder)
    dic = {}
    ps = PorterStemmer()
    for textfile in folder_path.glob("*.txt"):
        with open(textfile, "r", encoding="utf-8") as fin:
            text = fin.read().lower()
            text = text.replace("\\u00b7", "")
            words = preprocess(text, ps, stem, sanitize, keep_ui)
            if lines:
                words = " ".join([x for x in words if x])
            dic[str(textfile.stem)] = words
    return dic


def get_parsed_data(
    judge_folder: str,
    venture_folder: str,
    stem: bool,
    sanitize: bool,
    lines: bool,
    keep_ui: bool,
) -> tuple[dict, dict]:
    judges = parse_info(
        folder=judge_folder,
        stem=stem,
        sanitize=sanitize,
        lines=lines,
        keep_ui=keep_ui,
    )
    ventures = parse_info(
        folder=venture_folder,
        stem=stem,
        sanitize=sanitize,
        lines=lines,
        keep_ui=keep_ui,
    )
    return judges, ventures

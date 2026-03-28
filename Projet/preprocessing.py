import re
from unidecode import unidecode

def to_lower_case(text: str) -> str:
    return text.lower()

def remove_ponct(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text)

def remove_numbers(text: str) -> str:
    return re.sub(r"\d+", "", text)

def remove_accents(text: str) -> str:
    return unidecode(text)

def keep_first_line(text: str) -> str:
    return text.split("\n")[0]

def keep_last_line(text: str) -> str:
    return text.split("\n")[-1]

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def preprocess(texts: list,
               lower: bool = True,
               ponct: bool = True,
               numbers: bool = True,
               accents: bool = True,
               keep: str = None) -> list:
    
    pretreated_texts = []
    
    for text in texts:
        if keep == "first":
            text = keep_first_line(text)
        elif keep == "last":
            text = keep_last_line(text)
    
        if accents:
            text = remove_accents(text)
    
        if lower:
            text = to_lower_case(text)
    
        if ponct:
            text = remove_ponct(text)
    
        if numbers:
            text = remove_numbers(text)
    
        text = normalize_spaces(text)
        pretreated_texts.append(text)

    return pretreated_texts

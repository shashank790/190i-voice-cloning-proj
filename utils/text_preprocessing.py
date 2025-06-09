import re
from unidecode import unidecode
from datetime import datetime

_whitespace_re = re.compile(r"\s+")

_abbreviations = [
    (re.compile(r"\b%s\.?" % abbr, re.IGNORECASE), full)
    for abbr, full in [
        ("mrs", "misses"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "street"),
        ("rd", "road"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)

def convert_to_ascii(text):
    return unidecode(text)

def normalize_numbers(text):
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text

def normalize_currency(text):
    text = re.sub(r"\$(\d{1,3}(,\d{3})*(\.\d+)?)", lambda m: m.group(1).replace(",", ""), text)
    return text

def normalize_dates(text):
    def replace_date(match):
        date = datetime.strptime(match.group(), "%m/%d/%y")
        return date.strftime("%B %d, %Y")
    text = re.sub(r"\b\d{2}[-/]\d{2}[-/]\d{2}\b", replace_date, text)
    return text

def normalize_quotes(text):
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return text

def preprocess_text(text):
    text = convert_to_ascii(text)
    text = expand_abbreviations(text)
    text = normalize_numbers(text)
    text = normalize_currency(text)
    text = normalize_dates(text)
    text = normalize_quotes(text)
    text = collapse_whitespace(text)
    return text.strip()
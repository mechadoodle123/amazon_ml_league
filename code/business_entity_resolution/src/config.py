import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
STUDENT_RESOURCE_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "../../student_resource"))
OUTPUT_DIR = os.path.join(STUDENT_RESOURCE_DIR, "output")
DEFAULT_DATA_DIR = os.path.join(STUDENT_RESOURCE_DIR, "dataset")

# Blocking and inference parameters
MAX_CANDIDATES = 25
MAX_BUCKET_SIZE = 300
MATCH_THRESHOLD = 0.90

HONORIFICS = {"mr", "mrs", "ms", "dr", "shri", "smt", "prof", "er", "ca"}

LEGAL_SUFFIXES = {
    # US / General
    "inc", "incorporated", "corp", "corporation", "llc", "l.l.c.", "llp", "l.l.p.",
    "co", "company", "ltd", "limited", "pllc", "lp", "enterprises", "enterprise",
    "services", "solutions", "group", "holdings", "associates", "partners",
    "center", "centre", "international", "consulting", "consultants", "management", "industries",
    # Professional
    "dds", "md", "dmd", "do", "dvm", "esq",
    # India
    "pvt", "private", "pvtltd", "prv", "limited", "ltd",
    # France
    "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", "gie"
}

STOP_WORDS = {
    "the", "and", "of", "in", "for", "on", "at", "to", "a", "an", "by", "with",
    "near", "opp", "opposite", "behind", "beside", "floor", "flr", "flat", "unit",
    "plot", "road", "street", "st", "rd", "ave", "avenue", "lane", "ln", "dr", "drive",
    "cross", "main", "sector", "nagar", "colony", "bhavan", "complex", "building",
    "house", "hno", "door", "no", "shop", "office"
}

FEATURE_NAMES = [
    "f_name_ratio",
    "f_name_token_sort",
    "f_name_token_set",
    "f_core_token_sort",
    "jacc_name",
    "jacc_core",
    "len_ratio_name",
    "f_addr_token_sort",
    "f_addr_token_set",
    "jacc_addr",
    "num_common",
    "num_diff",
    "has_num_both",
    "num_exact_match",
    "is_addr_empty",
    "is_source2",
    "score_approx"
]

import re
from collections import defaultdict, Counter
from preprocess import clean_text, get_tokens, get_core_name, extract_addr_numbers
from config import STOP_WORDS, MAX_BUCKET_SIZE, MAX_CANDIDATES

def generate_blocking_keys(name: str, addr: str, country: str) -> list[str]:
    """Generate high-precision, high-recall blocking keys for candidate retrieval."""
    keys = []
    n_tokens = get_tokens(name)
    core_name = get_core_name(n_tokens)
    a_tokens = get_tokens(addr)
    
    # 1. Sorted core name 2-token signature
    if len(core_name) >= 2:
        s_two = sorted(core_name[:3])[:2]
        keys.append(f"n2:{country}:{s_two[0]}_{s_two[1]}")
    elif len(core_name) == 1 and len(core_name[0]) >= 4:
        keys.append(f"n1:{country}:{core_name[0]}")

    # 2. Compressed core name (e.g. 'Seven Sun' -> 'sevensun')
    comp = "".join(core_name)
    if len(comp) >= 5:
        keys.append(f"cn:{country}:{comp[:12]}")
    
    # 3. Individual rare core name tokens (length >= 5)
    for w in core_name[:3]:
        if len(w) >= 5 and w not in STOP_WORDS:
            keys.append(f"rw:{country}:{w}")

    # 4. Address number + informative address tokens (locality, city, state)
    nums = extract_addr_numbers(a_tokens)
    addr_words = [
        w for w in a_tokens
        if not re.search(r"\d", w) and len(w) >= 4 and w not in STOP_WORDS
    ]
    
    for num in nums[:2]:
        for aw in addr_words[:3]:
            keys.append(f"na:{country}:{num}_{aw}")
        if core_name:
            keys.append(f"nn:{country}:{num}_{core_name[0][:3]}")

    return list(set(keys))

def build_country_inverted_index(source2_path: str, source3_path: str, target_country: str):
    """
    Build an inverted index for a specific country from Source 2 and Source 3 files.
    Returns:
        inverted_index: dict mapping blocking_key -> list of entity_ids
        records_map: dict mapping entity_id -> (name, addr, country)
    """
    inverted_index = defaultdict(list)
    records_map = {}

    for src_path in [source2_path, source3_path]:
        with open(src_path, "r", encoding="utf-8") as f:
            f.readline()  # skip header
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 4 and parts[3] == target_country:
                    eid = parts[0]
                    name = parts[1]
                    addr = parts[2] if len(parts) > 2 else ""
                    country = parts[3]
                    records_map[eid] = (name, addr, country)
                    for k in generate_blocking_keys(name, addr, country):
                        inverted_index[k].append(eid)

    return inverted_index, records_map

def retrieve_top_candidates(
    name: str,
    addr: str,
    country: str,
    inverted_index: dict,
    max_cands: int = MAX_CANDIDATES,
    max_bucket: int = MAX_BUCKET_SIZE
) -> list[str]:
    """Retrieve top candidates for an S1 entity ranked by blocking key co-occurrence."""
    keys = generate_blocking_keys(name, addr, country)
    key_hits = Counter()
    for k in keys:
        bucket = inverted_index.get(k, [])
        if 0 < len(bucket) <= max_bucket:
            for mid in bucket:
                key_hits[mid] += 1

    if not key_hits:
        return []

    # Sort candidates by number of matching blocking keys
    top_candidates = [mid for mid, _ in key_hits.most_common(max_cands)]
    return top_candidates

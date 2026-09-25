from rapidfuzz import fuzz
from preprocess import clean_text, get_core_name, extract_addr_numbers

def extract_pair_features(r1: tuple, r2: tuple, m_id: str) -> list[float]:
    """
    Extract discriminative similarity features between Source 1 record r1 and candidate r2.
    r1: (name, addr, country)
    r2: (name, addr, country)
    m_id: candidate entity_id (e.g. S2-xxx or S3-xxx)
    """
    n1_clean = clean_text(r1[0])
    n2_clean = clean_text(r2[0])
    
    t_n1 = n1_clean.split()
    t_n2 = n2_clean.split()
    core1 = " ".join(get_core_name(t_n1))
    core2 = " ".join(get_core_name(t_n2))
    
    a1_clean = clean_text(r1[1])
    a2_clean = clean_text(r2[1])
    
    t_a1 = a1_clean.split()
    t_a2 = a2_clean.split()
    
    set_n1 = set(t_n1)
    set_n2 = set(t_n2)
    jacc_name = len(set_n1 & set_n2) / max(1, len(set_n1 | set_n2))
    
    set_c1 = set(core1.split())
    set_c2 = set(core2.split())
    jacc_core = len(set_c1 & set_c2) / max(1, len(set_c1 | set_c2))
    
    set_a1 = set(t_a1)
    set_a2 = set(t_a2)
    jacc_addr = len(set_a1 & set_a2) / max(1, len(set_a1 | set_a2)) if t_a2 else 0.0
    
    nums1 = set(extract_addr_numbers(t_a1))
    nums2 = set(extract_addr_numbers(t_a2))
    num_common = float(len(nums1 & nums2))
    num_diff = float(len(nums1 ^ nums2))
    has_num_both = 1.0 if (nums1 and nums2) else 0.0
    num_exact_match = 1.0 if (has_num_both and nums1 == nums2) else 0.0

    f_name_ratio = fuzz.ratio(n1_clean, n2_clean) / 100.0
    f_name_token_sort = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0
    f_name_token_set = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0
    f_core_token_sort = fuzz.token_sort_ratio(core1, core2) / 100.0
    
    f_addr_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0 if t_a2 else 0.0
    f_addr_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0 if t_a2 else 0.0
    
    is_addr_empty = 1.0 if not t_a2 else 0.0
    is_source2 = 1.0 if m_id.startswith("S2-") else 0.0
    len_ratio_name = min(len(n1_clean), len(n2_clean)) / max(1, max(len(n1_clean), len(n2_clean)))
    
    score_approx = 0.5 * f_name_token_sort + 0.5 * f_addr_token_sort
    
    return [
        f_name_ratio,
        f_name_token_sort,
        f_name_token_set,
        f_core_token_sort,
        jacc_name,
        jacc_core,
        len_ratio_name,
        f_addr_token_sort,
        f_addr_token_set,
        jacc_addr,
        num_common,
        num_diff,
        has_num_both,
        num_exact_match,
        is_addr_empty,
        is_source2,
        score_approx
    ]

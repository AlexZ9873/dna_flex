import yaml

def load_lookup_yaml(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def get_feature_table(data: dict, group: str, feature_name: str):
    return data[group][feature_name]

def kmer_to_dinuc_average(kmer: str, table: dict):
    kmer = kmer.upper()
    values = []
    for i in range(len(kmer) - 1):
        dinuc = kmer[i:i+2]
        values.append(table[dinuc])
    return sum(values) / len(values)

def sequence_to_dinuc_targets(seq: str, k: int, table: dict):
    seq = seq.upper()
    kmers = []
    targets = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmers.append(kmer)
        targets.append(kmer_to_dinuc_average(kmer, table))
    return kmers, targets

def sequence_to_multi_dinuc_targets(seq: str, k: int, lookup_data: dict, feature_names: list):
    all_feature_targets = []
    kmers_ref = None

    for feature_name in feature_names:
        table = get_feature_table(lookup_data, "dinucleotide", feature_name)
        kmers, targets = sequence_to_dinuc_targets(seq, k, table)

        if kmers_ref is None:
            kmers_ref = kmers
        else:
            assert kmers == kmers_ref

        all_feature_targets.append(targets)

    combined_targets = []
    num_tokens = len(kmers_ref)

    for token_idx in range(num_tokens):
        row = []
        for feature_idx in range(len(feature_names)):
            row.append(all_feature_targets[feature_idx][token_idx])
        combined_targets.append(row)

    return kmers_ref, combined_targets

def kmer_to_trinuc_average(kmer: str, table: dict):
    kmer = kmer.upper()
    values = []
    for i in range(len(kmer) - 2):
        trinuc = kmer[i:i+3]
        values.append(table[trinuc])
    return sum(values) / len(values)

def sequence_to_trinuc_targets(seq: str, k: int, table: dict):
    seq = seq.upper()
    kmers = []
    targets = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmers.append(kmer)
        targets.append(kmer_to_trinuc_average(kmer, table))
    return kmers, targets

def sequence_to_multi_trinuc_targets(seq: str, k: int, lookup_data: dict, feature_names: list):
    all_feature_targets = []
    kmers_ref = None

    for feature_name in feature_names:
        table = get_feature_table(lookup_data, "trinucleotide", feature_name)
        kmers, targets = sequence_to_trinuc_targets(seq, k, table)

        if kmers_ref is None:
            kmers_ref = kmers
        else:
            assert kmers == kmers_ref

        all_feature_targets.append(targets)

    combined_targets = []
    num_tokens = len(kmers_ref)

    for token_idx in range(num_tokens):
        row = []
        for feature_idx in range(len(feature_names)):
            row.append(all_feature_targets[feature_idx][token_idx])
        combined_targets.append(row)

    return kmers_ref, combined_targets

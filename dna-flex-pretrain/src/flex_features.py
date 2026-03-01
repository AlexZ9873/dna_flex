import yaml

def load_lookup_yaml(path: str):
    """
    Load the full lookup YAML file into a Python dictionary.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def get_feature_table(data: dict, group: str, feature_name: str):
    """
    Extract one feature table from the loaded YAML.

    Example:
        group = "dinucleotide"
        feature_name = "twistDisp"
    """
    return data[group][feature_name]

def kmer_to_dinuc_average(kmer: str, table: dict):
    """
    Convert one k-mer into one number by averaging the values
    of all overlapping dinucleotides inside it.

    Example:
        kmer = ACGTAC
        dinucs = AC, CG, GT, TA, AC
    """
    kmer = kmer.upper()
    values = []

    for i in range(len(kmer) - 1):
        dinuc = kmer[i:i+2]
        value = table[dinuc]
        values.append(value)

    avg_value = sum(values) / len(values)
    return avg_value

def sequence_to_dinuc_targets(seq: str, k: int, table: dict):
    """
    For a full sequence, split into overlapping k-mers,
    then compute one averaged dinucleotide target per k-mer.

    Returns:
        kmers: list of k-mer strings
        targets: list of floats, one per k-mer
    """
    seq = seq.upper()
    kmers = []
    targets = []

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        kmers.append(kmer)

        target = kmer_to_dinuc_average(kmer, table)
        targets.append(target)

    return kmers, targets

"""Show exactly how NT tokenizes different sequences."""
from transformers import AutoTokenizer
from config import PRETRAINED_MODEL

tok = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)

# Check vocabulary size and structure
print(f"Vocab size: {tok.vocab_size}")
print(f"Special tokens: {tok.all_special_tokens}")

# Sample some tokens from the vocabulary to understand the distribution
vocab = tok.get_vocab()
by_length = {}
for token, idx in vocab.items():
    if all(c in "ATGC" for c in token) and len(token) > 0:
        l = len(token)
        by_length.setdefault(l, []).append(token)

print("\nNucleotide tokens in vocabulary by length:")
for length in sorted(by_length.keys()):
    count = len(by_length[length])
    examples = by_length[length][:5]
    print(f"  {length}-mer: {count} tokens (e.g., {examples})")

# Now show tokenization of diverse sequences
seqs = [
    "GCCTCAGTCTGCTTCGCACC",
    "TTAGGGTTAGGGTTAGGGTTAGGG",
    "ATGAAAGCGATTATTGGTCTGGGTGCTTATGAGAATCTGTACTTCCAATCCGATAAAGCG",
    "GGGGGGGGGGGGGGGGGGGG",
    "ATCGATATCGATAGCTATCGATATCGAT",
]

print("\nTokenization examples:")
for seq in seqs:
    encoded = tok(seq, return_tensors="pt")
    ids = encoded["input_ids"].squeeze().tolist()
    tokens = tok.convert_ids_to_tokens(ids)
    # Show token boundaries
    content_tokens = [t for t in tokens if t != "<cls>"]
    lens = [len(t) for t in content_tokens]
    print(f"\n  {seq}")
    print(f"  Tokens: {content_tokens}")
    print(f"  Token lengths: {lens}")
    print(f"  Total chars from tokens: {sum(lens)} (seq length: {len(seq)})")

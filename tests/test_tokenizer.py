"""Check what the tokenizer actually produces."""
from transformers import AutoTokenizer
from config import PRETRAINED_MODEL

tok = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, trust_remote_code=True)

seqs = [
    "TTAGGGTTAGGGTTAGGGTTAGGG",
    "GCCTCAGTCTGCTTCGCACC",
    "ATGCTAGCTAACGTACGATC",
]

for seq in seqs:
    encoded = tok(seq, return_tensors="pt")
    ids = encoded["input_ids"].squeeze().tolist()
    tokens = tok.convert_ids_to_tokens(ids)
    print(f"Sequence: {seq} ({len(seq)} nt)")
    print(f"  Tokens ({len(tokens)}): {tokens}")
    print(f"  IDs: {ids}")
    # Check special token IDs
    print(f"  CLS id: {tok.cls_token_id}, SEP id: {tok.sep_token_id}, "
          f"EOS id: {tok.eos_token_id}, PAD id: {tok.pad_token_id}")
    print()

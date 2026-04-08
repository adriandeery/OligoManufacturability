"""
Chemical modification suggestions via Claude API.

Role of the LLM in this system:
  - The Transformer model identifies WHICH positions are problematic
    (via attention/IG attributions) and gives an overall score.
  - The LLM suggests WHAT chemical modifications could improve
    manufacturability at those positions.
  - This is a recommendation task, not an interpretation task.
    The model's explainability is intrinsic; the LLM adds domain
    knowledge about the modification chemistry design space.

Common oligonucleotide modifications and why they matter:

  Backbone modifications:
    - Phosphorothioate (PS): replaces O with S in backbone.
      Improves nuclease resistance AND can improve coupling in some contexts.
    - Phosphorodiamidate morpholino (PMO): entirely different backbone chemistry.

  Sugar modifications (2' position):
    - 2'-O-Methyl (2'-OMe): improves binding affinity and nuclease resistance.
      Generally synthesises well but can reduce coupling efficiency if overused.
    - 2'-O-Methoxyethyl (2'-MOE): gold standard for ASO wings (gapmers).
      Good synthesis compatibility.
    - 2'-Fluoro (2'-F): high binding affinity, good synthesis.
    - Locked Nucleic Acid (LNA): bicyclic sugar locks ribose in C3'-endo.
      Highest binding affinity but expensive and can cause hepatotoxicity.

  Base modifications:
    - 5-Methyl Cytosine (5mC): replaces C, reduces immunostimulation.
      Also slightly improves coupling efficiency for C-rich regions.
"""

import os
from typing import Dict, List, Optional
import numpy as np

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from config import CLAUDE_MODEL


def _build_prompt(
    sequence: str,
    score: float,
    features: Dict[str, float],
    attributions: np.ndarray,
    top_k_positions: int = 5,
) -> str:
    """
    Build a structured prompt for Claude to suggest modifications.

    We give the LLM:
      1. The sequence and its manufacturability score
      2. The computed features (GC, homopolymers, etc.)
      3. The top-K most problematic positions (from attributions)
      4. A clear instruction to suggest modifications, not interpret the model
    """
    # Find top-K problematic positions
    top_positions = np.argsort(attributions)[-top_k_positions:][::-1]
    position_details = []
    for pos in top_positions:
        if pos < len(sequence):
            # Show local context (±3 nt)
            start = max(0, pos - 3)
            end = min(len(sequence), pos + 4)
            context = sequence[start:end]
            marker_pos = pos - start
            marked = context[:marker_pos] + f"[{context[marker_pos]}]" + context[marker_pos + 1:]
            position_details.append(
                f"  Position {pos + 1} ({sequence[pos]}): "
                f"attribution={attributions[pos]:.3f}, context=...{marked}..."
            )

    positions_str = "\n".join(position_details)

    return f"""You are an oligonucleotide chemistry expert advising on synthesis optimisation.

Given the following oligonucleotide sequence and its manufacturability analysis, suggest specific chemical modifications to improve synthesis yield and purity.

SEQUENCE: 5'-{sequence}-3'
LENGTH: {len(sequence)} nt
MANUFACTURABILITY SCORE: {score:.1f}/100

COMPUTED FEATURES:
- GC content: {features.get('gc_content', 0):.1%}
- Longest homopolymer run: {features.get('max_homopolymer', 0):.0f} nt
- PolyG run: {features.get('polyG_run', 0):.0f} nt
- Self-complementarity: {features.get('self_complementarity', 0):.1%}
- Cumulative yield estimate: {features.get('length_yield', 0):.1%}
- Dinucleotide complexity: {features.get('dinucleotide_complexity', 0):.2f}

TOP PROBLEMATIC POSITIONS (from model attribution):
{positions_str}

INSTRUCTIONS:
1. Based on the features and flagged positions, suggest 2-4 specific chemical modifications.
2. For each modification, specify:
   - Which positions to modify
   - What modification to use (PS, 2'-OMe, 2'-MOE, 2'-F, LNA, 5mC, etc.)
   - Why this modification helps at this position (synthesis chemistry rationale)
   - Any trade-offs (cost, efficacy impact, toxicity)
3. If the sequence has fundamental design problems (e.g., long polyG), say so directly and suggest sequence redesign rather than just modifications.
4. Keep suggestions practical for a GMP manufacturing context.

Respond in structured format with clear headings."""


def suggest_modifications(
    sequence: str,
    score: float,
    features: Dict[str, float],
    attributions: np.ndarray,
    api_key: Optional[str] = None,
) -> str:
    """
    Get modification suggestions from Claude API.

    Falls back to a rule-based suggestion if no API key is available.
    """
    if not HAS_ANTHROPIC:
        return _fallback_suggestions(sequence, score, features, attributions)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return _fallback_suggestions(sequence, score, features, attributions)

    prompt = _build_prompt(sequence, score, features, attributions)

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def _fallback_suggestions(
    sequence: str,
    score: float,
    features: Dict[str, float],
    attributions: np.ndarray,
) -> str:
    """
    Rule-based fallback when Claude API is unavailable.

    These rules reflect standard oligo chemistry best practices.
    """
    suggestions = []

    # G-quadruplex detection: multi-tract GGG pattern
    # A G-quadruplex forms when 4+ separate GGG tracts are present,
    # even if no single run exceeds 3. This is the telomeric repeat problem.
    import re
    g_tracts = re.findall(r'G{3,}', sequence)
    if len(g_tracts) >= 4:
        suggestions.append(
            "**G-Quadruplex Risk ({} G-tracts detected)**\n"
            "- {} separate GGG+ tracts can fold into an intramolecular G-quadruplex\n"
            "- This is a known synthesis failure mode for telomeric and G-rich sequences\n"
            "- **First choice**: 7-deaza-dG substitution at the central G of each tract "
            "(disrupts Hoogsteen bonding, preserves Watson-Crick pairing)\n"
            "- **Alternative**: 2'-OMe-G at one position per tract\n"
            "- **Process**: extend coupling time to 6 min at G positions, "
            "consider elevated coupling temperature (40C)".format(len(g_tracts), len(g_tracts))
        )

    # PolyG runs (consecutive, separate from multi-tract G-quad)
    polyG = features.get("polyG_run", 0)
    if polyG >= 4 and len(g_tracts) < 4:
        suggestions.append(
            "**PolyG Run Detected (>=4 consecutive G)**\n"
            "- Consider 7-deaza-G substitution at internal G positions to disrupt "
            "G-quadruplex formation\n"
            "- Alternatively, introduce 2'-OMe-G modifications to reduce self-association\n"
            "- If sequence design allows, consider breaking the G-run with a mismatch-tolerant position"
        )

    # High GC
    gc = features.get("gc_content", 0.5)
    if gc > 0.65:
        suggestions.append(
            "**High GC Content ({:.0%})**\n"
            "- Consider 5-methyl-C substitutions to reduce secondary structure\n"
            "- 2'-OMe modifications in GC-rich regions can improve coupling access\n"
            "- Verify dissolution protocol — high-GC oligos may need heated TE buffer".format(gc)
        )
    elif gc < 0.35:
        suggestions.append(
            "**Low GC Content ({:.0%})**\n"
            "- AT-rich oligos generally synthesise well but check Tm for target binding\n"
            "- Consider LNA modifications at critical binding positions to compensate for low Tm".format(gc)
        )

    # Self-complementarity
    self_comp = features.get("self_complementarity", 0)
    if self_comp > 0.15:
        suggestions.append(
            "**Self-Complementarity Detected ({:.0%})**\n"
            "- Hairpin-forming regions reduce coupling efficiency\n"
            "- Consider 2'-OMe or 2'-MOE modifications in the stem region to destabilise intramolecular folding\n"
            "- If this is an ASO, verify the hairpin doesn't overlap with the target-binding region".format(self_comp)
        )

    # Long sequence
    length_yield = features.get("length_yield", 1.0)
    if length_yield < 0.65:
        suggestions.append(
            "**Long Sequence (estimated yield {:.0%})**\n"
            "- Consider phosphorothioate backbone throughout to improve stepwise coupling\n"
            "- Use extended coupling times (>3 min) for positions >40\n"
            "- May need PAGE purification instead of HPLC for better n/n-1 resolution".format(length_yield)
        )

    if not suggestions:
        suggestions.append(
            "**Sequence looks good for synthesis (score: {:.0f}/100)**\n"
            "- Standard phosphoramidite synthesis should work well\n"
            "- Consider PS backbone at terminal positions (3 nt each end) for nuclease resistance if this is a therapeutic ASO".format(score)
        )

    return "\n\n".join(suggestions)

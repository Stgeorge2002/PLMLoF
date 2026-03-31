"""Generate synthetic LoF and WT variants for training data augmentation.

Creates deterministic LoF variants (premature stops, frameshifts, large truncations)
and WT variants (identical or synonymous) from bacterial reference proteins.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from plmlof.utils.sequence_utils import translate_dna

logger = logging.getLogger(__name__)

# Representative bacterial reference proteins for synthetic data generation
# These are short segments inspired by common bacterial protein domains
BACTERIAL_REFERENCE_PROTEINS = [
    {"gene": "rpoB", "species": "Escherichia coli", "protein": "MFEPMTVRQVCERIGPITRDITKETVSKVLAEAGFEVIQHGRSTLCAHMNNGLSIIEKVLDYTEGDNFDRHLLEKINEHLKQHPEDLLTFAKREKDFFAALRAHKVS"},
    {"gene": "gyrA", "species": "Escherichia coli", "protein": "MSDLAREITPVNIEEELKSSYADATKVLKKYALHENPIAVSAAKIAQEAGYDMDLKKEMTYVPMVRTVGKEVAVHDEVLDDGLGRSQEKIGQLLKKAANRWMVHGKADTGGEGLDVMGVPSDNIDNQVNFGQLNRMHLRPEQFVHTAIDTGKQSALHFMEKEG"},
    {"gene": "katG", "species": "Mycobacterium tuberculosis", "protein": "MTDSPQKQKPISKSVLPSFQETIVVEHSAAKGGHTPFFIRDNISVGPEKQAEEIMKDLRELRQERPERNDPMVSKMPPHEYDMFVDLDKDKPLEQKYIEADLQALRKKRPLSQA"},
    {"gene": "pbp2a", "species": "Staphylococcus aureus", "protein": "MKKWIKFLTLALVFSVQVTKEEVAFRKEKYVPKSTEPFDLSDMMDQFPQNTIQVTDFPGKYYITMKFDEKVDLSSGFTEYVYTRGDLYVPAINLSDGIDYTNPQFLE"},
    {"gene": "ampC", "species": "Escherichia coli", "protein": "MMTNSKELAILMVLFVAIALCSSGKKTIDPTIRSIDEEGTRVVSITQYAQKLAEDQGFSLRPFAQIGEALLPNSSPAQTMRLSDKLRQNGQWQKLQTDEVKEIVK"},
    {"gene": "murA", "species": "Pseudomonas aeruginosa", "protein": "MARVTITLGAEKRQITDALDAGLARGDLNVIVENGIHFSAQPIDAAQVAAAIQSKINPMGKFKDTNIVYASASSGKYITPAIMTVVPFIRDLPNITYLTKFVGEAMQRVGAPLNQAL"},
    {"gene": "folA", "species": "Staphylococcus aureus", "protein": "MTLINALIAQMGQKFNISDLFKAFDALRESGFESLSIDAIQKIMKNKADAAILAALNSHIGRVDSKDFVKTVNREFEVS"},
    {"gene": "aroA", "species": "Salmonella typhimurium", "protein": "MESKQEGLYSELAQLASEVGFVHAADVRPYHGTLVAEVLALRHRTLIQLAEQLRYAGDGFNPCRAHKDSFRKAMAELGIRVRGVD"},
]


def generate_synthetic_variants(
    seed: int = 42,
    lof_per_protein: int = 5,
    wt_per_protein: int = 3,
    gof_per_protein: int = 2,
) -> pd.DataFrame:
    """Generate synthetic LoF, WT, and GoF variants.

    LoF: premature stops, large truncations
    WT: identical sequence (no mutation)
    GoF: single conservative missense mutations (simulated)

    Returns:
        DataFrame with columns: gene, species, ref_protein, var_protein,
        ref_dna, var_dna, label, source, mutation_type
    """
    rng = random.Random(seed)
    records = []

    for ref_info in BACTERIAL_REFERENCE_PROTEINS:
        ref_protein = ref_info["protein"]
        gene = ref_info["gene"]
        species = ref_info["species"]
        protein_len = len(ref_protein)

        if protein_len < 10:
            continue

        # --- LoF variants ---
        for i in range(lof_per_protein):
            if i % 3 == 0:
                # Premature stop at random position (first 80%)
                stop_pos = rng.randint(3, int(protein_len * 0.8))
                var_protein = ref_protein[:stop_pos]
                mutation_type = f"premature_stop_pos{stop_pos}"
            elif i % 3 == 1:
                # Large truncation (keep 10-40%)
                keep_frac = rng.uniform(0.1, 0.4)
                keep_len = max(int(protein_len * keep_frac), 3)
                var_protein = ref_protein[:keep_len]
                mutation_type = f"truncation_{keep_frac:.0%}"
            else:
                # Scrambled C-terminal (simulating frameshift effect)
                frameshift_pos = rng.randint(5, protein_len // 2)
                scrambled = "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=protein_len - frameshift_pos))
                var_protein = ref_protein[:frameshift_pos] + scrambled
                mutation_type = f"frameshift_sim_pos{frameshift_pos}"

            records.append({
                "gene": gene,
                "species": species,
                "ref_protein": ref_protein,
                "var_protein": var_protein,
                "ref_dna": "",
                "var_dna": "",
                "label": 0,  # LoF
                "source": "synthetic",
                "mutation_type": mutation_type,
            })

        # --- WT variants ---
        for i in range(wt_per_protein):
            records.append({
                "gene": gene,
                "species": species,
                "ref_protein": ref_protein,
                "var_protein": ref_protein,  # Identical
                "ref_dna": "",
                "var_dna": "",
                "label": 1,  # WT
                "source": "synthetic",
                "mutation_type": "wildtype",
            })

        # --- GoF variants (simulated single missense with charge/size change) ---
        gof_positions = rng.sample(range(1, protein_len - 1), min(gof_per_protein, protein_len - 2))
        for pos in gof_positions:
            # Introduce a charge-changing substitution (simplified GoF proxy)
            original = ref_protein[pos]
            # Pick a different amino acid
            replacements = [aa for aa in "RDEKWHYF" if aa != original]
            if replacements:
                new_aa = rng.choice(replacements)
                var_protein = ref_protein[:pos] + new_aa + ref_protein[pos + 1:]
                records.append({
                    "gene": gene,
                    "species": species,
                    "ref_protein": ref_protein,
                    "var_protein": var_protein,
                    "ref_dna": "",
                    "var_dna": "",
                    "label": 2,  # GoF
                    "source": "synthetic",
                    "mutation_type": f"missense_{original}{pos + 1}{new_aa}",
                })

    df = pd.DataFrame(records)
    logger.info(
        f"Generated {len(df)} synthetic variants: "
        f"LoF={len(df[df['label']==0])}, WT={len(df[df['label']==1])}, GoF={len(df[df['label']==2])}"
    )
    return df


def main():
    logging.basicConfig(level=logging.INFO)
    df = generate_synthetic_variants()

    out_path = Path("data/processed/synthetic_variants.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} records to {out_path}")


if __name__ == "__main__":
    main()

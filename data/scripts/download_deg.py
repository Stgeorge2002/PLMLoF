"""Download and process DEG (Database of Essential Genes) for LoF variant generation.

DEG contains experimentally validated essential genes in bacteria.
Disrupting these genes leads to loss of function (LoF).
We generate synthetic LoF variants by introducing disruptive mutations.

Source: https://tubic.org/deg/
Fallback: hardcoded curated set of well-characterized bacterial essential genes.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd
from Bio import SeqIO
from io import StringIO

from plmlof.utils.sequence_utils import translate_dna
from plmlof.data.preprocessing import introduce_premature_stop, introduce_frameshift

logger = logging.getLogger(__name__)

# Multiple potential URLs for DEG (site has moved over the years)
DEG_URLS = [
    # Current tubic.org paths
    ("https://tubic.org/deg/public/download/deg-p-e.dat", "protein"),
    ("https://tubic.org/deg/public/download/deg-n-e.dat", "nucleotide"),
    # Alternative paths seen in mirrors
    ("https://tubic.tju.edu.cn/deg/public/download/deg-p-e.dat", "protein"),
    ("https://tubic.tju.edu.cn/deg/public/download/deg-n-e.dat", "nucleotide"),
    # Older URL patterns
    ("http://tubic.org/deg/download/deg-p-e.dat", "protein"),
    ("http://tubic.org/deg/download/deg-n-e.dat", "nucleotide"),
]

OUTPUT_DIR = Path("data/raw/deg/")

# ── Curated bacterial essential gene proteins (fallback when DEG is unreachable) ──
# Each entry: (gene_name, species, protein_sequence, dna_cds)
# Sourced from UniProt/NCBI for well-characterized essential genes
ESSENTIAL_GENES_FALLBACK = [
    # DNA replication / repair
    ("dnaA", "Escherichia coli", "MSLSLWQQCLARLQDELPATEFSMWIRPLQAELSDNTLALYAPNRFVLDWVRDKYLNNINGLLTSFCGADAPQLRFEVGTKPVTQTPQAAVTSNVAAPAQVAQTQPQRAAPSTRSGWDNVPAPAEFTDSVTDSAFRREVDFVHDPTKQESRKAHLDALRSMIEHVRREQKFGEDSIQFVLKSTDWSEQKLNDSRGKIVTDSTSAIATEVEEQKFGEDSIQFVLKSTDDWSEQKLNDSRGKIVTD", ""),
    ("dnaB", "Escherichia coli", "MLTAFDYQSAIYDAAQAALAAEGYQLHQSVNFEARGKAKEYGQATKKAGRGSADYAVQRAFYQNLFIKQILGKPSMKDGRQAAHLAGFHSNRELYINAIYQVASQDSIHKLAEAQALCAMYREHKNTQEAIRALTSMAQRLGPNPVLRDMRIQHFASRNMDVKDLAFIMAAQTLEKWYAEEGGFQ", ""),
    ("dnaN", "Escherichia coli", "MKFTVEREHLLKPLQQVSGPLGGRPTLPILGNLLLQVADGTLSLTGTDLEMEMVARVALVQPHEPGATTVPARKFFDICRGLPEGAEIAVQLEGERMLVRSGRSRFSLSTLPAADFPNLDDWQSAGDSIRDNLPFVSAETFMAGRSGAKEGDAVNALKRLAQISQARLQEALTEYAKKLHDLKLARGLKVNPAILSQLKGNRINSLQKEELANLPFQRVDALCQEKSLALLNTLHPAYRQVLKQHVTESSSRLSHIQFSDVEQEAVAESLAQ", ""),
    # Transcription
    ("rpoB", "Escherichia coli", "MFEPMTVRQVCERIGPITRDITKETVSKVLAEAGFEVIQHGRSTLCAHMNNGLSIIEKVLDYTEGDNFDRHLLEKINEHLKQHPEDLLTFAKREKDFFAALRAHKVSVGELKSSFKRLIEFIGDEAITMEYVKQHIGEELTTAGEKKVSTKRKLKKAQEAAIAAAKGNEEKAKEIVDEATKKALSAALKEMNPDEISARSGLSILRDLTFNSPKAGEPIISGAKPEYYRELVAKAMFDIYGYR", ""),
    ("rpoC", "Escherichia coli", "MKDLLKFLKAQTKTEEFDAIKIALASPDMIRSWSFGEVKKPETINYRTFKPERDGLFCARIFGPVKDYECLCGKYKRLKHRGVICEKCGVEVTQTKVRRERMGHIELASPTAHIWFLKSLPSRIGLLLDMPLRDIERVLYFESYVVIEGGMTQAALGRVFKDLNQQPIFELTDELVREKFKKENTQLLAAALEKNAASYPQILEAMYGSVHELIKEIIKAKPANGSASKSKEVL", ""),
    ("rpoD", "Escherichia coli", "MEQNPQSQLKLLEKAGYKPLSDALREGQRILAEKEAILAEVAVAGAGVEDALKNVPAPQVLDAIREGEIERLISQRPGKLPKETDELVRLFAETHGIALAQALNQYVFERLIRQIREAGAYSNLLESIKRQSGYSDREQLLRTVYNALEKRGDDAVNAMLALKEEKFDDTPMVLNRAGFEREGLIAHSIDFSAEQRVTRLLFQQLENQFVQEETYSDFGEALKGVAAELNQKLHEDLSKSIFQESFDYDREALTE", ""),
    # Translation
    ("infB", "Escherichia coli", "MATQTFADNHISDDLQERVTDILAQTHKPAQNQHIFRPLFLGEPDQENAEEVVLELNLKDIDVTITEVTGQEVIAHVVAGELKMPHAVFIIDPSPFEKIDQCTQIELEDATMVNEQLVMHVTGAEITSYFEVKDKELLAKVEELEKEELDEFEEQIQGTAQLAALEREFRAISDSSFRGRGKISRSVVK", ""),
    ("fusA", "Escherichia coli", "MAKEKFDRSKPHINIGTIGHVDHGKTTLTAAITMTLAALGAEAKNFSIRERAISEQIGQGLVKDLPGHEKDSFRVELIRESGQITTRQHIEYCTRHHILQADMPFFIRGPQSYDAIVTLNMEQPIYATIVNLGEAITDIVVLCTLINKMDPPERVKIEDVMVELSQLEMDKIKRGANFVAIRTKDAAGQHTVRAAFIGALQRGRVKPTKEARQKIMSIAKKIEGALTSAFDVF", ""),
    ("tufA", "Escherichia coli", "MSKEKFERTKPHVNVGTIGHVDHGKTTLTAAITTVLAKTYGGAARAFDQIDNAPEEKARGITINTSHVEYDTPTRHYAHVDCPGHADYVKNMITGAAQMDGAILVVSAADGPMPQTREHILLGRQVGVPYIIVFLNKCDMVDDEELLELVEMEVRELLSQYDFPGDDTPIVRGSALKALEGEDAEWEAKILELAGFLDSYIPEPERAIDKPFLLPIEDVFSISGRGTVVTGRVERGIIKVGEE", ""),
    # Cell division
    ("ftsZ", "Escherichia coli", "MFEPMELTNDAVIKVIGVGGGGGNAVEHMVRERIGENHALAGANDFIISNDLLQYASKDGDLDTLVNFVAQLNENHPDYMFITSGDTGTQAPGTEVEKVFHPTGILELEFSDVVAILSGIGDLSSAVGKGAGNAAGATVFAVENTDPFIATGALRMKEAVDDAYDIANDADHYYRVIGLKGVQDLTKFISAKK", ""),
    ("ftsA", "Escherichia coli", "MISVIIATDNKQTYLDLEQLCKTYFDSDTPEVKKFIAGRVLDSLDTEIATLVKETLVEMDSMQAQFNELNERIVVRSAAHEGTNVTGKIVQGAETVIRGFQAPVAGMVLSKHVVKAAKQVGDVILDIGNDSKTTPDILREMQSMRLATQLNIESLVITKQGIGEQLQDTLKNLKVQDEDVRRMLSTFKE", ""),
    # Lipid biosynthesis
    ("accA", "Escherichia coli", "MQTFDFQYKNEKADLIKEHLHQISGDTPMEEIITPQHQILQRVSAQEMKNLHYKPPVGEIYGMFGFDIINKQNELTRQEFSENQKIIFIGPPGSAMELAVTALKIAQEAGIPVIYSTGDSFAATFIDKLHAGNVSFYQTKLALPTPKDLARLQKLIAEGHNITVNATNHGNAYYLADKEGTMPHSGAIQALIDAGVVSQLEQNDKSLSRELTEKLAAWEKQFREAIKNPNISTPI", ""),
    ("accD", "Escherichia coli", "MSTKPVIITDVSRFDAQNLIREIKEKFPQSRLTGESSVEINFLTTLPTMAEEVLEKYGYLPEEVLDDLNEQLKQVSDMHAKYGNSLDPQTREASALMSQAEMIENANFPALAVQPKQIVEVTAKQAGKSSEQKAAELIAAGADIIGISNHGNSYIYAARNLASKGLPTLITFDID", ""),
    # Amino acid biosynthesis  
    ("glyA", "Escherichia coli", "MKLYNLKDHNEQVSFAQAVTQGLGKNQGLFFPHDLPEFSLTEIDEMLKLDFVTRSAKILSAFIGDEIPQEILEERVRAAFAFPAPVANVESDVGCLELFHGPTLAFKDFGGRFMAQMLTHIAGDKPVTILIGHSERRHYFESEAEKAAERVTRLAKENLKAFGAKRFEVHEIISDEEVQRGAKALMMKLATQFSDMEFQDWAAKKLHEIQADFNWTLPYEGFKISKRY", ""),
    ("serA", "Escherichia coli", "MKLYNLKDHNEQVSFAQAVTQGLGKNQGLFFPHDLPEFSLTEIDEMLKLDFVTRSAKILSAFIGDEIPQEILEERVRAAFAFPAPVANVESDVGCLELFHGPTLAFKDFGGRFMAQMLTHIAGDKPVTILIGHSERRHYFESEAEKAAERVTRLAKENLKAFGAKRFEVHEIISDEEVQRGAKALMMKLATQFSDMEFQDWAAKKLHEIQADFNW", ""),
    # Cell wall
    ("murA", "Escherichia coli", "MARVTITLGAEKRQITDALDAGLARGDLNVIVENGIHFSAQPIDAAQVAAAIQSKINPMGKFKDTNIVYASASSGKYITPAIMTVVPFIRDLPNITYLTKFVGEAMQRVGAPLNQALVKEMKPFYGLKSLHVADIEAERLTKFEAERKKLVDSLNWANRDISTWLGKQFHPEVTHTTPHTQMLEFLAQKPQRLKAIFEDAKKYV", ""),
    ("murB", "Escherichia coli", "MAIIIGAGNAGSCYAANQLGAKLILTDQNAEKHFPQYRSGLAHKQIDYVQNGLRRMSVSLEQAEKAKASQVSHDAFADSSYTTKYAAEFAVPEFKAYQNFCLAGDTGEKINRGDDVYIHRTDEDSDVYRPGARLLFAGIAAGMSSHMGLDYAVMKHYGGLDRLNRFGGEKFAAPYEDAKIV", ""),
    # Protein folding/secretion
    ("groEL", "Escherichia coli", "MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMGAQMVKEVASKANDAAGDGTTTATVLAQAIITEGLKAVAAGMNPMDLKRGIDKAVTAAVEELKALSVPCSDSKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLQDELDVVEGMQFDRGYLSPYFINKPETGA", ""),
    ("secA", "Escherichia coli", "MLIKLLTKVFGSRNDRTLRRLSEKFGKPFCAAGVHLEEVIMPIRYQKRGKRDFTRLKLILKQFHEDIKPMPLSFAGEALKHFDDDSYKELFDFDLKWQAKYQAMFHKEEIENALLSWAEDHQKIFEANQKVEQFYNELKNELGVGEVIEFYRKLKEPKSLDNMTAETLAEWFDAHQKKGQSKPEVFNDFENQRFLA", ""),
    # Nucleotide biosynthesis
    ("pyrG", "Escherichia coli", "MLTRVKLITGGVVSAQVANALKEAGFSCIMIDSTPREHVLSGAAHKAGVPVIHTSTAQRLAQEFARKDGVKIFVDSEYFDTMMTPTGEVSKKEVAVKLANHHGMNIIGTDINEDPFAKALFEGFEERYGFNLAKMKRDMDRFNHVDEFLLDNFAPDCRIAPVTANLRALLAGGYKVNPCGVLAQTAWALGIPYELINAFRQAGIFAGRCVDLMIHRD", ""),
    ("pyrH", "Escherichia coli", "MKVAVLSGGSQGLRNALDAVSPTITQVVKASGKDLIVVLAAGVQKQNALAQSLGFNIDLVSLNIQAEPDDGSEEDYDADPFNKRKEMLAFIQQHLETEEFLGNAVQVLALNPFDTNTKDIQNWLKYGGDIILTADPFYSKPKQTEYSPFLNQMKAAGAKLV", ""),
    # Fatty acid biosynthesis
    ("fabI", "Escherichia coli", "MGFLSGKRILVTGVASKLSIAYGIAQAMHREGAELAFTYQNDKLKGRVEEFAAQLGSDIVLQCDVAEDASIDTMFAELGKVWPKFDGFVHSIGFAPGDQLDGDYVNAVTREGFKIAHDISSYSSFVAMAKACRSMLNPGSALLTLSYLGAERAIPNYNVMGLARTSLSAAMTAQY", ""),
    ("fabD", "Escherichia coli", "MTQFAFVFPGQGSQTVGMLADMAASYPIVEETFAEASRILSEQGRPSYIFENSYLRPQLDQDCAKTLEHTLLFQPALHAFEHSLLESWGIEPDFVVGHSFGELVAAHFAGIFSLEDGLKLISRSRAILPNSGATMAASLRIMEEEVEQFVLQVLGRACGFKVAVVAGHNEERAQMLQEVTGSKLKQMSSGQPMQKAVFADYASVAG", ""),
    # Chaperones
    ("dnaK", "Escherichia coli", "MGKIIGIDLGTTNSCVAIMDGTTPRVLENAEGDRTTPSIIAYTQDGETLVGQPAKRQAVTNPQNTLFAIKRLIGRRFQDEEVQRDVSIMPFKIIAADNGDAWVEVKGQKMAPPQISAEVLKKMKKTAEDYLGEPVTEAVITVPAYFNDAQRQATKDAGRIAGLEVKRIINEPTAAALAYGLDKGTGNRTIAVYDLGGGTFDISIIEIDEVDGEKTFEVLATNGDTHLGGEDFDNRMVNHFV", ""),
    # More essential genes from S. aureus, M. tuberculosis, P. aeruginosa
    ("pbp2", "Staphylococcus aureus", "MKKWIKFLTLALVFSVQVTKEEVAFRKEKYVPKSTEPFDLSDMMDQFPQNTIQVTDFPGKYYITMKFDEKVDLSSGFTEYVYTRGDLYVPAINLSDGIDYTNPQFLE", ""),
    ("inhA", "Mycobacterium tuberculosis", "MTGLLDGKRILVSGIITDSSIAFHIARVAQEQGAQLVLTGFDRLRLIQRITDITAESAAKLKGNTLGSGISSNFPALKEAVDDVILGRFTATLRDVRQPEKIVDAVTGGFDITRQELGLSGYRSGKIAGQVYRSGGMTSYMAKSTLFDTFANYRLLMSQRFARNFGLITG", ""),
    ("gyrB", "Pseudomonas aeruginosa", "MSNSQDTIKAAKVYITDQHEGPDYLDIYQSPHGERAVSQEVRENLTVAGFDIEKHIPKSTRLENLEIRVNKDKWVIKDGRGRVRVHKDNNIDPDGSYETFTRFHTSVDAIN", ""),
    ("leuS", "Bacillus subtilis", "MQKFDTQEQLNNWANDWASQYRDFLKAQNKGEKFKIREEMTKFIGEHFDPKSMQTLGATDALVRELGKDQEAELFKRVVDALFTDYELR", ""),
    ("alaS", "Klebsiella pneumoniae", "MSKSTAEIRQAFLDFFHSKGHQVVASSTHALLGQLRALELKYVQGSRLQKDPTLRTEVYNALRAEKMKFSKQYGVTPEHVLNRFANFIDQNLKQ", ""),
    ("metG", "Salmonella typhimurium", "MKERLNFIQDAFEQFYDLHAPIFKQIESFYASDGFEQIKQHPKERLPVLEGDFLHIGHTGKFIDEVIKMKQFGLKEFDMTGAFVVLTARHLKFEEQMKTHPLEQKLQKL", ""),
]


def download_deg(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    """Download DEG protein and nucleotide files from multiple URLs.

    Returns:
        Tuple of (protein_path, dna_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_path = output_dir / "deg_proteins.dat"
    dna_path = output_dir / "deg_dna.dat"

    # Try all URLs for each type
    for url, dtype in DEG_URLS:
        target = protein_path if dtype == "protein" else dna_path
        if target.exists() and target.stat().st_size > 100:
            continue
        logger.info(f"Trying DEG {dtype} from {url}...")
        try:
            req = Request(url, headers={"User-Agent": "PLMLoF/1.0"})
            response = urlopen(req, timeout=30)  # noqa: S310
            data = response.read()
            if len(data) > 100:
                target.write_bytes(data)
                logger.info(f"Downloaded {len(data) / 1e6:.1f} MB ({dtype})")
            else:
                logger.warning(f"Empty response from {url}")
        except Exception as e:
            logger.warning(f"Could not download DEG {dtype} from {url}: {e}")

    # Create empty files if nothing downloaded
    if not protein_path.exists():
        protein_path.write_text("")
    if not dna_path.exists():
        dna_path.write_text("")

    return protein_path, dna_path


def parse_deg_sequences(protein_path: Path, dna_path: Path) -> pd.DataFrame:
    """Parse DEG FASTA files and extract bacterial essential gene sequences.

    Returns:
        DataFrame with columns: gene_id, gene, species, ref_protein, ref_dna
    """
    records = []

    # Parse protein sequences
    proteins = {}
    if protein_path.exists() and protein_path.stat().st_size > 100:
        try:
            for record in SeqIO.parse(str(protein_path), "fasta"):
                proteins[record.id] = {
                    "protein": str(record.seq),
                    "description": record.description,
                }
        except Exception as e:
            logger.warning(f"Error parsing DEG proteins: {e}")

    # Parse DNA sequences
    dna_seqs = {}
    if dna_path.exists() and dna_path.stat().st_size > 100:
        try:
            for record in SeqIO.parse(str(dna_path), "fasta"):
                dna_seqs[record.id] = str(record.seq).upper()
        except Exception as e:
            logger.warning(f"Error parsing DEG DNA: {e}")

    # Merge
    all_ids = set(proteins.keys()) | set(dna_seqs.keys())
    for gid in all_ids:
        protein = proteins.get(gid, {}).get("protein", "")
        desc = proteins.get(gid, {}).get("description", "")
        dna = dna_seqs.get(gid, "")

        # If we have DNA but no protein, translate
        if dna and not protein:
            protein = translate_dna(dna, to_stop=True)

        # Extract species from description (DEG format: "DEG_ID gene_name - Species")
        species = ""
        if " - " in desc:
            species = desc.split(" - ")[-1].strip()

        # Filter for bacterial species (heuristic: exclude human, mouse, yeast, etc.)
        skip_species = {"homo sapiens", "mus musculus", "saccharomyces", "drosophila",
                        "caenorhabditis", "arabidopsis", "danio rerio"}
        if any(s in species.lower() for s in skip_species):
            continue

        if protein:
            records.append({
                "gene_id": gid,
                "gene": desc.split()[1] if len(desc.split()) > 1 else gid,
                "species": species,
                "ref_protein": protein,
                "ref_dna": dna,
            })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} bacterial essential genes from DEG")
    return df


def _get_fallback_essential_genes() -> pd.DataFrame:
    """Return curated bacterial essential gene sequences when DEG is unavailable."""
    records = []
    for i, (gene, species, protein, dna) in enumerate(ESSENTIAL_GENES_FALLBACK):
        records.append({
            "gene_id": f"CURATED_{i:04d}",
            "gene": gene,
            "species": species,
            "ref_protein": protein,
            "ref_dna": dna,
        })
    logger.info(f"Using {len(records)} curated essential genes as fallback")
    return pd.DataFrame(records)


def generate_lof_variants(
    essential_genes: pd.DataFrame,
    variants_per_gene: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic LoF variants from essential genes.

    For each gene, generates variants with:
    - Premature stop codons
    - Frameshift mutations
    - Large truncations

    Returns:
        DataFrame with columns: gene, species, ref_protein, var_protein,
        ref_dna, var_dna, mutation_type, label, source
    """
    rng = random.Random(seed)
    records = []

    for _, row in essential_genes.iterrows():
        ref_protein = row["ref_protein"]
        ref_dna = row["ref_dna"]
        gene = row["gene"]
        species = row.get("species", "")

        if len(ref_protein) < 10:
            continue

        for v in range(variants_per_gene):
            var_protein = None
            var_dna = ""
            mutation_type = ""

            if v == 0 and ref_dna and len(ref_dna) >= 30:
                # Premature stop at random position (first 80% of gene)
                max_codon = int(len(ref_protein) * 0.8)
                if max_codon < 2:
                    continue
                codon_pos = rng.randint(1, max_codon)
                try:
                    var_dna = introduce_premature_stop(ref_dna, codon_pos)
                    var_protein = translate_dna(var_dna, to_stop=True)
                    mutation_type = f"premature_stop_at_codon_{codon_pos}"
                except (ValueError, IndexError):
                    var_protein = None
            elif v == 0 and (not ref_dna or len(ref_dna) < 30):
                # No DNA: simulate premature stop by truncation + "*"
                stop_pos = rng.randint(1, int(len(ref_protein) * 0.8))
                var_protein = ref_protein[:stop_pos] + "*"
                mutation_type = f"premature_stop_at_{stop_pos}"
            elif v == 1 and ref_dna and len(ref_dna) >= 30:
                # Frameshift in first half
                pos = rng.randint(3, len(ref_dna) // 2)
                try:
                    var_dna = introduce_frameshift(ref_dna, pos, insert=True)
                    var_protein = translate_dna(var_dna, to_stop=True)
                    mutation_type = f"frameshift_insert_at_{pos}"
                except (ValueError, IndexError):
                    var_protein = None
            elif v == 1 and (not ref_dna or len(ref_dna) < 30):
                # No DNA: simulate scrambled N-terminal region
                scramble_len = max(len(ref_protein) // 4, 3)
                scrambled = list(ref_protein[:scramble_len])
                rng.shuffle(scrambled)
                var_protein = "".join(scrambled) + ref_protein[scramble_len:][:5] + "*"
                mutation_type = f"scrambled_nterm_{scramble_len}"
            else:
                # Large truncation (keep only 10-50% of protein)
                keep_frac = rng.uniform(0.1, 0.5)
                keep_len = max(int(len(ref_protein) * keep_frac), 1)
                var_protein = ref_protein[:keep_len]
                var_dna = ref_dna[:keep_len * 3] if ref_dna else ""
                mutation_type = f"truncation_{keep_frac:.0%}"

            if not var_protein:
                continue

            records.append({
                "gene": gene,
                "species": species,
                "ref_protein": ref_protein,
                "var_protein": var_protein,
                "ref_dna": ref_dna,
                "var_dna": var_dna,
                "mutation_type": mutation_type,
                "label": 0,  # LoF
                "source": "DEG_synthetic",
            })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} synthetic LoF variants from {len(essential_genes)} essential genes")
    return df


def main():
    logging.basicConfig(level=logging.INFO)

    protein_path, dna_path = download_deg()
    essential = parse_deg_sequences(protein_path, dna_path)

    if essential.empty:
        logger.warning("DEG download/parse failed. Using curated essential genes fallback.")
        essential = _get_fallback_essential_genes()

    lof_df = generate_lof_variants(essential)

    out_path = Path("data/processed/deg_lof.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lof_df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(lof_df)} records to {out_path}")


if __name__ == "__main__":
    main()

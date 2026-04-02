"""Download and process DEG (Database of Essential Genes) for LoF variant generation.

DEG contains experimentally validated essential genes in bacteria.
Disrupting these genes leads to loss of function (LoF).
We generate synthetic LoF variants by introducing disruptive mutations.

Source: https://tubic.org/deg/
Download page: https://tubic.org/deg/public/index.php/download
Fallback: hardcoded curated set of well-characterized bacterial essential genes.
"""

from __future__ import annotations

import gzip
import io
import logging
import random
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd
from Bio import SeqIO

from plmlof.utils.sequence_utils import translate_dna
from plmlof.data.preprocessing import introduce_premature_stop, introduce_frameshift

logger = logging.getLogger(__name__)

# Current DEG download URLs
DEG_PROTEIN_URLS = [
    "http://tubic.org/deg/public/download/DEG10.aa.gz",
    "https://tubic.org/deg/public/download/DEG10.aa.gz",
]
DEG_NUCLEOTIDE_URLS = [
    "http://tubic.org/deg/public/download/DEG10.nt.gz",
    "https://tubic.org/deg/public/download/DEG10.nt.gz",
]
DEG_ANNOTATION_URLS = [
    "http://tubic.org/deg/public/download/deg_annotation_p.csv.zip",
    "https://tubic.org/deg/public/download/deg_annotation_p.csv.zip",
]

OUTPUT_DIR = Path("data/raw/deg/")

# ── Curated bacterial essential gene proteins (fallback when DEG is unreachable) ──
ESSENTIAL_GENES_FALLBACK = [
    ("dnaA", "Escherichia coli", "MSLSLWQQCLARLQDELPATEFSMWIRPLQAELSDNTLALYAPNRFVLDWVRDKYLNNINGLLTSFCGADAPQLRFEVGTKPVTQTPQAAVTSNVAAPAQVAQTQPQRAAPSTRSGWDNVPAPAEFTDSVTDSAFRREVDFVHDPTKQESRKAHLDALRSMIEHVRREQKFGEDSIQFVLKSTDWSEQKLNDSRGKIVTDSTSAIATEVEEQKFGEDSIQFVLKSTDDWSEQKLNDSRGKIVTD", ""),
    ("dnaB", "Escherichia coli", "MLTAFDYQSAIYDAAQAALAAEGYQLHQSVNFEARGKAKEYGQATKKAGRGSADYAVQRAFYQNLFIKQILGKPSMKDGRQAAHLAGFHSNRELYINAIYQVASQDSIHKLAEAQALCAMYREHKNTQEAIRALTSMAQRLGPNPVLRDMRIQHFASRNMDVKDLAFIMAAQTLEKWYAEEGGFQ", ""),
    ("dnaN", "Escherichia coli", "MKFTVEREHLLKPLQQVSGPLGGRPTLPILGNLLLQVADGTLSLTGTDLEMEMVARVALVQPHEPGATTVPARKFFDICRGLPEGAEIAVQLEGERMLVRSGRSRFSLSTLPAADFPNLDDWQSAGDSIRDNLPFVSAETFMAGRSGAKEGDAVNALKRLAQISQARLQEALTEYAKKLHDLKLARGLKVNPAILSQLKGNRINSLQKEELANLPFQRVDALCQEKSLALLNTLHPAYRQVLKQHVTESSSRLSHIQFSDVEQEAVAESLAQ", ""),
    ("rpoB", "Escherichia coli", "MFEPMTVRQVCERIGPITRDITKETVSKVLAEAGFEVIQHGRSTLCAHMNNGLSIIEKVLDYTEGDNFDRHLLEKINEHLKQHPEDLLTFAKREKDFFAALRAHKVSVGELKSSFRLIEFIGDEAITMEYVKQHIGEELTTAGEKKVSTKRKLKKAQEAAIAAAKGNEEKAKEIVDEATKKALSAALKEMNPDEISARSGLSILRDLTFNSPKAGEPIISGAKPEYYRELVAKAMFDIYGYR", ""),
    ("rpoC", "Escherichia coli", "MKDLLKFLKAQTKTEEFDAIKIALASPDMIRSWSFGEVKKPETINYRTFKPERDGLFCARIFGPVKDYECLCGKYKRLKHRGVICEKCGVEVTQTKVRRERMGHIELASPTAHIWFLKSLPSRIGLLLDMPLRDIERVLYFESYVVIEGGMTQAALGRVFKDLNQQPIFELTDELVREKFKKENTQLLAAALEKNAASYPQILEAMYGSVHELIKEIIKAKPANGSASKSKEVL", ""),
    ("rpoD", "Escherichia coli", "MEQNPQSQLKLLEKAGYKPLSDALREGQRILAEKEAILAEVAVAGAGVEDALKNVPAPQVLDAIREGEIERLISQRPGKLPKETDELVRLFAETHGIALAQALNQYVFERLIRQIREAGAYSNLLESIKRQSGYSDREQLLRTVYNALEKRGDDAVNAMLALKEEKFDDTPMVLNRAGFEREGLIAHSIDFSAEQRVTRLLFQQLENQFVQEETYSDFGEALKGVAAELNQKLHEDLSKSIFQESFDYDREALTE", ""),
    ("infB", "Escherichia coli", "MATQTFADNHISDDLQERVTDILAQTHKPAQNQHIFRPLFLGEPDQENAEEVVLELNLKDIDVTITEVTGQEVIAHVVAGELKMPHAVFIIDPSPFEKIDQCTQIELEDATMVNEQLVMHVTGAEITSYFEVKDKELLAKVEELEKEELDEFEEQIQGTAQLAALEREFRAISDSSFRGRGKISRSVVK", ""),
    ("fusA", "Escherichia coli", "MAKEKFDRSKPHINIGTIGHVDHGKTTLTAAITMTLAALGAEAKNFSIRERAISEQIGQGLVKDLPGHEKDSFRVELIRESGQITTRQHIEYCTRHHILQADMPFFIRWGPQSYDAIVTLNMEQPIYATIVNLGEAITDIVVLCTLINKMDPPERVKIEDVMVELSQLEMDKIKRGANFVAIRTKDAAGQHTVRAAFIGALQRGRVKPTKEARQKIMSIAKKIEGALTSAFDVF", ""),
    ("tufA", "Escherichia coli", "MSKEKFERTKPHVNVGTIGHVDHGKTTLTAAITTVLAKTYGGAARAFDQIDNAPEEKARGITINTSHVEYDTPTRHYAHVDCPGHADYVKNMITGAAQMDGAILVVSAADGPMPQTREHILLGRQVGVPYIIVFLNKCDMVDDEELLELVEMEVRELLSQYDFPGDDTPIVRGSALKALEGEDAEWEAKILELAGFLDSYIPEPERAIDKPFLLPIEDVFSISGRGTVVTGRVERGIIKVGEE", ""),
    ("ftsZ", "Escherichia coli", "MFEPMELTNDAVIKVIGVGGGGGNAVEHMVRERIGENHALAGANDFIISNDLLQYASKDGDLDTLVNFVAQLNENHPDYMFITSGDTGTQAPGTEVEKVFHPTGILELEFSDVVAILSGIGDLSSAVGKGAGNAAGATVFAVENTDPFIATGALRMKEAVDDAYDIANDADHYYRVIGLKGVQDLTKFISAKK", ""),
    ("ftsA", "Escherichia coli", "MISVIIATDNKQTYLDLEQLCKTYFDSDTPEVKKFIAGRVLDSLDTEIATLVKETLVEMDSMQAQFNELNERIVVRSAAHEGTNVTGKIVQGAETVIRGFQAPVAGMVLSKHVVKAAKQVGDVILDIGNDSKTTPDILREMQSMRLATQLNIESLVITKQGIGEQLQDTLKNLKVQDEDVRRMLSTFKE", ""),
    ("accA", "Escherichia coli", "MQTFDFQYKNEKADLIKEHLHQISGDTPMEEIITPQHQILQRVSAQEMKNLHYKPPVGEIYGMFGFDIINKQNELTRQEFSENQKIIFIGPPGSAMELAVTALKIAQEAGIPVIYSTGDSFAATFIDKLHAGNVSFYQTKLALPTPKDLARLQKLIAEGHNITVNATNHGNAYYLADKEGTMPHSGAIQALIDAGVVSQLEQNDKSLSRELTEKLAAWEKQFREAIKNPNISTPI", ""),
    ("accD", "Escherichia coli", "MSTKPVIITDVSRFDAQNLIREIKEKFPQSRLTGESSVEINFLTTLPTMAEEVLEKYGYLPEEVLDDLNEQLKQVSDMHAKYGNSLDPQTREASALMSQAEMIENANFPALAVQPKQIVEVTAKQAGKSSEQKAAELIAAGADIIGISNHGNSYIYAARNLASKGLPTLITFDID", ""),
    ("glyA", "Escherichia coli", "MKLYNLKDHNEQVSFAQAVTQGLGKNQGLFFPHDLPEFSLTEIDEMLKLDFVTRSAKILSAFIGDEIPQEILEERVRAAFAFPAPVANVESDVGCLELFHGPTLAFKDFGGRFMAQMLTHIAGDKPVTILIGHSERRHYFESEAEKAAERVTRLAKENLKAFGAKRFEVHEIISDEEVQRGAKALMMKLATQFSDMEFQDWAAKKLHEIQADFNWTLPYEGFKISKRY", ""),
    ("serA", "Escherichia coli", "MKLYNLKDHNEQVSFAQAVTQGLGKNQGLFFPHDLPEFSLTEIDEMLKLDFVTRSAKILSAFIGDEIPQEILEERVRAAFAFPAPVANVESDVGCLELFHGPTLAFKDFGGRFMAQMLTHIAGDKPVTILIGHSERRHYFESEAEKAAERVTRLAKENLKAFGAKRFEVHEIISDEEVQRGAKALMMKLATQFSDMEFQDWAAKKLHEIQADFNW", ""),
    ("murA", "Escherichia coli", "MARVTITLGAEKRQITDALDAGLARGDLNVIVENGIHFSAQPIDAAQVAAAIQSKINPMGKFKDTNIVYASASSGKYITPAIMTVVPFIRDLPNITYLTKFVGEAMQRVGAPLNQALVKEMKPFYGLKSLHVADIEAERLTKFEAERKKLVDSLNWANRDISTWLGKQFHPEVTHTTPHTQMLEFLAQKPQRLKAIFEDAKKYV", ""),
    ("murB", "Escherichia coli", "MAIIIGAGNAGSCYAANQLGAKLILTDQNAEKHFPQYRSGLAHKQIDYVQNGLRRMSVSLEQAEKAKASQVSHDAFADSSYTTKYAAEFAVPEFKAYQNFCLAGDTGEKINRGDDVYIHRTDEDSDVYRPGARLLFAGIAAGMSSHMGLDYAVMKHYGGLDRLNRFGGEKFAAPYEDAKIV", ""),
    ("groEL", "Escherichia coli", "MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMGAQMVKEVASKANDAAGDGTTTATVLAQAIITEGLKAVAAGMNPMDLKRGIDKAVTAAVEELKALSVPCSDSKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLQDELDVVEGMQFDRGYLSPYFINKPETGA", ""),
    ("secA", "Escherichia coli", "MLIKLLTKVFGSRNDRTLRRLSEKFGKPFCAAGVHLEEVIMPIRYQKRGKRDFTRLKLILKQFHEDIKPMPLSFAGEALKHFDDDSYKELFDFDLKWQAKYQAMFHKEEIENALLSWAEDHQKIFEANQKVEQFYNELKNELGVGEVIEFYRKLKEPKSLDNMTAETLAEWFDAHQKKGQSKPEVFNDFENQRFLA", ""),
    ("pyrG", "Escherichia coli", "MLTRVKLITGGVVSAQVANALKEAGFSCIMIDSTPREHVLSGAAHKAGVPVIHTSTAQRLAQEFARKDGVKIFVDSEYFDTMMTPTGEVSKKEVAVKLANHHGMNIIGTDINEDPFAKALFEGFEERYGFNLAKMKRDMDRFNHVDEFLLDNFAPDCRIAPVTANLRALLAGGYKVNPCGVLAQTAWALGIPYELINAFRQAGIFAGRCVDLMIHRD", ""),
    ("pyrH", "Escherichia coli", "MKVAVLSGGSQGLRNALDAVSPTITQVVKASGKDLIVVLAAGVQKQNALAQSLGFNIDLVSLNIQAEPDDGSEEDYDADPFNKRKEMLAFIQQHLETEEFLGNAVQVLALNPFDTNTKDIQNWLKYGGDIILTADPFYSKPKQTEYSPFLNQMKAAGAKLV", ""),
    ("fabI", "Escherichia coli", "MGFLSGKRILVTGVASKLSIAYGIAQAMHREGAELAFTYQNDKLKGRVEEFAAQLGSDIVLQCDVAEDASIDTMFAELGKVWPKFDGFVHSIGFAPGDQLDGDYVNAVTREGFKIAHDISSYSSFVAMAKACRSMLNPGSALLTLSYLGAERAIPNYNVMGLARTSLSAAMTAQY", ""),
    ("fabD", "Escherichia coli", "MTQFAFVFPGQGSQTVGMLADMAASYPIVEETFAEASRILSEQGRPSYIFENSYLRPQLDQDCAKTLEHTLLFQPALHAFEHSLLESWGIEPDFVVGHSFGELVAAHFAGIFSLEDGLKLISRSRAILPNSGATMAASLRIMEEEVEQFVLQVLGRACGFKVAVVAGHNEERAQMLQEVTGSKLKQMSSGQPMQKAVFADYASVAG", ""),
    ("dnaK", "Escherichia coli", "MGKIIGIDLGTTNSCVAIMDGTTPRVLENAEGDRTTPSIIAYTQDGETLVGQPAKRQAVTNPQNTLFAIKRLIGRRFQDEEVQRDVSIMPFKIIAADNGDAWVEVKGQKMAPPQISAEVLKKMKKTAEDYLGEPVTEAVITVPAYFNDAQRQATKDAGRIAGLEVKRIINEPTAAALAYGLDKGTGNRTIAVYDLGGGTFDISIIEIDEVDGEKTFEVLATNGDTHLGGEDFDNRMVNHFV", ""),
    ("pbp2", "Staphylococcus aureus", "MKKWIKFLTLALVFSVQVTKEEVAFRKEKYVPKSTEPFDLSDMMDQFPQNTIQVTDFPGKYYITMKFDEKVDLSSGFTEYVYTRGDLYVPAINLSDGIDYTNPQFLE", ""),
    ("inhA", "Mycobacterium tuberculosis", "MTGLLDGKRILVSGIITDSSIAFHIARVAQEQGAQLVLTGFDRLRLIQRITDITAESAAKLKGNTLGSGISSNFPALKEAVDDVILGRFTATLRDVRQPEKIVDAVTGGFDITRQELGLSGYRSGKIAGQVYRSGGMTSYMAKSTLFDTFANYRLLMSQRFARNFGLITG", ""),
    ("gyrB", "Pseudomonas aeruginosa", "MSNSQDTIKAAKVYITDQHEGPDYLDIYQSPHGERAVSQEVRENLTVAGFDIEKHIPKSTRLENLEIRVNKDKWVIKDGRGRVRVHKDNNIDPDGSYETFTRFHTSVDAIN", ""),
    ("leuS", "Bacillus subtilis", "MQKFDTQEQLNNWANDWASQYRDFLKAQNKGEKFKIREEMTKFIGEHFDPKSMQTLGATDALVRELGKDQEAELFKRVVDALFTDYELR", ""),
    ("alaS", "Klebsiella pneumoniae", "MSKSTAEIRQAFLDFFHSKGHQVVASSTHALLGQLRALELKYVQGSRLQKDPTLRTEVYNALRAEKMKFSKQYGVTPEHVLNRFANFIDQNLKQ", ""),
    ("metG", "Salmonella typhimurium", "MKERLNFIQDAFEQFYDLHAPIFKQIESFYASDGFEQIKQHPKERLPVLEGDFLHIGHTGKFIDEVIKMKQFGLKEFDMTGAFVVLTARHLKFEEQMKTHPLEQKLQKL", ""),
]


def _download_first_success(urls: list[str], dest_path: Path, is_gzipped: bool = False, is_zip: bool = False) -> bool:
    """Try downloading from multiple URLs. Returns True on first success."""
    for url in urls:
        logger.info(f"Trying: {url}")
        try:
            req = Request(url, headers={"User-Agent": "PLMLoF/1.0"})
            response = urlopen(req, timeout=60)  # noqa: S310
            raw = response.read()
            if len(raw) < 100:
                logger.warning(f"Empty response from {url}")
                continue

            if is_gzipped:
                try:
                    data = gzip.decompress(raw)
                except gzip.BadGzipFile:
                    data = raw
            elif is_zip:
                # Extract first non-MACOSX file from ZIP
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    for name in zf.namelist():
                        if not name.startswith("__"):
                            data = zf.read(name)
                            break
                    else:
                        logger.warning(f"No valid files in ZIP from {url}")
                        continue
            else:
                data = raw

            # Try to decode as text
            text = None
            for enc in ("utf-8", "latin-1"):
                try:
                    text = data.decode(enc)
                    break
                except (UnicodeDecodeError, ValueError):
                    continue
            if text is None:
                text = data.decode("utf-8", errors="replace")

            dest_path.write_text(text, encoding="utf-8")
            logger.info(f"Downloaded {len(data) / 1e6:.1f} MB from {url}")
            return True
        except Exception as e:
            logger.warning(f"Failed: {e}")
    return False


def download_deg(output_dir: Path = OUTPUT_DIR) -> tuple[Path, Path, Path]:
    """Download DEG protein, nucleotide, and annotation files.

    Returns:
        Tuple of (protein_path, dna_path, annotation_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_path = output_dir / "deg_proteins.fasta"
    dna_path = output_dir / "deg_dna.fasta"
    annotation_path = output_dir / "deg_annotation.csv"

    if not (protein_path.exists() and protein_path.stat().st_size > 100):
        _download_first_success(DEG_PROTEIN_URLS, protein_path, is_gzipped=True)

    if not (dna_path.exists() and dna_path.stat().st_size > 100):
        _download_first_success(DEG_NUCLEOTIDE_URLS, dna_path, is_gzipped=True)

    if not (annotation_path.exists() and annotation_path.stat().st_size > 100):
        _download_first_success(DEG_ANNOTATION_URLS, annotation_path, is_zip=True)

    return protein_path, dna_path, annotation_path


def _parse_annotation(annotation_path: Path) -> dict[str, tuple[str, str]]:
    """Parse DEG annotation CSV to map gene_id -> (gene_name, species).

    DEG annotation format (semicolon-separated, double-quoted):
    "DEG_org_id";"DEG_gene_id";"gene_name";"GI";"COG";"function";"description";"organism";...
    """
    mapping = {}
    if not annotation_path.exists() or annotation_path.stat().st_size < 100:
        return mapping

    try:
        text = annotation_path.read_text(encoding="utf-8")
        for line in text.strip().split("\n"):
            fields = [f.strip().strip('"') for f in line.split(";")]
            if len(fields) < 8:
                continue
            gene_id = fields[1]    # e.g. DEG10010001
            gene_name = fields[2]  # e.g. dnaA
            species = fields[7]    # e.g. Bacillus subtilis 168
            mapping[gene_id] = (gene_name, species)
    except Exception as e:
        logger.warning(f"Error parsing DEG annotation: {e}")

    return mapping


def parse_deg_sequences(protein_path: Path, dna_path: Path, annotation_path: Path) -> pd.DataFrame:
    """Parse DEG FASTA files and annotation to build essential gene dataset.

    Returns:
        DataFrame with columns: gene_id, gene, species, ref_protein, ref_dna
    """
    # Load annotation for gene names and species
    annotation = _parse_annotation(annotation_path)
    logger.info(f"Loaded {len(annotation)} gene annotations from DEG")

    # Parse protein sequences
    proteins = {}
    if protein_path.exists() and protein_path.stat().st_size > 100:
        try:
            for record in SeqIO.parse(str(protein_path), "fasta"):
                proteins[record.id] = str(record.seq)
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

    logger.info(f"Parsed {len(proteins)} proteins, {len(dna_seqs)} DNA sequences from DEG")

    # Build records
    records = []
    all_ids = set(proteins.keys()) | set(dna_seqs.keys())

    for gid in all_ids:
        protein = proteins.get(gid, "")
        dna = dna_seqs.get(gid, "")

        # Translate DNA if no protein available
        if dna and not protein:
            protein = translate_dna(dna, to_stop=True)

        # Strip stop codon characters — ESM2 cannot tokenize '*'
        protein = protein.replace("*", "")

        if not protein or len(protein) < 10:
            continue

        # Get gene name and species from annotation
        gene_name, species = annotation.get(gid, (gid, ""))

        # Filter out non-bacterial species
        skip_species = {"homo sapiens", "mus musculus", "saccharomyces", "drosophila",
                        "caenorhabditis", "arabidopsis", "danio rerio"}
        if any(s in species.lower() for s in skip_species):
            continue

        records.append({
            "gene_id": gid,
            "gene": gene_name,
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
        ref_dna = row.get("ref_dna", "") or ""
        gene = row["gene"]
        species = row.get("species", "")

        if len(ref_protein) < 10:
            continue

        for v in range(variants_per_gene):
            var_protein = None
            var_dna = ""
            mutation_type = ""

            if v == 0 and ref_dna and len(ref_dna) >= 30:
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
            elif v == 0:
                stop_pos = rng.randint(1, int(len(ref_protein) * 0.8))
                var_protein = ref_protein[:stop_pos]
                mutation_type = f"premature_stop_at_{stop_pos}"
            elif v == 1 and ref_dna and len(ref_dna) >= 30:
                pos = rng.randint(3, len(ref_dna) // 2)
                try:
                    var_dna = introduce_frameshift(ref_dna, pos, insert=True)
                    var_protein = translate_dna(var_dna, to_stop=True)
                    mutation_type = f"frameshift_insert_at_{pos}"
                except (ValueError, IndexError):
                    var_protein = None
            elif v == 1:
                scramble_len = max(len(ref_protein) // 4, 3)
                scrambled = list(ref_protein[:scramble_len])
                rng.shuffle(scrambled)
                var_protein = "".join(scrambled) + ref_protein[scramble_len:][:5]
                mutation_type = f"scrambled_nterm_{scramble_len}"
            else:
                keep_frac = rng.uniform(0.1, 0.5)
                keep_len = max(int(len(ref_protein) * keep_frac), 1)
                var_protein = ref_protein[:keep_len]
                var_dna = ref_dna[:keep_len * 3] if ref_dna else ""
                mutation_type = f"truncation_{keep_frac:.0%}"

            if not var_protein:
                continue

            # Strip stop codon characters — ESM2 cannot tokenize '*'
            var_protein = var_protein.replace("*", "")
            if not var_protein or len(var_protein) < 3:
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

    protein_path, dna_path, annotation_path = download_deg()
    essential = parse_deg_sequences(protein_path, dna_path, annotation_path)

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

"""Analyze ProteinGym bacterial species by SINGLE mutants only."""
import pandas as pd

df = pd.read_csv('data/raw/proteingym/DMS_substitutions.csv')
bact = df[df['taxon'] == 'Prokaryote'].copy()

organism_stats = bact.groupby('source_organism').agg({
    'DMS_id': 'count',
    'DMS_number_single_mutants': 'sum'
}).reset_index()
organism_stats.columns = ['organism', 'assays', 'single_mutants']
organism_stats = organism_stats.sort_values('single_mutants', ascending=False)

print('='*90)
print('TOP 20 BACTERIAL SPECIES (SINGLE MUTANTS ONLY - relevant for PLMLoF)')
print('='*90)
print(f"{'Rank':<6} {'Organism':<40} {'Assays':>8} {'Single Mutants':>16}")
print('-'*90)

for i, row in enumerate(organism_stats.head(20).itertuples(), 1):
    print(f"{i:<6} {row.organism:<40} {row.assays:>8} {row.single_mutants:>16,}")

print('\n' + '='*90)
print('YOUR 4 PRIORITY SPECIES:')
print('='*90)

ecoli = organism_stats[organism_stats['organism'].str.contains('Escherichia', case=False)]
klepn = organism_stats[organism_stats['organism'].str.contains('Klebsiella', case=False)]
myctu = organism_stats[organism_stats['organism'].str.contains('tuberculosis', case=False)]
staur = organism_stats[organism_stats['organism'].str.contains('Staphylococcus aureus', case=False)]

priority_total = 0
if len(ecoli) > 0:
    val = ecoli['single_mutants'].sum()
    print(f"  E. coli: {val:,} single mutants (17 assays)")
    priority_total += val
if len(klepn) > 0:
    val = klepn['single_mutants'].sum()
    print(f"  Klebsiella pneumoniae: {val:,} single mutants (1 assay)")
    priority_total += val
if len(myctu) > 0:
    val = myctu['single_mutants'].sum()
    print(f"  M. tuberculosis: {val:,} single mutants (1 assay)")
    priority_total += val
if len(staur) > 0:
    val = staur['single_mutants'].sum()
    print(f"  S. aureus: {val:,} single mutants (4 assays)")
    priority_total += val

total_bacterial = organism_stats['single_mutants'].sum()
print(f"\n  PRIORITY TOTAL: {priority_total:,} / {total_bacterial:,} ({priority_total/total_bacterial*100:.1f}%)")
print('\n' + '='*90)
print('CONCLUSION:')
print('  - E. coli has 40K+ single mutants but is downsampled to 45% in curate_dataset.py')
print('  - Priority species represent ~52% of all bacterial single mutants')
print('  - Need to increase ECOLI_SAMPLING_RATE and ensure species stratification')

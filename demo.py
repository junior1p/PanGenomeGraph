"""
PanGenomeGraph — Demo with synthetic bacterial genome data.
"""

import numpy as np
import pandas as pd
import os

# Synthetic E. coli-like pangenome: 10 strains, 5000 genes, 1000bp genome segments
np.random.seed(42)
n_strains = 10
n_genes = 5000
genome_size = 5000000  # 5 Mb

# Core genes: present in all strains (80% of genome)
# Accessory genes: present in 20-80% of strains
# Shell genes: present in <20% of strains
n_core = int(n_genes * 0.60)
n_accessory = int(n_genes * 0.30)
n_shell = n_genes - n_core - n_accessory

# Gene presence/absence matrix (binary)
gene_matrix = np.zeros((n_strains, n_genes), dtype=int)
gene_matrix[:, :n_core] = 1  # Core: all present

# Accessory: each gene has probability 0.5 of being present per strain
for j in range(n_core, n_core + n_accessory):
    gene_matrix[:, j] = (np.random.rand(n_strains) > 0.5).astype(int)

# Shell: each gene present in <20% of strains
for j in range(n_core + n_accessory, n_genes):
    gene_matrix[:, j] = (np.random.rand(n_strains) > 0.85).astype(int)

# Ensure at least 2 strains have each shell gene
for j in range(n_core + n_accessory, n_genes):
    if gene_matrix[:, j].sum() < 2:
        strain_indices = np.random.choice(n_strains, 2, replace=False)
        gene_matrix[strain_indices, j] = 1

gene_names = [f"gene_{i:05d}" for i in range(n_genes)]
strain_names = [f"strain_{i:03d}" for i in range(n_strains)]
gene_df = pd.DataFrame(gene_matrix, index=strain_names, columns=gene_names)

# Synthetic phenotypes (e.g., antibiotic resistance: 0=sensitive, 1=resistant)
phenotypes = pd.DataFrame({
    'strain': strain_names,
    'resistance': np.random.choice([0, 1], n_strains),
    'growth_rate': np.random.uniform(0.5, 2.0, n_strains),
})

# Core-accessory breakdown
n_total = n_genes
core_fraction = n_core / n_total
accessory_fraction = n_accessory / n_total
shell_fraction = n_shell / n_total

# Create synthetic GFA (simplified)
gfa_content = f"""H\tVN:Z:1.0
S\tREF\tATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
L\tREF\t+\tS1\t+\t1000\tM
L\tREF\t+\tS2\t+\t1000\tM
"""

os.makedirs('/tmp/pangenome_demo', exist_ok=True)
gene_df.to_csv('/tmp/pangenome_demo/gene_presence_absence.csv')
phenotypes.to_csv('/tmp/pangenome_demo/phenotypes.csv', index=False)
with open('/tmp/pangenome_demo/synthetic.gfa', 'w') as f:
    f.write(gfa_content)

print("=== PanGenomeGraph Demo ===")
print(f"Synthetic pangenome: {n_strains} strains × {n_genes} genes")
print(f"Genome size: {genome_size:,} bp per strain")
print(f"Core genes (>95% frequency): {n_core} ({core_fraction*100:.0f}%)")
print(f"Accessory genes (20-95%): {n_accessory} ({accessory_fraction*100:.0f}%)")
print(f"Shell genes (<20%): {n_shell} ({shell_fraction*100:.0f}%)")
print(f"Phenotypes: resistance (binary), growth_rate (continuous)")
print("\n=== Expected Outputs ===")
print("1. pangenome.vg — variation graph in vg binary format")
print("2. gene_presence_absence.csv — strain × gene binary matrix")
print("3. gwas_hits.tsv — GWAS hits (phenotype ~ gene presence)")
print("4. variation.bed — graph variation sites")
print("5. core_accessory_summary.tsv — per-strain core/accessory/shell fractions")
print("\nNote: Full implementation requires vg (https://github.com/vgteam/vg) and minigraph.")

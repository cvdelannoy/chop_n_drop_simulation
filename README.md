# Chop-n-drop: <i>in silico</i> assessment of a novel single-molecule protein fingerprinting method employing fragmentation and nanopore detection

Chop-n-drop is a concept single-molecule protein fingerprinting approach utilizing a proteasome-nanopore construct 
[[1]](#1). In short, proteins are processed and cleaved by a 
proteasome engineered to cleave at certain motifs, after which the molecular weight of the fragments is estimated from
the blockade they cause while traversing the nanopore. The sequence of measured fragment weights forms a fingerprint,
by which proteins can be recognized at single-molecule resolution. The code in this repository was used to simulate this
process and subsequent identification of fingerprints using a simple alignment-based classification algorithm. It can 
be used to repeat experiments described in our simulation study paper, or experiment with parameter settings of choice.

## Install
A miniconda installation is required. Pipelines have been tested on Ubuntu 20.04 and miniconda3.
From the repo main directory, install and activate the provided conda env:
```
conda env create -f env.yml
conda activate chop_n_drop_simulation
```

# Collect data
Query UniProt for fastas of reviewed human proteome entries:

```
python get_uniprot_targets.py \
    --out-dir path/to/output
```
Alternatively, the fastas used in our paper are included in this repo under `fasta/`

# Run simulation
A single simulation experiment is run as follows:
```
python run_sim.py \
    --efficiency 0.99 \ # proteasome cleaving efficiency
    --resolution 5 \ # measurement resolution in Da
    --catch-rate 0.99 \ # capture rate of nanopore
    --fasta-dir path/to/fastas \
    --out-dir store/results/here \
    --mode perfect_db \
    --cores nb_cores
```

  - `--mode` dictates what kind of simulation is run:
    - `perfect_db`: run using a single combination of noise parameter settings. Classification is done by aligning 
      fingerprints generated using given noise parameters against a database of fingerprints generated under noiseless
      conditions.
    - `pefect_db_range`: run using a range of values of noise parameters
  - To speed up running, use `--db` to provide a pre-generated database of perfect fingerprints for classification. 
    A database for the sequences in `fasta` is provided in this repo: `digested_products_database_full_perfect.pkl`
  - It is advisable to run on multiple CPU cores. Runs on the whole Swissprot human proteome (>20,000 entries) using
    24 cores should take roughly two hours.

For additional parameter settings, consult the script:
```
python run_sim.py --help
```

# Running examples
To repeat experiments for paper figure 2 (from repo main directory, otherwise adapt paths accordingly):
```
python run_sim.py \
    --fasta-dir fasta/ \
    --out-dir store/results/here  <-- adapt to desired working directory location
```

To repeat experiments for paper figure 3A:
```
python run_sim.py \
    --mode perfect_db_range \
    --fasta-dir fasta/ \
    --out-dir store/results/here  <-- adapt to desired working directory location
```

To repeat experiments for paper figure 3B:
```
python run_sim.py \
    --fasta-dir fasta/ \
    --out-dir store/results/here  <-- adapt to desired working directory location
    --resolution 10 \
    --catch-rate 0.9 \
    --efficiency 0.9
```
# Experimental data
We also demonstrated experimentally that FraC pores can discern a 4Da difference in protein weights. Data and analysis
script for this experiment are found [here](experimental/README.md).

# References
<a id="1">[1]</a> Shengli Zhang, Gang Huang, Roderick Versloot, Bart Marlon Herwig, Paulo Cesar Telles de Souza, Siewert-Jan Marrink, 
and Giovanni Maglia. Bottom-up fabrication of a multi-component nanopore sensor that unfolds, processes and 
recognizes single proteins. <i>bioRxiv</i>, doi: 10.1101/2020.12.04.411884, 2020.


---
with &hearts; from Wageningen University
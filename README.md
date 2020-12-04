# Chop-n-Drop simulation

## Quick-start
Install/activate provided conda env in `env.yml`. Query UniProt for 
fastas of reviewed human proteome entries:

```
python get_uniprot_targets.py \
    --out-dir path/to/output
``` 

Then run:
```
python run_sim.py \ 
    --fasta-dir path/to/fastas \
    --enzyme trypsin \
    --res-range 1000 4000 \
    --cores 4 \
    --out-dir store/results/here
    
```
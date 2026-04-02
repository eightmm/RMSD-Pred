#!/bin/bash

# RMSD-Pred Example Inference Script

if command -v rmsdpred &> /dev/null; then
    rmsdpred \
        -r prot.pdb \
        -l ligs.sdf \
        -o result.tsv \
        --batch_size 64 \
        --device cuda
else
    cd ..
    python -m rmsdpred.inference \
        -r example/prot.pdb \
        -l example/ligs.sdf \
        -o example/result.tsv \
        --batch_size 64 \
        --device cuda
    cd example
fi

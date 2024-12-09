#!/bin/bash
# Run the experiments and generate the figures used in the paper.

# python experiments/forward_backward/scale_with_derivs.py

ADAM_EPOCHS=15000
LBFGS_EPOCHS=30000
LR=0.00015

python experiments/self_similar_burgers/profiles.py 0 ntp --n_adam_epochs $ADAM_EPOCHS --n_lbfgs_epochs $LBFGS_EPOCHS --lr $LR
python experiments/self_similar_burgers/profiles.py 0 ad --n_adam_epochs $ADAM_EPOCHS --n_lbfgs_epochs $LBFGS_EPOCHS --lr $LR
python experiments/self_similar_burgers/profiles.py 1 ntp --n_adam_epochs $ADAM_EPOCHS --n_lbfgs_epochs $LBFGS_EPOCHS --lr $LR
python experiments/self_similar_burgers/profiles.py 1 ad --n_adam_epochs $ADAM_EPOCHS --n_lbfgs_epochs $LBFGS_EPOCHS --lr $LR
python experiments/self_similar_burgers/profiles.py 2 ntp --n_adam_epochs $ADAM_EPOCHS --n_lbfgs_epochs $LBFGS_EPOCHS --lr $LR

python experiments/self_similar_burgers/profiles.py 3 ntp --n_adam_epochs $ADAM_EPOCHS --n_lbfgs_epochs $LBFGS_EPOCHS --lr $LR
python experiments/self_similar_burgers/profiles.py 4 ntp --n_adam_epochs $ADAM_EPOCHS --n_lbfgs_epochs $LBFGS_EPOCHS --lr $LR

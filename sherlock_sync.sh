#!/bin/bash
dest=sherlock:/home/users/dtfrake

rsync -avK --delete \
  --exclude .git --exclude .venv --exclude __pycache__ \
  --exclude 'Data_And_Samplers' --exclude output \
  --include 'inferring_alpha_beta_last_position.npy' \
  --exclude '*.npy' --exclude '*.fits' \
  --exclude dustmaps_data \
  --exclude '*:Zone.Identifier' \
  /home/dtfrake/hmc-dust $dest
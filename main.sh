#!/bin/bash

# Base models
python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 1 --exp_id base --lr 0.00001 --weight_decay 0.0001 --train_disjoint --n_epochs 25
python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 2 --exp_id base --lr 0.00001 --weight_decay 0.0001 --train_disjoint --n_epochs 25
python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model LSTM_layers --layer -1 --N 3 --exp_id base --lr 0.00001 --weight_decay 0.0001 --train_disjoint --n_epochs 25

# Exemplar models
python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 1 --exp_id ex --ex_size 8 --lr 0.000002  --weight_decay 0.0001 --train_disjoint --n_epochs 50
python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 2 --exp_id ex --ex_size 8 --lr 0.000002  --weight_decay 0.0001 --train_disjoint --n_epochs 50
python MAIN_predict_intel_correctness_feats.py --feats WhisperFull --model ExLSTM_layers --layer -1 --N 3 --exp_id ex --ex_size 8 --lr 0.000002  --weight_decay 0.0001 --train_disjoint --n_epochs 50

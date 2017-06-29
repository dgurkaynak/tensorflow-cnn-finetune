#!/bin/sh

python finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc8,fc7,fc6"

python finetune.py \
	--num_epochs 30 \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1" # full training

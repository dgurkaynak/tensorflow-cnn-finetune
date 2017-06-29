#!/bin/sh

python finetune.py \
	--learning_rate "0.00001" \
	--num_epochs 25 \
	--multi_scale "225,256" \
	--train_layers "fc8,fc7,fc6,conv5_3,conv5_2,conv5_1,conv4_3,conv4_2,conv4_1,conv3_3,conv3_2,conv3_1"

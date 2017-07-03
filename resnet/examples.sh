#!/bin/sh

python finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc"

python finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc,scale5/block3"

python finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc,scale5/block3,scale5/block2"

python finetune.py \
	--learning_rate "0.00001" \
	--train_layers "fc,scale5/block3,scale5/block2,scale5/block1"

python finetune.py \
	--learning_rate "0.00001" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5"

python finetune.py \
	--learning_rate "0.00001" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4/block6"

python finetune.py \
	--learning_rate "0.00001" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4/block6,scale4/block5"

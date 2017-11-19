#!/bin/sh

python finetune.py \
	--learning_rate "0.00003" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5"

python finetune.py \
	--learning_rate "0.00003" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4"

python finetune.py \
	--learning_rate "0.00003" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4,scale3"

python finetune.py \
	--learning_rate "0.00003" \
	--batch_size "64" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4,scale3,scale2"

python finetune.py \
	--learning_rate "0.00005" \
	--batch_size "64" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4,scale3,scale2,scale1"




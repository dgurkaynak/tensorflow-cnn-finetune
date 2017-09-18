#!/bin/sh


python finetune.py \
	--learning_rate "0.00001" \
	--multi_scale "225,256" \
	--train_layers "fc,scale5,scale4,scale3"

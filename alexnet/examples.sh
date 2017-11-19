#!/bin/sh

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8"

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7"

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7,fc6"

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7,fc6,conv5"

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7,fc6,conv5,conv4"

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7,fc6,conv5,conv4,conv3"

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7,fc6,conv5,conv4,conv3,conv2"

python finetune.py \
	--learning_rate "0.0003" \
	--multi_scale "228,256" \
	--train_layers "fc8,fc7,fc6,conv5,conv4,conv3,conv2,conv1"

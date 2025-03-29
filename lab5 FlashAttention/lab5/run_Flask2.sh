#! /bin/bash
#change Flash2 batch size 32 OOM
python lab5.py \
	--batch_size 4 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_b=4.json

python lab5.py \
	--batch_size 8 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_b=8.json

python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_b=16.json

#change Flash2 seq_len
python lab5.py \
	--batch_size 16 \
	--seq_len 128 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_seq=128.json

python lab5.py \
	--batch_size 16 \
	--seq_len 256 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_seq=256.json
python lab5.py \
	--batch_size 16 \
	--seq_len 512 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_seq=512.json

#change Flash2 num_head
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 16 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_head=16.json

python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_head=32.json

#OOM
# python lab5.py \
# 	--batch_size 16 \
# 	--seq_len 1024 \
# 	--num_heads 64 \
# 	--emb_dim 2048 \
# 	--impl Flash2 \
# 	--causal \
# 	--repeats 30 \
# 	--output Flash2_head=64.json

# python lab5.py \
# 	--batch_size 16 \
# 	--seq_len 1024 \
# 	--num_heads 128 \
# 	--emb_dim 2048 \
# 	--impl Flash2 \
# 	--causal \
# 	--repeats 30 \
# 	--output Flash2_head=128.json

#change Flash2 emb_dim
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 1024 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_dim=1024.json
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_dim=2048.json
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 4096 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_dim=4096.json
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 8192 \
	--impl Flash2 \
	--causal \
	--repeats 30 \
	--output Flash2_dim=8192.json




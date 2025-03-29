batch_size 16#! /bin/bash
#change Pytorch batch size 32 OOM
python lab5.py \
	--batch_size 4 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_b=4.json

python lab5.py \
	--batch_size 8 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_b=8.json

python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_b=16.json

#change Pytorch seq_len
python lab5.py \
	--batch_size 16 \
	--seq_len 128 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_seq=128.json

python lab5.py \
	--batch_size 16 \
	--seq_len 256 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_seq=256.json
python lab5.py \
	--batch_size 16 \
	--seq_len 512 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_seq=512.json

#change Pytorch num_head
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 16 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_head=16.json

python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_head=32.json

#OOM
# python lab5.py \
# 	--batch_size 16 \
# 	--seq_len 1024 \
# 	--num_heads 64 \
# 	--emb_dim 2048 \
# 	--impl Pytorch \
# 	--causal \
# 	--repeats 30 \
# 	--output Pytorch_head=64.json

# python lab5.py \
# 	--batch_size 16 \
# 	--seq_len 1024 \
# 	--num_heads 128 \
# 	--emb_dim 2048 \
# 	--impl Pytorch \
# 	--causal \
# 	--repeats 30 \
# 	--output Pytorch_head=128.json

#change Pytorch emb_dim
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 1024 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_dim=1024.json
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 2048 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_dim=2048.json
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 4096 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_dim=4096.json
python lab5.py \
	--batch_size 16 \
	--seq_len 1024 \
	--num_heads 32 \
	--emb_dim 8192 \
	--impl Pytorch \
	--causal \
	--repeats 30 \
	--output Pytorch_dim=8192.json




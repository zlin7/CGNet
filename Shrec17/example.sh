
#compute the coefs
python precomputation.py --start_position 0 --end_position 2499 --dataset test --bw 128
python precomputation.py --start_position 2500 --end_position 4999 --dataset test --bw 128
python precomputation.py --start_position 5000 --end_position 7499 --dataset test --bw 128
python precomputation.py --start_position 7500 --end_position 10264 --dataset test --bw 128

python precomputation.py --start_position 0 --end_position 2999 --dataset train --bw 128
python precomputation.py --start_position 3000 --end_position 5999 --dataset train --bw 128
python precomputation.py --start_position 6000 --end_position 8999 --dataset train --bw 128
python precomputation.py --start_position 9000 --end_position 11999 --dataset train --bw 128
python precomputation.py --start_position 12000 --end_position 14999 --dataset train --bw 128
python precomputation.py --start_position 15000 --end_position 17999 --dataset train --bw 128
python precomputation.py --start_position 18000 --end_position 20999 --dataset train --bw 128
python precomputation.py --start_position 21000 --end_position 23999 --dataset train --bw 128
python precomputation.py --start_position 24000 --end_position 26999 --dataset train --bw 128
python precomputation.py --start_position 27000 --end_position 29999 --dataset train --bw 128
python precomputation.py --start_position 30000 --end_position 32999 --dataset train --bw 128
python precomputation.py --start_position 33000 --end_position 35763 --dataset train --bw 128

python precomputation.py --start_position 0 --end_position 2999 --dataset val --bw 128
python precomputation.py --start_position 3000 --end_position 5132 --dataset val --bw 128

#after all coefs are computed, do the following to merge them for training and testing (can skip if you always lazy read)
python precomputation.py --merge --dataset test
python precomputation.py --merge --dataset train
python precomputation.py --merge --dataset val

#Now can run training..
python Shrec17_main_fast.py --cuda --num-epoch 1 --tau_type 1 --tau_man 3 --nlayers 2 --batch-size 50 --lmax 8 --norm 1 --skip 3 --nfc 1 --lr 0.0005 --weight-decay 0.001 --dropout 0.5
#After training, run with the same spec, but with --predict. This evaluates the result..
python Shrec17_main_fast.py --cuda --num-epoch 1 --tau_type 1 --tau_man 3 --nlayers 2 --batch-size 50 --lmax 8 --norm 1 --skip 3 --nfc 1 --lr 0.0005 --weight-decay 0.001 --dropout 0.5 --predict
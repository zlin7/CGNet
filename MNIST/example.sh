#~!/bin/bash

#Rotate vs Rotate
#The following specification gives accuracy on test 96.6%
python main_fast.py --cuda --nlayers 5 --tau_type 3 --tau_man 12 --num-epoch 30 --lmax 10 --batch-size 100 --skip 1 --norm 1 --rotate-train --dropout 0.5

#NR/NR
#The following specification gives accuracy on test 96.4%
python main_fast.py --cuda --nlayers 5 --tau_type 3 --tau_man 12 --num-epoch 30 --lmax 10 --batch-size 100 --skip 1 --norm 1 --unrot-test --dropout 0.5

#NR/R
#The following specification gives accuracy on test 96.0%
python main_fast.py --cuda --nlayers 5 --tau_type 3 --tau_man 12 --num-epoch 30 --lmax 10 --batch-size 100 --skip 1 --norm 1 --dropout 0.5


#('Shapes of parameters', [(1643, 2), (31023, 2), (31023, 2), (31023, 2), (31023, 2), (122,), (122,), (256, 122), (256,), (10, 256), (10,)], 285772)
#285772 parameters in total, 
#~!/bin/bash

#Rotate vs Rotate
python -m MNIST.main --num-epoch 30 --lmax 11 --batch-size 100 --skip 1 --norm 1 --rotate-train --dropout 0.5

#NR/NR
python -m MNIST.main --num-epoch 30 --lmax 11 --batch-size 100 --skip 1 --norm 1  --unrot-test --dropout 0.5

#NR/R
python -m MNIST.main --num-epoch 30 --lmax 11 --batch-size 100 --skip 1 --norm 1  --dropout 0.5

#('Shapes of parameters', [(1643, 2), (31023, 2), (31023, 2), (31023, 2), (31023, 2), (122,), (122,), (256, 122), (256,), (10, 256), (10,)], 285772)
#285772 parameters in total, 
import os
#import random
CUR_DIR = os.path.dirname(os.path.realpath(__file__))


if not os.path.exists(CUR_DIR + "/temp/"):
	os.makedirs(CUR_DIR + "/temp/")

#precomputing the coefficients
with open(CUR_DIR + "/temp/data_mother.sh", 'w') as fm:
	fm.write("#!/bin/bash\n")
	for i in range(10):
		fm.write("sbatch -p cpu -c1 ./temp/preprocessing_sh{}.sh\n".format(i))
		st = i * 6000
		ed = st + 6000
		with open(CUR_DIR + "/temp/preprocessing_sh{}.sh".format(i), 'w') as fo:
			fo.write("#!/bin/bash\n")
			fo.write("source activate pytorch\n")
			fo.write("python datautils.py {} {} {}".format(st, ed, 2))
	for i in range(2):
		fm.write("sbatch -p cpu -c1 ./temp/preprocessing_test_sh{}.sh\n".format(i))
		st = i * 5000
		ed = st + 5000
		with open(CUR_DIR + "/temp/preprocessing_test_sh{}.sh".format(i), 'w') as fo:
			fo.write("#!/bin/bash\n")
			fo.write("source activate pytorch\n")
			fo.write("python datautils.py {} {} {}\n".format(st, ed, 3))


child_id = 0
def write_a_child_sh(mother_file, params=" ", core_num=1, repeat=3):
	global child_id
	child_id += 1
	for rep in range(repeat):
		mother_file.write("sbatch -p {}pu -c{} -J se_{} -d singleton ./temp/train_sh{}.sh\n".format('g' if 'cuda' in params else 'c', core_num, child_id, child_id))
	with open(CUR_DIR + "/temp/train_sh{}.sh".format(child_id), 'w') as fo:
		fo.write("#!/bin/bash\n")
		fo.write("source activate pytorch\n")
		fo.write("python main_fast.py" + params + "\n")


with open(CUR_DIR + "/temp/mother.sh", 'w') as fm:
	fm.write("#!/bin/bash\n")
	"""
	write_a_child_sh(fm, " --cuda --num-epoch 30 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1", core_num=1, repeat=0)
	write_a_child_sh(fm, " --cuda --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1", core_num=1, repeat=3)
	write_a_child_sh(fm, " --cuda --num-epoch 30 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 3", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 30 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 2", core_num=1, repeat=0)
	write_a_child_sh(fm, " --cuda --num-epoch 30 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 2 --skip 1", core_num=1, repeat=0)
	write_a_child_sh(fm, " --cuda --num-epoch 30 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 2 --skip 3", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 3 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --turn-off-rotate", core_num=1, repeat=0)
	"""
	#Experiment 4.21
	"""
	write_a_child_sh(fm, " --cuda --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 0 --skip 1 --relu", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --relu", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 2 --batch-size 50 --lmax 8 --norm 1 --skip 2", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 2 --batch-size 50 --lmax 8 --norm 0 --skip 2 --relu", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 2 --batch-size 50 --lmax 8 --norm 1 --skip 2 --relu", core_num=1, repeat=0)


	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --rotate-train", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --relu --rotate-train", core_num=1, repeat=0)

	write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 2 --batch-size 50 --lmax 8 --norm 0 --skip 2 --relu --rotate-train", core_num=1, repeat=0)

	#Experiment 4.22
	write_a_child_sh(fm, " --cuda --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --rotate-train", core_num=1, repeat=0)

	#Experiment 4.26
	write_a_child_sh(fm, " --cuda --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 10 --lmax 10 --norm 1 --skip 1 --rotate-train", core_num=1, repeat=4)
	write_a_child_sh(fm, " --cuda --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 10 --lmax 12 --norm 1 --skip 1 --rotate-train", core_num=1, repeat=4)
	#write_a_child_sh(fm, " --cuda --num-epoch 20 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --rotate-train", core_num=1, repeat=6)
	write_a_child_sh(fm, " --cuda --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 10 --lmax 12 --norm 1 --skip 1 --rotate-train", core_num=1, repeat=4)
	"""
	#write_a_child_sh(fm, " --num-epoch 30 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --rotate-train --use-old --cuda", core_num=1, repeat=4)
	#write_a_child_sh(fm, " --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --dropout 0.5 --rotate-train --use-old --cuda", core_num=1, repeat=4)
	#write_a_child_sh(fm, " --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --dropout 0.5 --nfc 2 --rotate-train --use-old --cuda", core_num=1, repeat=4)
	
	write_a_child_sh(fm, " --num-epoch 30 --cuda --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --rotate-train", core_num=1, repeat=0)
	write_a_child_sh(fm, " --num-epoch 10 --cuda --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --dropout 0.5 --rotate-train", core_num=1, repeat=0)
	write_a_child_sh(fm, " --num-epoch 10 --cuda --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1 --dropout 0.5 --nfc 2 --rotate-train", core_num=1, repeat=0)
	write_a_child_sh(fm, " --num-epoch 20 --cuda --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 200 --lmax 8 --norm 1 --skip 1 --rotate-train --nfc 1 --dropout 0.5", core_num=1, repeat=1)
	#write_a_child_sh(fm, " --num-epoch 10 --tau_type 3 --tau_man 12 --nlayers 5 --batch-size 50 --lmax 8 --norm 1 --skip 1", core_num=1, repeat=4)
echo "----------- Mini-Batch Size Experiment ---------------"

echo 'one'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha06_batch100_epochs5.pickle' --alpha .06 --batch-size 100 --num-epochs 5

echo 'two'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha06_batch1000_epochs5.pickle' --alpha .06 --batch-size 1000 --num-epochs 5

echo 'three'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha06_batch500_epochs5.pickle' --alpha .06 --batch-size 500 --num-epochs 5

echo 'four'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha06_batch50_epochs5.pickle' --alpha .06 --batch-size 50 --num-epochs 5

echo 'five'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha06_batch5_epochs5.pickle' --alpha .06 --batch-size 5 --num-epochs 5

echo "----------- Full-Batch Size Experiment ---------------"

echo 'zero. Zero indexed by mistake, but its staying this way'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha05_batchFULL_epochs20.pickle' --alpha .05 --batch-size -1 --num-epochs 20

echo 'one'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha01_batchFULL_epochs20.pickle' --alpha .1 --batch-size -1 --num-epochs 20

echo 'two'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha20_batchFULL_epochs20.pickle' --alpha .2 --batch-size -1 --num-epochs 20

echo 'three'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha40_batchFULL_epochs20.pickle' --alpha .4 --batch-size -1 --num-epochs 20

echo 'four'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha01_batchFULL_epochs20.pickle' --alpha .01 --batch-size -1 --num-epochs 20

echo 'five'

python batch_mnistNN.py --error-file 'batch_H1_300_H2_100_alpha06_batchFULL_epochs20.pickle' --alpha .6 --batch-size -1 --num-epochs 20

echo "----------- NN Size Experiment ---------------"

echo 'zero. Zero indexed by mistake, but its staying this way'
#python batch_mnistNN.py --error-file 'train_error_online_NN0.pickle' --test-error-on 1 --test-error-file 'test_error_online_NN0.pickle' --alpha .02 --batch-size 1 --NN-setup 0 --num-epochs 3
echo 'one'
python batch_mnistNN.py --error-file 'pickles/train_error_online_NN1.pickle' --test-error-on 1 --test-error-file 'pickles/test_error_online_NN1.pickle' --alpha .02 --batch-size 1 --NN-setup 1 --num-epochs 3
echo 'two'
python batch_mnistNN.py --error-file 'pickles/train_error_online_NN2.pickle' --test-error-on 1 --test-error-file 'pickles/test_error_online_NN2.pickle' --alpha .02 --batch-size 1 --NN-setup 2 --num-epochs 3
echo 'three'
python batch_mnistNN.py --error-file 'pickles/train_error_online_NN3.pickle' --test-error-on 1 --test-error-file 'pickles/test_error_online_NN3.pickle' --alpha .02 --batch-size 1 --NN-setup 3 --num-epochs 3
echo 'four'
python batch_mnistNN.py --error-file 'pickles/train_error_online_NN4.pickle' --test-error-on 1 --test-error-file 'pickles/test_error_online_NN4.pickle' --alpha .02 --batch-size 1 --NN-setup 4 --num-epochs 3
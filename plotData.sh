for D in ./ttpickle/train*
do
    echo $D
    python testplot.py --filename $D
done

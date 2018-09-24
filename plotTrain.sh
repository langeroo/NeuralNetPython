for D in ./pickles/*
do
    echo $D
    python trainplot.py --filename $D
done

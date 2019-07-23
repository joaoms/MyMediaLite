#!/bin/bash -e

PROGRAM="bin/rating_prediction"
DATA_DIR=data/ml-100k
K=2

echo "MyMediaLite load/save test script"
echo "This will take about 9 minutes ..."

echo
echo "rating predictors"
echo "-----------------"

for method in MatrixFactorization BiasedMatrixFactorization UserItemBaseline GlobalAverage UserAverage
do
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR | perl -pe "s/\w+_time\s*\S+//g" | tee output1.txt
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR | perl -pe "s/\w+_time\s*\S+//g" | tee output2.txt
	diff --ignore-space-change output1.txt output2.txt
	rm tmp.model*
done


method=ItemAverage
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --save-user-mapping=um.txt --save-item-mapping=im.txt
     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --save-user-mapping=um.txt --save-item-mapping=im.txt | perl -pe "s/\w+_time\s*\S+//g" | tee output1.txt
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --load-user-mapping=um.txt --load-item-mapping=im.txt
     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --load-user-mapping=um.txt --load-item-mapping=im.txt | perl -pe "s/\w+_time\s*\S+//g" | tee output2.txt
diff --ignore-space-change output1.txt output2.txt
rm tmp.model* um.txt im.txt


for method in SVDPlusPlus SigmoidSVDPlusPlus SigmoidUserAsymmetricFactorModel SigmoidItemAsymmetricFactorModel
do
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --save-user-mapping=um.txt --save-item-mapping=im.txt --recommender-options=\"num_iter=1\"
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --save-user-mapping=um.txt --save-item-mapping=im.txt --recommender-options="num_iter=1" | perl -pe "s/\w+_time\s*\S+//g" | tee output1.txt
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --load-user-mapping=um.txt --load-item-mapping=im.txt --recommender-options=\"num_iter=1\"
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --load-user-mapping=um.txt --load-item-mapping=im.txt --recommender-options="num_iter=1" | perl -pe "s/\w+_time\s*\S+//g" | tee output2.txt
	diff --ignore-space-change output1.txt output2.txt
	rm tmp.model* um.txt im.txt
done


for method in UserKNN ItemKNN
do
	for c in BinaryCosine Pearson ConditionalProbability
	do
		echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$c" --save-model=tmp.model --data-dir=$DATA_DIR
		     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$c" --save-model=tmp.model --data-dir=$DATA_DIR | perl -pe "s/\w+_time \S+//g" | tee output1.txt
		echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$c" --load-model=tmp.model --data-dir=$DATA_DIR
		     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$c" --load-model=tmp.model --data-dir=$DATA_DIR | perl -pe "s/\w+_time \S+//g" | tee output2.txt
		diff --ignore-space-change output1.txt output2.txt
		rm tmp.model*
	done
done

method=ItemAttributeKNN
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --save-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt
     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --save-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt | perl -pe "s/\w+_time \S+//g" | tee output1.txt
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --load-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt
     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --load-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt | perl -pe "s/\w+_time \S+//g" | tee output2.txt
diff --ignore-space-change output1.txt output2.txt
rm tmp.model*


echo
echo "item recommenders"
echo "-----------------"

PROGRAM="bin/item_recommendation"

method=MostPopular
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --random-seed=1
     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output1.txt
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --random-seed=1
     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output2.txt
diff --ignore-all-space output1.txt output2.txt
rm tmp.model*


for method in WRMF BPRMF
do
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --save-user-mapping=um.txt --save-item-mapping=im.txt --recommender-options=\"num_iter=1\" --random-seed=1
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --save-model=tmp.model --data-dir=$DATA_DIR --save-user-mapping=um.txt --save-item-mapping=im.txt --recommender-options="num_iter=1" --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output1.txt
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --load-user-mapping=um.txt --load-item-mapping=im.txt --recommender-options=\"num_iter=1\"  --random-seed=1
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --load-model=tmp.model --data-dir=$DATA_DIR --load-user-mapping=um.txt --load-item-mapping=im.txt --recommender-options="num_iter=1" --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output2.txt
	diff --ignore-all-space output1.txt output2.txt
	rm tmp.model*
	rm um.txt im.txt
done

for cor in Cosine BidirectionalConditionalProbability
do
	for method in UserKNN ItemKNN
	do
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$cor" --save-model=tmp.model --data-dir=$DATA_DIR --measures=prec@5 --random-seed=1
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$cor" --save-model=tmp.model --data-dir=$DATA_DIR --measures=prec@5 --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output1.txt
	echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$cor" --load-model=tmp.model --data-dir=$DATA_DIR --measures=prec@5 --random-seed=1
	     $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K correlation=$cor" --load-model=tmp.model --data-dir=$DATA_DIR --measures=prec@5 --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output2.txt
	diff --ignore-all-space output1.txt output2.txt
	rm tmp.model*
	done
done

method=ItemAttributeKNN
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --save-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt --random-seed=1
$PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --save-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output1.txt
echo $PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --load-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt --random-seed=1
$PROGRAM --training-file=u1.base --test-file=u1.test --recommender=$method --recommender-options="k=$K" --load-model=tmp.model --data-dir=$DATA_DIR --item-attributes=item-attributes-genres.txt --random-seed=1 | perl -pe "s/\w+_time \S+//g" | tee output2.txt
diff --ignore-all-space output1.txt output2.txt
rm tmp.model*

rm output1.txt output2.txt

all: ../lib/mymedialite
	xbuild /p:Configuration=Debug MyMediaLite-nogui.sln
	cp -r Programs/ItemRecommendation/bin/Debug/*.dll ../lib/mymedialite
	cp -r Programs/ItemRecommendation/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/ItemRecommendation/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/RatingPrediction/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/RatingPrediction/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/RatingBasedRanking/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/RatingBasedRanking/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/OnlineEvaluator/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/OnlineEvaluator/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/OnlineCrossval/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/OnlineCrossval/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/BaggingEvaluator/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/BaggingEvaluator/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/BatchEvaluator/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/BatchEvaluator/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/ForgettingProfile/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/ForgettingProfile/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/OnlineLearner/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/OnlineLearner/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/KFoldOnlineEval/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/KFoldOnlineEval/bin/Debug/*.mdb ../lib/mymedialite
	cp -r Programs/KFoldPrequentialEval/bin/Debug/*.exe ../lib/mymedialite
	cp -r Programs/KFoldPrequentialEval/bin/Debug/*.mdb ../lib/mymedialite

../lib/mymedialite:
	mkdir ../lib/mymedialite

clean:
	rm -f */bin/Debug/*
	rm -f */bin/Release/*
	rm -f */*/bin/Debug/*
	rm -f */*/bin/Release/*
	rm -rf */obj
	rm -rf */*/obj

${PREFIX}/lib:
	mkdir ${PREFIX}/lib

${PREFIX}/bin:
	mkdir ${PREFIX}/bin

install: uninstall ${PREFIX}/lib ${PREFIX}/bin
	mkdir ${PREFIX}/lib/mymedialite
	cp ../lib/mymedialite/* ${PREFIX}/lib/mymedialite
	cp ../bin/* ${PREFIX}/bin

uninstall:
	rm -rf ${PREFIX}/lib/mymedialite
	rm -rf ${PREFIX}/bin/item_recommendation
	rm -rf ${PREFIX}/bin/rating_prediction
	rm -rf ${PREFIX}/bin/rating_based_ranking

test:
	nunit-console --run=Tests MyMediaLite-nogui.sln --config=Debug

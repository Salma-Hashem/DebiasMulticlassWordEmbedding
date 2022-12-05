#!/bin/bash
javac -cp maxent-3.0.0.jar:trove.jar *.java
python3 new_main.py WSJ_02-21.pos-chunk training.feature
java -Xmx32g -cp .:maxent-3.0.0.jar:trove.jar MEtrain training.feature model.chunk
python3 new_main.py WSJ_24.pos test.feature
java -Xmx32g -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk response.chunk
python3 score.chunk.py WSJ_24.pos-chunk response.chunk
python3 new_main.py WSJ_23.pos test.feature
java -Xmx32g -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk WSJ_23.chunk

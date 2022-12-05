#!/bin/bash
python3 main.py WSJ_23.pos test.feature
java -Xmx16g -cp .:maxent-3.0.0.jar:trove.jar MEtag test.feature model.chunk WSJ_23.chunk


#!/bin/bash

ANTLR_JAR="antlr-4.13.2-complete.jar"
java -jar $ANTLR_JAR -Dlanguage=Python3 QueryLanguage.g4

#!/bin/bash

CAFFEINE_JAR="caffeine-3.2.0.jar"

HEADER_DIR="jni_headers"

if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32" ]]; then
    CP_DELIM=";"
else
    CP_DELIM=":"
fi

CLASSPATH="$CAFFEINE_JAR${CP_DELIM}."

SOURCES="OperationCache.java OperationResult.java"

mkdir -p "$HEADER_DIR"

javac -cp "$CLASSPATH" -h "$HEADER_DIR" $SOURCES

if [ $? -eq 0 ]; then
    echo "✅ Java コンパイル & ヘッダ生成成功！"
    echo "📁 ヘッダファイルは $HEADER_DIR/ に出力されました。"
else
    echo "❌ 失敗しました..."
fi

#!/bin/bash

# =========================
# Java JNIヘッダ生成スクリプト
# =========================

# ---- 設定 ----
CAFFEINE_JAR="caffeine-3.2.0.jar"
HEADER_DIR="jni_headers"

# プラットフォームによるクラスパス区切り
if [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* || "$OSTYPE" == "win32"* ]]; then
    CP_DELIM=";"
else
    CP_DELIM=":"
fi

CLASSPATH="$CAFFEINE_JAR${CP_DELIM}."

# Javaファイル自動検出（OperationCache.java, OperationResult.javaが必須）
SOURCES=$(ls OperationCache.java OperationResult.java 2>/dev/null)
if [[ -z "$SOURCES" ]]; then
    echo "❌ 必要なJavaファイル(OperationCache.java, OperationResult.java)が見つかりません。"
    exit 1
fi

# ヘッダディレクトリ作成
mkdir -p "$HEADER_DIR"

# コンパイル&JNIヘッダ生成
echo "🔄 javac -cp \"$CLASSPATH\" -h \"$HEADER_DIR\" $SOURCES"
javac -cp "$CLASSPATH" -h "$HEADER_DIR" $SOURCES

if [ $? -eq 0 ]; then
    echo "✅ Javaコンパイル ＆ JNIヘッダ生成成功！"
    echo "📁 ヘッダファイルは $HEADER_DIR/ に出力されました。"
    ls -1 "$HEADER_DIR"/*.h 2>/dev/null
else
    echo "❌ Javaコンパイル or ヘッダ生成に失敗しました..."
    exit 2
fi
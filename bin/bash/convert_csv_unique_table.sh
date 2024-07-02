#!/bin/bash

# 入力ファイル名と出力ファイル名
input_file="unique_table.txt"
output_file="unique_table.csv"

# CSVファイルのヘッダーを書き込む
echo "Node ID,Address" > $output_file

# テキストファイルを1行ずつ読み取り、CSV形式に変換する
while IFS= read -r line; do
    # "Node ID: " と ", Address: " を区切り文字として使用して分割
    node_key=$(echo $line | awk -F'[ ,:]+' '{print $3}')
    node_value=$(echo $line | awk -F'[ ,:]+' '{print $6}')

    # CSV形式で出力ファイルに書き込む
    echo "$node_key,$node_value" >> $output_file
done < "$input_file"

#!/bin/bash

python3 ./creation/bretrieve_bug_files.py
directory=$1
if [ -z "$directory" ]; then
    echo "Please provide a directory"
    exit 1
fi

for file in $(find "$directory" -type f -name "*.java"); do
    java_file=$(basename "$file")
    dir=$(dirname "$file")

    for graph in "ast,cfg,dfg" "ast,cfg" "ast,dfg" "ast" "dfg" "cfg"; do
        output_file="$dir/output-${graph//,/-.}.dot"
        if [ ! -f "$output_file" ]; then
            # Run comex command with a timeout of 10 seconds
            timeout 20s comex --lang "java" --code-file "$file" --graphs "$graph"
            # Check if timeout command succeeded or not
            if [ $? -eq 124 ]; then
                echo "Command 'comex' timed out for file $dir and graph $graph"
            else
                mv "output.dot" "$output_file"
            fi
        fi
    done

    echo "Converted $java_file into dot files"
done

echo "Conversion complete!"

python3 ./creation/bcomex_data.py

python3 ./models/Train_ALL.py > output/txt/All.txt
python3 ./models/Train_ast_cfg.py > output/txt/ac.txt
python3  ./models/Train_ast_dfg.py > output/txt/ad.txt
python3 ./models/Train_ast.py > output/txt/a.txt
python3 ./models/Train_cfg.py > output/txt/c.txt
python3 ./models/Train_dfg.py > output/txt/d.txt
import os
import json
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def process_dot_file(dot_file_path):
    try:
        with open(dot_file_path, 'r') as file:
            dot_content = file.read()

        # Extracting node and edge labels from .dot content
        node_labels = []
        edge_labels = []
        lines = dot_content.split('\n')
        for line in lines:
            if 'label=' in line:
                label_start = line.find('label="') + len('label="')
                label_end = line.find('"', label_start)
                label = line[label_start:label_end]
                node_labels.extend(label.split())
            elif 'label=' not in line:
                label_start = line.find('label=') + len('label=')
                label_end = line.find(']', label_start)
                label = line[label_start:label_end]
                edge_labels.append(label)

        # Tokenize the node and edge labels
        node_tokens = [simple_preprocess(label) for label in node_labels]
        edge_tokens = [simple_preprocess(label) for label in edge_labels]

        # Train Word2Vec model
        model = Word2Vec(node_tokens + edge_tokens, vector_size=10, window=5, min_count=1, workers=4)

        # Prepare dictionary to store word vectors
        word_vectors = {}
        for word in model.wv.index_to_key:
            word_vectors[word] = model.wv[word].tolist()

        # Write word vectors to JSON file
        # output_file = os.path.splitext(dot_file_path)[0] + '.json'
        # with open(output_file, 'w') as json_file:
        #     json.dump(word_vectors, json_file)

        # print(f"Word vectors for {dot_file_path} have been saved to {output_file}")
        return word_vectors
    except:
        print(f"Error processing {dot_file_path}")
        return {}

def process_dot_in_df(directory_name):
    # os.system(f"./bdotfile.sh {directory_name}")

    bug_data = pd.read_csv('creation/Aspectj.csv')
    print(bug_data.columns)
    # Iterate over each row in the DataFrame
    bug_data['word_vector_all'] = None
    bug_data['word_vector_ast_cfg'] = None
    bug_data['word_vector_ast_dfg'] = None
    bug_data['word_vector_ast'] = None
    bug_data['word_vector_cfg'] = None
    bug_data['word_vector_dfg'] = None

    for index, row in bug_data.iterrows():
        directory = row['content'].rsplit('/', 1)[0]
        word_vectors_all = process_dot_file(f"{directory}/output-all.dot")
        word_vectors_ast_cfg = process_dot_file(f"{directory}/output-ast-cfg.dot")
        word_vectors_ast_dfg = process_dot_file(f"{directory}/output-ast-dfg.dot")
        word_vectors_ast = process_dot_file(f"{directory}/output-ast.dot")
        word_vectors_cfg = process_dot_file(f"{directory}/output-cfg.dot")
        word_vectors_dfg = process_dot_file(f"{directory}/output-dfg.dot")

        bug_data.at[index, 'word_vector_all'] = word_vectors_all
        bug_data.at[index, 'word_vector_ast_cfg'] = word_vectors_ast_cfg
        bug_data.at[index, 'word_vector_ast_dfg'] = word_vectors_ast_dfg
        bug_data.at[index, 'word_vector_ast'] = word_vectors_ast
        bug_data.at[index, 'word_vector_cfg'] = word_vectors_cfg
        bug_data.at[index, 'word_vector_dfg'] = word_vectors_dfg
    
    bug_data.to_csv('creation/Aspectj_word_vectors.csv', index=False, sep="\t")
    return bug_data

directory_name = 'creation/processed_data/eclipse-aspectj/aspectJ'
process_dot_in_df(directory_name)
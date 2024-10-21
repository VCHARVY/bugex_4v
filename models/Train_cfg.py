import csv
import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.metrics import accuracy_score
import pandas as pd
import os

print("PreProcessing csv file")
def file_exists(file_path):
    """
    Check if a file exists at the specified path.
    """
    return os.path.exists(file_path)

def filter_csv(csv_file_path):
    """
    Read the CSV file, modify the file paths in the "files-changed" column,
    check if the modified file paths exist, and delete the rows if the files do not exist.
    """
    rows_to_keep = []
    rows_deleted = 0

    with open(csv_file_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Read the header
        rows_to_keep.append(header)  # Add header to the list of rows to keep
        for row in csv_reader:
            issue, file_path, y_value = row
            modified_file_path = file_path[:-5] + "-ALL.json"  # Modify the file path
            if file_exists(modified_file_path):
                rows_to_keep.append(row)
            else:
                #print(f"File not found: {modified_file_path}. Deleting row...")
                rows_deleted += 1

    # Write the filtered rows back to the CSV file
    with open(csv_file_path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerows(rows_to_keep)

    #print(f"Deleted {rows_deleted} rows from the CSV file.")

# Example usage
# csv_file_path = "dataset_ALL.csv"  # Replace with your CSV file path
# filter_csv(csv_file_path)


# Load pre-trained BERT model and tokenizer
print("Loading BERT tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read CSV file and extract data
print("Reading CSV file...")
csv_file_path = "Aspectj_word_vectors.csv"
vector_words = []
y_values = []
issues = []

bug_Data = pd.read_csv('./creation/Aspectj_word_vectors.csv', sep ='\t')
y_values = bug_Data['y_values'].tolist()
issues = bug_Data['content'].tolist()
file_paths = bug_Data['filename'].tolist()

# Vectorize issues using BERT
print("Vectorizing issues using BERT...")
vectorized_issues = []
for i, issue in enumerate(issues):
    print(f"Vectorizing issue {i+1}/{len(issues)}...")
    inputs = tokenizer(issue, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vector_representation = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    vectorized_issues.append(vector_representation)

# Read JSON files and extract vectorized data
print("Reading JSON files and extracting vectorized data...")
vectorized_java_code = []
# max_vector_size = max(len(sublist) for sublist in vectorized_java_code)  # Initialize variable to keep track of the maximum vector size
max_vector_size = 0
for i, vector in enumerate(bug_Data['word_vector_cfg']):
    print(f"Processing vector {i+1}/{len(bug_Data['word_vector_ast_cfg'])}")
    vectorized_data_found = False
    vectors = []
    if isinstance(vector, list) and all(isinstance(x, float) for x in vector):
        print(f"Vectorized data extracted")
        vectors = vector  # Assign the vector to the list
        vectorized_data_found = True
    if not vectorized_data_found:
        print(f"No vectorized data found in row {i+1}")
    vectorized_java_code.append(vectors)
    max_vector_size = max(max_vector_size, len(vectors))  # Update maximum vector size  # Update maximum vector size

# Pad or truncate vectors to match the size of the largest vector
for i, vectors in enumerate(vectorized_java_code):
    if len(vectors) < max_vector_size:
        # Pad vectors with zeros
        vectorized_java_code[i] += [0.0] * (max_vector_size - len(vectors))
    elif len(vectors) > max_vector_size:
        # Truncate vectors
        vectorized_java_code[i] = vectors[:max_vector_size]

# Prepare data for model training
print("Preparing data for model training...")
X = np.concatenate((vectorized_issues, vectorized_java_code), axis=1)
y = np.array(y_values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Convert data to PyTorch tensors and create DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class BLT(nn.Module):
    def __init__(self, input_size, output_size):
        super(BLT, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.transformer(x)
        x = self.fc(x)
        return torch.sigmoid(x)

# class BLT_FT(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(BLT_FT, self).__init__()
#         self.bert_model = BertModel.from_pretrained('bert-base-uncased')
#         self.fc = nn.Linear(self.bert_model.config.hidden_size, output_size)

#     def forward(self, x):
#         x = x.unsqueeze(0)  # Add batch dimension
#         attention_mask = torch.ones(x.shape, dtype=torch.long)  # Create attention mask
#         x = self.bert_model(x, attention_mask=attention_mask)
#         x = self.fc(x.last_hidden_state[:, 0, :])
#         return torch.sigmoid(x)
    
class HybridDL(nn.Module):
    def __init__(self, input_size, output_size):
        super(HybridDL, self).__init__()
        self.cnn = nn.Conv1d(1, 10, kernel_size=5)
        self.rnn = nn.LSTM(input_size, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.cnn(x.unsqueeze(1))
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)

# Define the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the loss function and optimizer for each model
criterion = nn.BCELoss()
optimizer_blt = torch.optim.Adam(BLT(input_size=X.shape[1], output_size=1).parameters(), lr=0.0001)
# optimizer_blt_ft = torch.optim.Adam(BLT_FT(input_size=X.shape[1], output_size=1).parameters(), lr=0.0001)
# optimizer_hybrid_dl = torch.optim.Adam(HybridDL(input_size=X.shape[1], output_size=1).parameters(), lr=0.0001)

# Train each model separately
for epoch in range(10):  # Adjust number of epochs as needed
    print(f"Epoch {epoch+1}/10")
    for i, (inputs, labels) in enumerate(train_loader):
        # Train BLT model
        optimizer_blt.zero_grad()
        outputs = BLT(input_size=X.shape[1], output_size=1)(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer_blt.step()
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{len(train_loader)}: Loss {loss.item():.4f}")

        # # Train BLT-FT model
        # optimizer_blt_ft.zero_grad()
        # outputs = BLT_FT(input_size=X.shape[1], output_size=1)(inputs)
        # loss = criterion(outputs, labels.unsqueeze(1))
        # loss.backward()
        # optimizer_blt_ft.step()
        # if (i+1) % 10 == 0:
        #     print(f"Iteration {i+1}/{len(train_loader)}: Loss {loss.item():.4f}")

        # Train HybridDL model
        # optimizer_hybrid_dl.zero_grad()
        # outputs = HybridDL(input_size=X.shape[1], output_size=1)(inputs)
        # loss = criterion(outputs, labels.unsqueeze(1))
        # loss.backward()
        # optimizer_hybrid_dl.step()
        if (i+1) % 10 == 0:
            print(f"Iteration {i+1}/{len(train_loader)}: Loss {loss.item():.4f}")

# Save each model separately
torch.save(BLT(input_size=X.shape[1], output_size=1).state_dict(), "BLT_ast_cfg.pth")
# torch.save(BLT_FT(input_size=X.shape[1], output_size=1).state_dict(), "BLT_FT.pth")
# torch.save(HybridDL(input_size=X.shape[1], output_size=1).state_dict(), "HybridDL_ast_cfg.pth")

# Evaluate each model separately
with torch.no_grad():
    blt_outputs = BLT(input_size=X.shape[1], output_size=1)(torch.tensor(X_test, dtype=torch.float))
    # blt_ft_outputs = BLT_FT(input_size=X.shape[1], output_size=1)(torch.tensor(X_test, dtype=torch.float))
    # hybrid_dl_outputs = HybridDL(input_size=X.shape[1], output_size=1)(torch.tensor(X_test, dtype=torch.float))

blt_accuracy = accuracy_score(y_test, (blt_outputs > 0.5).squeeze().numpy())
# blt_ft_accuracy = accuracy_score(y_test, (blt_ft_outputs > 0.5).squeeze().numpy())
# hybrid_dl_accuracy = accuracy_score(y_test, (hybrid_dl_outputs > 0.5).squeeze().numpy())

print("BLT Accuracy:", blt_accuracy)
# print("BLT-FT Accuracy:", blt_ft_accuracy)
# print("HybridDL Accuracy:", hybrid_dl_accuracy)

# Function to predict top files for an issue
def predict_top_files_for_issue(Bmodel, tokenizer, issue, file_paths, top_n=10):
    # Vectorize the issue
    inputs = tokenizer(issue, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    vectorized_issue = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    # Concatenate issue vector with each Java file vector and predict probabilities
    concatenated_vectors = [np.concatenate((vectorized_issue, vectorized_java), axis=0) 
                            for vectorized_java in vectorized_java_code]
    concatenated_tensors = torch.tensor(concatenated_vectors, dtype=torch.float)
    with torch.no_grad():
        probabilities = Bmodel(concatenated_tensors).squeeze().numpy()

    # Sort files based on probabilities and return top n
    top_indices = np.argsort(probabilities)[::-1][:top_n]
    top_files = [file_paths[i] for i in top_indices]
    top_probabilities = [probabilities[i] for i in top_indices]

    return top_files, top_probabilities

# Function to calculate MAP and MRR
def calculate_map_mrr(Bmodel, tokenizer, issues, file_paths, y_values):
    unique_issues = list(set(issues))
    all_aps = []
    all_rrs = []
    total_issues = len(unique_issues)
    
    for idx, issue_text in enumerate(unique_issues):
        print(f"Calculating MAP and MRR for issue {idx + 1}/{total_issues}: '{issue_text}'")
        relevant_files = [file_paths[i] for i, issue in enumerate(issues) if issue == issue_text and y_values[i] == 1]
        predicted_files, predicted_probabilities = predict_top_files_for_issue(Bmodel, tokenizer, issue_text, file_paths, top_n=200)

        # Calculate Average Precision
        ap = 0.0
        num_correct = 0
        for j, file in enumerate(predicted_files):
            if file in relevant_files:
                num_correct += 1
                ap += num_correct / (j + 1)
        if len(relevant_files)!=0:
            ap /= len(relevant_files)
            all_aps.append(ap)
        else:
            all_aps.append(0.0)

        # Calculate Reciprocal Rank
        rr = 0.0
        for j, file in enumerate(predicted_files):
            if file in relevant_files:
                rr = 1 / (j + 1)
                break
        all_rrs.append(rr)

    map_score = np.mean(all_aps)
    mrr_score = np.mean(all_rrs)
    return map_score, mrr_score

blt_map, blt_mrr = calculate_map_mrr(BLT(input_size=X.shape[1], output_size=1), tokenizer, bug_Data['content'].tolist(), bug_Data["filename"].tolist(), bug_Data['y_values'].tolist())
# blt_ft_map, blt_ft_mrr = calculate_map_mrr(BLT_FT(input_size=X.shape[1], output_size=1), tokenizer, bug_Data['content'].tolist(), bug_Data["filename"].tolist(), bug_Data['y_values'].tolist())
# hybrid_dl_map, hybrid_dl_mrr = calculate_map_mrr(HybridDL(input_size=X.shape[1], output_size=1), tokenizer, bug_Data['content'].tolist(), bug_Data["filename"].tolist(), bug_Data['y_values'].tolist())

print("BLT MAP CFG:", blt_map)
print("BLT MRR CFG:", blt_mrr)
# print("BLT-FT MAP:", blt_ft_map)
# print("BLT-FT MRR:", blt_ft_mrr)
# print("HybridDL MAP CFG:", hybrid_dl_map)
# print("HybridDL MRR CFG:", hybrid_dl_mrr)

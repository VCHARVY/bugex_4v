import pandas as pd
import os
import requests
import base64

df = pd.read_excel('/home/CS21B025/bugex_4v/OG_dataset/AspectJ.xlsx')
list_of_file = df['files']
print(list_of_file[0])
List_of_string = list_of_file[0].split()




GITHUB_API_URL = "https://api.github.com"
REPO_OWNER = "eclipse-aspectj"  
REPO_NAME = "aspectj"  
  #tk

def get_commit_details(commit_sha):
    """
    Get details of a specific commit, including the modified files.
    """
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/commits/{commit_sha}"
    headers = {
        "Authorization": f"token {ACCESS_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching commit details: {response.status_code}")
        return None

def file_exists_in_commit(commit_sha, file_path):
    """
    Check if a file exists in a specific commit.
    """
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}?ref={commit_sha}"
    headers = {
        "Authorization": f"token {ACCESS_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    
   
    if response.status_code == 200:
        return True
    return False

def get_file_content_at_commit(commit_sha, file_path):
    """
    Get the content of a file at a specific commit using the GitHub API.
    """
    url = f"{GITHUB_API_URL}/repos/{REPO_OWNER}/{REPO_NAME}/contents/{file_path}?ref={commit_sha}"
    headers = {
        "Authorization": f"token {ACCESS_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content_data = response.json()
        return content_data['content']  
    else:
        print(f"Error fetching file content: {response.status_code}")
        return None

def find_previous_commit_with_file(commit_sha, file_path):
    """
    Traverse back in history to find the previous commit where the file exists.
    """
    current_commit = commit_sha
    while current_commit:
        print(current_commit)
        commit_details = get_commit_details(current_commit)
        if not commit_details:
            break
        
      
        parent_commit_sha = commit_details['parents'][0]['sha'] if commit_details['parents'] else None
        if parent_commit_sha and file_exists_in_commit(parent_commit_sha, file_path):
            return parent_commit_sha  
        else:
            return current_commit   


    """
    Find the previous commit that contains the given file.
    """
#     url = f'https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/commits?path={file_path}'
#     # Make the API call
#     response = requests.get(url)

# # Check if the request was successful
#     if response.status_code == 200:
#         commits_data = response.json()

#     # Find the current commit in the list
#         previous_commit_sha = None
#         for i, commit in enumerate(commits_data):
#             print(commit['sha'])
#             if commit['sha'].startswith(current_commit):
#             # Get the commit before the current one (if it exists)
#                 if i + 1 < len(commits_data):
#                     previous_commit_sha = commits_data[i + 1]['sha']
#                     break

#         if previous_commit_sha:
#             print(f"The previous commit that changed {file_path} is {previous_commit_sha}")
#             return previous_commit_sha
#         else:
#             print(f"No previous commit found for {file_path} before {current_commit}")
#             return current_commit
#     else:
#         print(f"Error: {response.status_code} - {response.json().get('message')}")





def create_commit_directory(base_directory, commit_sha):
    """
    Create a directory for the given commit if it doesn't exist.
    """
    commit_dir = os.path.join(base_directory, commit_sha)
    if not os.path.exists(commit_dir):
        os.makedirs(commit_dir)
    return commit_dir

def write_to_file(commit_dir, file_path, content):
    """
    Write the content of a Java file to the appropriate path within the commit directory.
    """
    # Extract the file name from the file path
    file_name = os.path.basename(file_path)

    # Ensure the subdirectories (if any) in the file path are created
    sub_directory = os.path.dirname(file_path)
    full_directory_path = os.path.join(commit_dir, sub_directory)

    if not os.path.exists(full_directory_path):
        os.makedirs(full_directory_path)

    # Define the full path for the Java file
    full_file_path = os.path.join(full_directory_path, file_name)

    # Write the content to the Java file
    with open(full_file_path, 'w') as java_file:
        java_file.write(content)

    print(f"Written to: {full_file_path}")


def main():
    commit_sha = "dd88d21" #input("Enter the commit SHA: ")
    for file in List_of_string:
        file_path = file #input("Enter the file path: ")
    
    # Find the commit where the file last existed
        previous_commit_sha = find_previous_commit_with_file(commit_sha, file_path)
        print(file_path+previous_commit_sha)
        if previous_commit_sha:
            print(f"File {file_path} exists in commit {previous_commit_sha}. Retrieving content...")
            file_content = get_file_content_at_commit(previous_commit_sha, file_path)
            if file_content:
                decoded_bytes = base64.b64decode(file_content)
                decoded_string = decoded_bytes.decode('utf-8')
                File_name=file_path
                commit_dir=create_commit_directory("/home/CS21B025/bugex_4v/OG_dataset",commit_sha)
                File_name=File_name.replace('/', '_')
               
                write_to_file(commit_dir,File_name, decoded_string)
# Output the decoded string
                #print(decoded_string+"\n")
           
        else:
            print(f"File {file_path} does not exist in the commit history before {commit_sha}.")

if __name__ == "__main__":
    main()



# import os

# def write_to_file(file_name, content):
#     # Extract the directory path from the file name
#     directory = os.path.dirname(file_name)
    
#     # If the directory does not exist, create it
#     if directory and not os.path.exists(directory):
#         os.makedirs(directory)
    
#     # Write the content to the file
#     with open(file_name, 'w') as file:
#         file.write(content)
#     print(f"File '{file_name}' has been written successfully.")

# # Example usage
# commit_sha = "dd88d21"
# file_path = "org.aspectj.ajdt.core/src/org/aspectj/ajdt/internal/core/builder/AjState.java"
# content = "This is the content of the file."

# write_to_file(commit_sha + file_path, content)

























# import subprocess

# def get_modified_files(commit_id):
#     """
#     Returns the list of files modified in a given commit
#     """
#     # Run 'git diff-tree' to get the list of modified files in the commit
#     try:
#         result = subprocess.run(
#             ['git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_id],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         # The output is a newline-separated list of files
#         modified_files = result.stdout.strip().split('\n')
#         return modified_files
#     except subprocess.CalledProcessError as e:
#         print(f"Error getting modified files for commit {commit_id}: {e}")
#         return []

# def get_file_content_before_commit(commit_id, file_path):
#     """
#     Get the content of a file before the specified commit
#     """
#     try:
#         # 'git show' to get the file content at the commit parent
#         result = subprocess.run(
#             ['git', 'show', f'{commit_id}~1:{file_path}'],
#             capture_output=True,
#             text=True,
#             check=True
#         )
#         return result.stdout
#     except subprocess.CalledProcessError as e:
#         print(f"Error getting content of {file_path} before commit {commit_id}: {e}")
#         return None

# def main():
#     commit_id = input("Enter the commit ID: ")
    
#     # Step 1: Get the list of files modified in the commit
#     modified_files = get_modified_files(commit_id)
    
#     if not modified_files:
#         print("No files modified in the commit or failed to retrieve them.")
#         return
    
#     print(f"Files modified in commit {commit_id}:")
#     for file in modified_files:
#         print(file)
    
#     # Step 2: Retrieve and print the content of each modified file before the commit
#     for file_path in modified_files:
#         print(f"\nContent of {file_path} before commit {commit_id}:")
#         file_content = get_file_content_before_commit(commit_id, file_path)
#         if file_content:
#             print(file_content)

# if __name__ == "__main__":
#     main()
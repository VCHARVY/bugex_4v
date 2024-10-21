from github import Github
import time
import pandas as pd
import requests
import os


def get_previous_commit_id(current_commit_id, repository_name, access_token):
    api_url = f"https://api.github.com/repos/{repository_name}/commits/{current_commit_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        commit_data = response.json()
        if "parents" in commit_data and len(commit_data["parents"]) > 0:

            previous_commit_id = commit_data["parents"][0]["sha"]
            return previous_commit_id
        else:
            print("No parents found in the commit data.")
            return None
    else:
        print(f"Failed to get commit data. Status code: {response.status_code}")
        return None
    

def get_files_changed(commit_id, repository_name, access_token):
    api_url = f"https://api.github.com/repos/{repository_name}/commits/{commit_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        commit_data = response.json()
        files_changed = [file["filename"] for file in commit_data["files"]]
        return files_changed
    else:
        print(f"Failed to get files changed. Status code: {response.status_code}")
        return []



def process_commit(current_commit_id, repo, repository_name, access_token):
    prev_id = get_previous_commit_id(current_commit_id, repository_name, access_token)
    current_commit_files = get_files_changed(prev_id, repository_name, access_token)
    file_contents = []
    for file in current_commit_files:
        try:
            file_content = repo.get_contents(file, ref=prev_id)    
            file_contents.append({
                'filename':file,
                'content':file_content.decoded_content.decode('utf-8')})
        except:
            print(f"Failed to get file contents for {file}")
            
    # time.sleep(10)
    print(prev_id)
    return file_contents

# helper function to write 'content' into the 'filename' with directory 'root_folder+parent_directory+sub_directory'
# Output: crrates a file writes the content to it, also returns the file_path of location of file saved
def write_to_file(root_folder, parent_directory, sub_directory, file_name, content):
    if not sub_directory.endswith('.java'):
        return ""
    sub_directory_path = os.path.join(root_folder, parent_directory, sub_directory)

    # Create the file path
    file_path = os.path.join(sub_directory_path, file_name)

    # Create the parent directory and subdirectory if they don't exist
    os.makedirs(sub_directory_path, exist_ok=True)

    try:
        # Write content to the file
        with open(file_path, 'w') as f:
            f.write(content)
    except IOError as e:
        print("Error creating file:", e)
        
    return file_path

# given repository name and the dataframe of fectches all the files which were buggy in the prior to resolved commit
# Output: datarame with bug_id, y_values(buggy(T)/ Not buggy(F)), content(filepath of the file), target(line number of changed location), also it writes a *.csv file containing dataframe
def get_files_for_corresponding_bug(repository_name, access_token, bug_data):
    # Initialize GitHub instance with access token
    g = Github(access_token)
    df = []
    for i in range(len(repository_name)):
    # Get the repository
        repo=g.get_repo(repository_name[i])
        new_data = pd.DataFrame(columns = ['bug_id', 'target', 'content'])

        for _, rows in bug_data[i].iterrows():    
            if rows['result'] is None:
                continue
            try:
                results = rows['result'].split()
                files = process_commit(rows['commit'], repo, repository_name[i], access_token)
                root = "./processed_dataset"

                #
                for result in results:
                    count = 0
                    # checking for buggy files which are changed in the commit
                    for file in files:
                        line, filee = result.split(':')
                        if filee == file['filename']:
                            print(filee)
                            count += 1
                            file_path  = write_to_file(root,repository_name[i], rows['commit']+'/'+file['filename'], "common.java",file['content'])
                            if file_path !=  "":
                                new_row = pd.DataFrame({'commit':rows['commit'], 'description':rows['description'], 'summary':rows['summary'],  'filename': rows['commit']+'/'+filee, 'bug_id': [rows['bug_id']], 'y_values': [True], 'target': [int(line)], 'content': f"{file_path}"}, index=['bug_id'])
                                new_data = pd.concat([new_data, new_row], ignore_index=True)
                            break
                    if count == 0:
                        file_path = write_to_file(root, repository_name[i], rows['commit']+'/'+file['filename'], "common.java",file['content'])
                        print(file['filename'])
                        if  file_path !=  "":
                            new_row = pd.DataFrame({'commit':rows['commit'], 'description':rows['description'], 'summary':rows['summary'],'filename': rows['commit']+'/'+file['filename'], 'bug_id': rows['bug_id'], 'y_values': False, 'target': -1, 'content': f"{file_path}"}, index = ['bug_id'])
                            new_data = pd.concat([new_data, new_row], ignore_index=True)
            except  Exception as e:
                print(f"Error processing commit {rows['commit']}: {e}")
        df.append(new_data)
        print(repository_name[i])
        print("-"*30)

    co_df = pd.concat(df, ignore_index=True)
    co_df.to_csv('creation/Aspectj.csv', index=False)

    return new_data

# def get_previous_version_of_files(current_commit_id, repository_name, access_token):
#     # Initialize GitHub instance
#     g = Github(access_token)
#     repo = g.get_repo(repository_name)
    
#     # Step 1: Get the previous commit ID
#     previous_commit_id = get_previous_commit_id(current_commit_id, repository_name, access_token)
    
#     if not previous_commit_id:
#         print("Could not retrieve the previous commit ID.")
#         return None
    
#     # Step 2: Get the files changed in the current commit
#     changed_files = get_files_changed(current_commit_id, repository_name, access_token)
    
#     if not changed_files:
#         print("No files changed in the current commit.")
#         return None
    
#     previous_files_content = []
    
#     # Step 3: Fetch the previous version of each changed file
#     for file in changed_files:
#         try:
#             # Fetch the file content as it was in the previous commit
#             file_content = repo.get_contents(file, ref=previous_commit_id)
#             previous_files_content.append({
#                 'filename': file,
#                 'previous_content': file_content.decoded_content.decode('utf-8')
#             })
#         except Exception as e:
#             print(f"Failed to get file content for {file}: {e}")
    
#     return previous_files_content

repository_name = ["eclipse-aspectj/aspectj", "eclipse-birt/birt", "eclipse-platform/eclipse.platform.ui", "eclipse-jdt/eclipse.jdt.ui", "eclipse-platform/eclipse.platform.swt", "apache/tomcat"]
filename = ["AspectJ.xlsx", "Birt.xlsx", "Eclipse_Platform_UI.xlsx", "JDT.xlsx", "SWT.xlsx", "Tomcat.xlsx"]


dfs = []
for i in range(len(repository_name)):
    dfs.append(pd.read_excel(f"DataPreprocessing/OG_dataset/{filename[i]}"))

bug_data = pd.concat(dfs, ignore_index=True)
bug_data.columns.values[-1] = 'result'
print(bug_data.head())
print(len(bug_data))
print(get_files_for_corresponding_bug(repository_name, access_tokens, dfs).head())













































# from github import Github
# import requests
# import os

# def get_previous_version_of_files(current_commit_id, repository_name, access_token):
#     # Initialize GitHub instance
#     g = Github(access_token)
#     repo = g.get_repo(repository_name)
    
#     # Step 1: Get the previous commit ID
#     previous_commit_id = get_previous_commit_id(current_commit_id, repository_name, access_token)
    
#     if not previous_commit_id:
#         print("Could not retrieve the previous commit ID.")
#         return None
    
#     # Step 2: Get the files changed in the current commit
#     changed_files = get_files_changed(current_commit_id, repository_name, access_token)
    
#     if not changed_files:
#         print("No files changed in the current commit.")
#         return None
    
#     previous_files_content = []
    
#     # Step 3: Fetch the previous version of each changed file
#     for file in changed_files:
#         try:
#             # Fetch the file content as it was in the previous commit
#             file_content = repo.get_contents(file, ref=previous_commit_id)
#             previous_files_content.append({
#                 'filename': file,
#                 'previous_content': file_content.decoded_content.decode('utf-8')
#             })
#         except Exception as e:
#             print(f"Failed to get file content for {file}: {e}")
    
#     return previous_files_content

# # Example usage
# repository_name = "eclipse-aspectj/aspectj"
# current_commit_id = "YOUR_CURRENT_COMMIT_ID"
# access_token = "YOUR_ACCESS_TOKEN"

# previous_files = get_previous_version_of_files(current_commit_id, repository_name, access_token)

# # Output the results
# if previous_files:
#     for file_info in previous_files:
#         print(f"File: {file_info['filename']}")
#         print("Previous Content:")
#         print(file_info['previous_content'])
#         print("-" * 80)
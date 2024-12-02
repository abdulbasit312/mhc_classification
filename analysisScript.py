import os
import json
import sys
import random

def process_json_files_in_folders(root_folder,strip_size=500):

    # Check if the provided folder exists
    if not os.path.exists(root_folder):
        print(f"The folder '{root_folder}' does not exist.")
        return
    file_content="user_id\t\tflag\ttotal\tprct\n"
    # Iterate over all subfolders in the given folder
    for folder_name in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, folder_name)
        
        if os.path.isdir(subfolder_path):
            print(f"\nProcessing folder: {folder_name}")
            
        # Iterate through all files in the subfolder

            file_name="tweets.json"
            file_path=os.path.join(subfolder_path,file_name)

            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                dt=0
                total=0
                tweets=[]
                disorder_flag_indices=[]
                for key in data:
                    for tweet in data[key]:
                        tweets.append(tweet)
                        flag=tweet["disorder_flag"]
                        if flag==True:
                            dt+=1
                            disorder_flag_indices.append(len(tweets)-1)
                        total+=1
                prct=dt*100.0/total
                selected=[]
                if(len(tweets)<strip_size):
                    selected=tweets
                else:
                    if(dt<strip_size and prct<70):
                        selected.extend([tweets[i] for i in disorder_flag_indices])
                    else:
                        #if disorder posts is greater than 60% keep dataset conc at 60% only
                        max_disorder_posts=int(strip_size*0.7)
                        select_disorder_posts_indices = random.sample(disorder_flag_indices, max_disorder_posts)
                        selected.extend([tweets[i] for i in select_disorder_posts_indices])

                    remaining=strip_size-len(selected)
                    valid_indices = [x for x in range(len(tweets)) if x not in disorder_flag_indices]
                    # Specify the number of random indices you want to selec
                    # Randomly select the indices
                    non_disorder_posts_indices = random.sample(valid_indices, min(remaining,len(valid_indices)))
                    selected.extend([tweets[i] for i in non_disorder_posts_indices])

                compressed_file_path=os.path.join(subfolder_path,"compressed.json")
                print(f"writing to {compressed_file_path}")
                with open(compressed_file_path,"w") as json_file:
                    json.dump(selected,json_file,indent=4)
                file_content+=f"{folder_name}\t{dt}\t{total}\t{dt*100.0/total}\n"
            
    
    with open(f"/w/331/abdulbasit/{root_folder.split('/')[-1]}.txt", "w") as file:
        file.write(file_content)  # Write the string to the file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
    else:
        folder = sys.argv[1]
    folder_path=f"/w/247/abdulbasit/{folder}"
    process_json_files_in_folders(folder_path)

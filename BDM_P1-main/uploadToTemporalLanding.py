from hdfs import InsecureClient
import os
import posixpath as psp
from dotenv import load_dotenv

load_dotenv()

# Uploading everything from a local data folder (idealista, lookup tables, opendata) to a folder called temporal_landing in HDFS

local_path = os.getenv("LOCAL_DATA_PATH")
temporal_landing=os.getenv("TEMPORAL_LANDING")
host= "http://"+os.getenv("MACHINE_IP")+":"+os.getenv("MACHINE_PORT")
user_name=os.getenv("USER_NAME")
hdfs_cli = InsecureClient(host, user=user_name)

def delete_hdfs_directory(dir_name):
    if not hdfs_cli.status(dir_name, strict=False):
        print(f"The folder {dir_name} does not exist.")
        return
    for root, dirs, files in hdfs_cli.walk(dir_name):
        for file in files:
            hdfs_cli.delete(os.path.join(root, file))
    hdfs_cli.delete(dir_name, recursive=True)
    print(f"The folder {dir_name} was deleted.")


# List all files that are in local directory
def local_files(dir_name):
    all_files = []
    for root, dirs, files in os.walk(dir_name):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)
    return all_files


# List directories  in hdfs
def hdfs_files(dir_name):
    if hdfs_cli.status(dir_name, strict=False):
        print('Directory already exists, so continue')
    else:
        print(f"The folder {dir_name} does not exist.")
        hdfs_cli.makedirs(dir_name)
        print(f"The folder {dir_name} was created.")
    f_formats = []
    f_paths = [psp.join(dpath, f_name)
               for dpath, _, f_names in hdfs_cli.walk(dir_name)
               for f_name in f_names]
    return f_paths, f_formats

def progress_callback(file_name, bytes_uploaded):
    if bytes_uploaded == -1:
        print(f"Finished uploading file {file_name}")
    else:
        print(f"Uploaded {bytes_uploaded} for file {file_name}")

# uploads all the files from the local folder (idealista, lookup tables, opendata) to a folder called temporal_landing in hdfs
# first parameter is the path in the virtual machine for hdfs and the second one is the local folder where all the data is

hdfs_cli.upload(temporal_landing, local_path, progress=progress_callback,overwrite=True)

# For testing : delete hdfs directory if previously created
# delete_hdfs_directory('/user/bdm/temporal_landing')


import os
import csv
import glob
import io
import sys
import subprocess
import numpy as np

def _pip(*pkgs: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

try:
    from azure.storage.blob import BlobServiceClient
    from azure.core.exceptions import ResourceNotFoundError
except ImportError:
    print("Installing azure-storage-blob...")
    _pip("azure-storage-blob")
    from azure.storage.blob import BlobServiceClient
    
from azure.core.exceptions import ResourceNotFoundError

def main():
    print("="*60)
    print("AZURE SIGLIP CHECKER & READ TESTER")
    print("="*60)
    
    connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        connection_string = input("Please enter your AZURE_STORAGE_CONNECTION_STRING: ")
    
    if not connection_string:
        print("No connection string provided. Exiting.")
        return

    # 1. Read all expected video IDs from CSV files
    data_dir = r"d:\Bachelor\Sinhviennam3\Semes2\NLP_UD\Project\src\NLP4B\data-processing\data"
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory does not exist -> {data_dir}")
        return
        
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    expected_video_ids = set()
    for file in csv_files:
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip header
            for row in reader:
                if row:
                    expected_video_ids.add(row[0].strip())
                    
    print(f"Total expected video IDs from CSVs in data directory: {len(expected_video_ids)}\n")

    # 2. Connect to Azure and check existing
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    # Based on azure_migrator.py, embeddings are in the 'embeddings' container
    container_name = "embeddings"
    container_client = blob_service_client.get_container_client(container_name)
    
    print(f"Checking Azure container '{container_name}'...")
    try:
        blobs = container_client.list_blobs()
    except ResourceNotFoundError:
        print(f"Error: Container '{container_name}' does not exist.")
        return
    except Exception as e:
        print(f"Failed to access container '{container_name}': {e}")
        return

    # SigLIP embeddings are stored as video_id.npy in 'embeddings' container
    found_video_ids = set()
    sample_blob_name = None
    
    for blob in blobs:
        # According to embedding.py, it's saved as {video_id}.npy, and sometimes {video_id}/filename.npy via migrator
        # We handle both flat and nested (e.g. video_id/video_id.npy) formats
        if blob.name.endswith('.npy'):
            filename = os.path.basename(blob.name) # Extracts just the filename part
            vid_id = filename[:-4] # Remove .npy
            
            # The script embedding.py generates video_id.npy.  
            found_video_ids.add(vid_id)
            if not sample_blob_name:
                sample_blob_name = blob.name

    print(f"Total `.npy` files found in container: {len(found_video_ids)}")
    
    # 3. Compare sets
    missing = expected_video_ids - found_video_ids
    if missing:
        print(f"❌ MISSING {len(missing)} video IDs:")
        for m in list(missing)[:20]:
            print(f" - {m}")
        if len(missing) > 20:
            print(f" ... and {len(missing)-20} more.")
    else:
        print("✅ ALL EXPECTED VIDEO IDs ARE PRESENT IN AZURE!\n")

    # 4. Test reading from Azure
    print("\n" + "-"*40)
    print("TEST READING SIGLIP `.npy` FROM AZURE")
    print("-" * 40)
    if sample_blob_name:
        print(f"Testing read on blob: {sample_blob_name}")
        try:
            blob_client = container_client.get_blob_client(sample_blob_name)
            
            # Download blob content into memory
            download_stream = blob_client.download_blob()
            data_bytes = download_stream.readall()
            
            # Load into numpy array from BytesIO
            arr = np.load(io.BytesIO(data_bytes))
            
            print(f"✅ Successfully loaded '{sample_blob_name}' directly from Azure into memory!")
            print(f"   -> Array shape: {arr.shape}  |  dtype: {arr.dtype}  |  dim: 1152 expected")
            
            if len(arr.shape) == 2 and arr.shape[1] == 1152:
                print("   -> Dimensions match SigLIP (N, 1152).")
            else:
                print("   -> Warning: Dimensions don't match typical SigLIP output (N, 1152).")
                
        except Exception as e:
            print(f"❌ Failed to read from Azure: {e}")
    else:
        print("No .npy files found to test.")

if __name__ == "__main__":
    main()

import os
import sys
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def download_file(url: str, filename: str, download_dir: str):
    """Download a file if it does not already exist."""

    try:
        filepath = os.path.join(download_dir, filename)

        # Check if the file already exists and its size matches the remote file
        if os.path.isfile(filepath):
            local_file_size = os.path.getsize(filepath)
            remote_file_size = int(requests.head(url).headers.get("content-length", 0))
            if local_file_size == remote_file_size:
                print(f"{filepath} already exists. Skipping download.")
                return
            else:
                print(f"{filepath} already exists, but its size does not match. Redownloading.")

        print(f"Downloading {filename} from {url}")

        # Start download, stream=True allows for progress tracking
        response = requests.get(url, stream=True)

        # Check if request was successful
        response.raise_for_status()

        # Create progress bar
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(
            total=total_size,
            unit='iB',
            unit_scale=True,
            ncols=70,
            file=sys.stdout
        )

        # Write response content to file using a buffer
        with open(filepath, 'wb') as f:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                progress_bar.update(len(data))  # Update progress bar

        # Close progress bar
        progress_bar.close()

        # Error handling for incomplete downloads
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR, something went wrong while downloading")
            raise Exception()

    except Exception as e:
        print(f"An error occurred: {e}")

def download_files_parallel(config, download_dir, num_threads=4):
    """Download files in parallel using ThreadPoolExecutor."""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        for url, filename in config.items():
            futures.append(executor.submit(download_file, url, filename, download_dir))

        for future in futures:
            future.result()

def main():
    """Main function to download files from URLs in a config file."""

    # Get JSON config file path
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(script_dir, "download_models.json")

    # Set download directory
    download_dir = "checkpoints"
    os.makedirs(download_dir, exist_ok=True)

    # Load URL and filenames from JSON
    with open(config_file_path, "r") as f:
        config = json.load(f)

    # Download files in parallel
    download_files_parallel(config, download_dir, num_threads=4)

if __name__ == "__main__":
    main()

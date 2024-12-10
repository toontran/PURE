import os
import requests
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import tarfile

# ImageNet API endpoints
IMAGENET_API_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
IMAGENET_SYNSET_LIST_URL = "http://www.image-net.org/api/text/imagenet.synset.obtain_synset_list"

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def download_imagenet_subset(output_dir, samples_per_class=10, num_classes=10):
    os.makedirs(output_dir, exist_ok=True)

    # Get list of all synsets
    response = requests.get(IMAGENET_SYNSET_LIST_URL)
    all_synsets = response.text.split()

    # Randomly select synsets
    selected_synsets = random.sample(all_synsets, num_classes)

    for synset in tqdm(selected_synsets, desc="Downloading classes"):
        # Create directory for this class
        class_dir = os.path.join(output_dir, synset)
        os.makedirs(class_dir, exist_ok=True)

        # Get image URLs for this synset
        response = requests.get(IMAGENET_API_URL + synset)
        urls = response.text.strip().split('\n')

        # Randomly select URLs
        selected_urls = random.sample(urls, min(samples_per_class, len(urls)))

        # Download images
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, url in enumerate(selected_urls):
                save_path = os.path.join(class_dir, f"{i}.jpg")
                futures.append(executor.submit(download_image, url, save_path))

            for future in futures:
                future.result()

def create_imagenet_subset_tarball(input_dir, output_tarball):
    with tarfile.open(output_tarball, "w:gz") as tar:
        tar.add(input_dir, arcname=os.path.basename(input_dir))

# Usage
if __name__ == "__main__":
    output_dir = "imagenet_subset"
    download_imagenet_subset(output_dir, samples_per_class=10, num_classes=10)
    create_imagenet_subset_tarball(output_dir, "imagenet_subset.tar.gz")
    print("Download complete. Dataset saved as imagenet_subset.tar.gz")


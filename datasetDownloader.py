import os
import random
from duckduckgo_search import DDGS
import urllib.request
from pathlib import Path
from PIL import Image
import time

def download_images(query, label, max_results=50, output_dir='./DogDetector/dataset/bulk data'):
    save_path = Path(output_dir) / label
    save_path.mkdir(parents=True, exist_ok=True)
    time.sleep(random.randint(1,2))
    #search
    print(f"Searching for '{query}'...")
    results = DDGS().images(query, max_results=max_results)
    #starts number from the number of images already there
    images_in_folder = len(list(save_path.glob("*.jpg")))
    success_count = images_in_folder
    for i, results in enumerate(results):
        if success_count >= max_results + images_in_folder:
            break
        url = results['image']
        try:
            #download image to needed folder and name them
            file_path = save_path / f"{label}_{success_count}.jpg"

            #add user-agent(anti bot-detection) and timeout
            headers = {'User-Agent': 'Mozilla/5.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                with open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
            #edit their resolution
            with Image.open(file_path) as img:
                width, height = img.size
                #too small
                if width < 256 or height < 256:
                    print(f"skipping {file_path} too small: {width}x{height}")
                    file_path.unlink()
                #too big
                else:
                    img_resized = img.resize((256,256))
                    img_resized.save(file_path)
                    success_count +=1
        except Exception as e:
            print(f"Failed to donwload {url}: {e}")

#Dog download
dog_queries = ["dog", "puppy", "golden retriever", "shih tzu", "beagle", "labrador"]
for query in dog_queries:
    download_images(query, label="dog", max_results=50)

#No dog download
no_dog_queries = ["city street", "kitchen", "forest", "living room", "beach", "mountains"]
for query in no_dog_queries:
    download_images(query, label="no_dog", max_results=50)


print("Download Complete")
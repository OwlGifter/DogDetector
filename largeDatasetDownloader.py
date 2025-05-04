import os
import random
import urllib.request
import tarfile
from pathlib import Path
from PIL import Image
import time
import shutil


#Dog image download
# # Define the URL and download path
# url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
# download_path = "./DogDetector/images.tar"
# extract_path = "./DogDetector/dataset/bulk data/dog"

# # Create the directory if it doesn't exist
# os.makedirs(extract_path, exist_ok=True)

# # Download the dataset
# print("Downloading Stanford Dogs Dataset...")
# urllib.request.urlretrieve(url, download_path)
# print("Download complete.")

# # Extract the dataset
# print("Extracting the dataset...")
# with tarfile.open(download_path) as tar:
#     tar.extractall(path=extract_path)
# print("Extraction complete.")

# # Remove the tar file to save space
# os.remove(download_path)

# target_image_count=18000
# for img_path in Path("./DogDetector/dataset/bulk data/dog").glob("*"):
#     if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
#                 continue
#     try:
#         with Image.open(img_path) as img:
#             img = img.convert("RGB")
#             img_resized = img.resize((256, 256))
#             out_path = img_path
#             img_resized.save(f"./DogDetector/dataset/bulk data/no_dog/{img_path.name}")
#             total_images += 1
#             print(f"✅ Saved ({total_images}): {out_path.name}")
#             if total_images >= target_image_count:
#                 print("🎯 Reached 18,000 images!")
#                 break
#     except Exception as e:
#         skipped_images += 1
#         print(f"⚠️ Skipped ({skipped_images}): {img_path.name} — {e}")



# List of non-dog ImageNet synsets (safe categories)
non_dog_synsets = [
    "n02958343", "n03085013", "n03770679", "n02691156", "n04392985", "n04070727",
    "n04254680", "n03642806", "n04152593", "n03207941", "n03100240", "n04285008",
    "n03769881", "n02951358", "n03272010", "n04086273", "n04286575", "n03250847",
    "n03417042", "n03444034", "n02782093", "n03773504", "n02814533", "n02808440",
    "n04465501", "n03109150", "n02791270", "n02892767", "n03676483", "n03498962"
]

# Stop when this many resized images are saved
target_image_count = 18000

# Paths
base_dir = Path("./ImageNet")
raw_dir = base_dir / "no_dog_raw"
resized_dir = base_dir / "no_dog_resized"
raw_dir.mkdir(parents=True, exist_ok=True)
resized_dir.mkdir(parents=True, exist_ok=True)

# Counters
total_images = 0
skipped_images = 0

# Start download/resizing
for wnid in non_dog_synsets:
    if total_images >= target_image_count:
        break

    print(f"\n📥 Downloading synset: {wnid}")
    url = f"https://image-net.org/data/winter21_whole/{wnid}.tar"
    tar_path = raw_dir / f"{wnid}.tar"
    extract_path = raw_dir / wnid
    try:
        # Download tar file
        urllib.request.urlretrieve(url, tar_path)
        print("✅ Downloaded.")

        # Extract contents
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_path)
        print("✅ Extracted.")

        # Process each image
        for img_path in extract_path.glob("*"):
            if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                continue
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img_resized = img.resize((256, 256))
                    out_path = resized_dir / f"{wnid}_{img_path.name}"
                    img_resized.save(f"./DogDetector/dataset/bulk data/no_dog/{wnid}_{img_path.name}")
                    total_images += 1
                    print(f"✅ Saved ({total_images}): {out_path.name}")
                    if total_images >= target_image_count:
                        print("🎯 Reached 18,000 images!")
                        break
            except Exception as e:
                skipped_images += 1
                print(f"⚠️ Skipped ({skipped_images}): {img_path.name} — {e}")

        # Clean up
        os.remove(tar_path)
        shutil.rmtree(extract_path)

    except Exception as e:
        print(f"❌ Failed to process {wnid}: {e}")

print(f"\n✅ Finished!")
print(f"📊 Total images resized: {total_images}")
print(f"⚠️ Total images skipped: {skipped_images}")
print(f"📂 Output folder: {resized_dir.resolve()}")

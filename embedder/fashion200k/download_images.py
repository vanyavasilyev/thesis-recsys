import requests
import os
import tqdm

FASHION200K_DIR = ""

with open(f"{FASHION200K_DIR}/image_urls.txt", "r") as f:
    urls_data = f.read()

loaded_paths = []
for line in tqdm.tqdm(urls_data.split("\n")):
    path, url = line.split()
    path = os.path.join(FASHION200K_DIR, path)
    if os.path.exists(path):
        loaded_paths.append(path)
        continue
    try:
        r = requests.get(url)
    except requests.ConnectionError:
        continue
    if r.ok:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(r.content)
        loaded_paths.append(path)

with open(f"{FASHION200K_DIR}/loaded_paths.txt", "w") as f:
    f.write("\n".join(loaded_paths))

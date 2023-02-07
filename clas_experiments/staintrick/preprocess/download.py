from tqdm import tqdm
import requests
from typing import Optional


def download_data(
    url: str,
    zip_path: str,
    file_path: Optional[str] = None,
):
    """
    Download data (zip files) from url link.

    url (str): url link.
    zip_path (str): path to the zip file.
    file_path (str): path to the unziped file.
    """

    if file_path is None:
        file_path = zip_path.split(".zip")[0]

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024**2  # MB
    progress_bar = tqdm(total=total_size_in_bytes, unit="MB", unit_scale=True)
    with open(zip_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
        return


# url = 'https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K-NONORM.zip?download=1'
# path = '/home/yiqing/data/NCT-CRC-HE-100K-NONORM.zip'

# url = 'https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip?download=1'
# path = '/home/yiqing/data/CRC-VAL-HE-7K.zip'


url = "https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip?download=1"
path = "/home/yiqing/data/NCT-CRC-HE-100K.zip"
download_data(url, path)

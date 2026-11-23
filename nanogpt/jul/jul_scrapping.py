"""
Script to parse all Jul lyrics ever.

Install dependencies
```bash
!uv pip install webdriver_manager selenium codecarbon tqdm.rich
```

Install brave-browser on wsl
```bash
sudo apt update && sudo apt install curl
sudo curl -fsSLo /usr/share/keyrings/brave-browser-archive-keyring.gpg https://brave-browser-apt-release.s3.brave.com/brave-browser-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/brave-browser-archive-keyring.gpg] https://brave-browser-apt-release.s3.brave.com/ stable main" | sudo tee /etc/apt/sources.list.d/brave-browser-release.list
sudo apt update && sudo apt install brave-browser
```
"""


import os
import time
import pickle
from typing import List

import numpy as np
from tqdm.rich import tqdm
from codecarbon import EmissionsTracker
from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException


def get_options_services(sandbox: bool = True):
    """Get the driver object, using Brave browser."""
    # Option object
    options = Options()
    options.binary_location = "/usr/bin/brave-browser"
    options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    if not sandbox:
        options.add_argument("--no-sandbox")  # sandbox can be removed on WSL2
    # Anti-detection settings
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    # Service object
    service = Service(ChromeDriverManager().install())
    # Return
    return options, service

def get_song_list(
        driver: WebDriver,
        url: str = "https://www.azlyrics.com/j/jul.html") -> List[str]:
    """Get a list of all url songs on Jul's webpage (doesn't work on any webpage)."""
    # Get webpage
    driver.get(url)
    # Get song links
    all_div = driver.find_elements(By.CLASS_NAME, "listalbum-item")
    song_links = []
    for div in tqdm(all_div):
        try:
            song = div.find_element(By.TAG_NAME, "a")
            song_links += [song.get_attribute("href")]
        except NoSuchElementException:
            print("Exception encountered.")
    return song_links

def read_song_links(song_link_path: str) -> List[str]:
    """Read the song links pickle file."""
    with open(song_link_path, "rb") as f:
        song_links = pickle.load(f)
    return song_links

def write_song_links(song_links: List[str], song_link_path: str):
    """Write in the song links pickle file."""
    with open(song_link_path, "wb") as f:
        pickle.dump(song_links, f)

def extract_lyrics(driver: WebDriver, song_link: str) -> bool:
    """Extract Jul lyrics from the the URL of a webpage."""
    # Get webpage
    driver.get(song_link)
    # Get all <div> and <b>
    try:
        big_div = driver.find_element(By.CSS_SELECTOR,
                                      "div.col-xs-12.col-lg-8.text-center")
        all_b = driver.find_elements(By.TAG_NAME, "b")
    except NoSuchElementException:
        print(">>> NoSuchElementException.")
        time.sleep(20)
        return False
    # Get title
    all_b_texts = [b.text for b in all_b if b.text != '']
    title = all_b_texts[1]
    # Handle empty edge case
    if big_div.text == "":
        print(">>> `big_div` is empty.")
        return None
    if len(all_b_texts) < 2:
        print(f">>> {len(all_b_texts) = } < 2.")
        print(f"{all_b_texts = }")
        title = all_b_texts[0] if len(all_b_texts) == 1 else "TITLE NOT FOUND"
    # Get lyrics
    children = big_div.find_elements(
        By.XPATH,
        "./div[not(contains(@class, 'songlist-panel'))]"
    )  # direct children
    children_filtered = [child for child in children if child.text != '']
    num_children = [len(child.text) for child in children_filtered]
    longest_idx = np.argmax(np.array(num_children))
    assert longest_idx in [2, 3], "Longest string isn't the 2nd/3rd one"
    lyrics = children_filtered[longest_idx].text.replace("\n\n", '\n')
    # Write lyrics
    with open("nanogpt/jul.txt", "a") as f:
        f.write(title + '\n\n')
        f.write(lyrics + '\n\n')
    print(f"[FINISHED] {title}")
    return True


with EmissionsTracker(
    output_dir="/mnt/c/Users/Emile/.ecocarbonai/",
    project_name="nanoGPT",
    experiment_id="Jul dataset - run 005",
    log_level="warning",
    ) as tracker:

    # Driver object
    options, service = get_options_services()
    with webdriver.Chrome(service=service, options=options) as driver:

        # Get url song list
        song_link_path = "nanogpt/song_links.pkl"
        if os.path.exists(song_link_path):
            song_links = read_song_links(song_link_path)
        else:
            song_links = get_song_list(driver, "https://www.azlyrics.com/j/jul.html")
            write_song_links(song_links, song_link_path)
        print("Number of songs retrieved:", len(song_links))

        # Get lyrics for each song
        while song_links != []:
            song_link = song_links.pop()
            completed = extract_lyrics(driver, song_link)
            if completed:
                write_song_links(song_links, song_link_path)
                print(f"[REMOVED] {song_link}")
            else:
                song_links.append(song_link)
            time.sleep(5)  # free memory space + avoid DoS

    print("Ecocarbon run finished.")
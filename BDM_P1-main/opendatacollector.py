import os
import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup
import requests
from dotenv import load_dotenv

load_dotenv()


## Downloads CSV files from opendata and saves them to a local directory

async def get_csv_links(url):
    browser = await launch()
    page = await browser.newPage()
    await page.goto(url)
    soup = BeautifulSoup(await page.content(), 'html.parser')
    csv_links = [f"{url}{item['href']}" for item in soup.select("a[href$='lloguer_preu_trim.csv']")]
    await browser.close()
    return csv_links


if __name__ == '__main__':
    url = os.getenv("OPEN_DATA_URL")
    path = os.getenv("LOCAL_PATH_DIRECTORY")
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created new directory " + path)
    print("Connecting...")
    loop = asyncio.get_event_loop()
    csv_links = loop.run_until_complete(get_csv_links(url))
    for link in csv_links:
        data = link.split("//")
        url = 'https://' + data[-1]
        response = requests.get(url)
        file_name = data[-1].split('/')[-1]
        all_path = path + file_name
        if os.path.exists(all_path):
            print(f"The file {all_path} already exists")
            continue
        with open(all_path, 'wb') as f:
            print(f"Downloading {url}")
            f.write(response.content)

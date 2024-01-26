import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd 
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException


# Set up Selenium with ChromeDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode (without opening a GUI window)
service = Service(ChromeDriverManager().install())

driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the page to scrape
url = 'https://www.economie.gouv.fr/recherche-resultat?search_api_views_fulltext=IA&page=0'


import undetected_chromedriver as uc
from selenium import webdriver

options = webdriver.ChromeOptions() 
options.add_argument("start-maximized")
driver = uc.Chrome(options=options)


# Navigate to the page
driver.get(url)

# Wait for the necessary elements to load, if needed
# driver.implicitly_wait(10)  # Waits up to 10 seconds before throwing a TimeoutException

# Set a longer timeout duration
timeout = 60  # Increase the timeout duration as needed

try:
    # Wait for a specific element that indicates the search results have loaded
    wait = WebDriverWait(driver, timeout)
    wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'specific-class')))  # Replace 'specific-class' with the actual class
except TimeoutException:
    print(f"Timed out waiting for page to load after {timeout} seconds")

# Once the page is loaded, you can use BeautifulSoup to parse the rendered HTML
soup = BeautifulSoup(driver.page_source, 'html.parser')
print(soup)


# Find the elements containing the search results
# Find the parent div that contains all the search results
parent_div = soup.find('div', class_='view-content fr-grid-row fr-grid-row--gutters')

# Check if the parent div is found
if parent_div:
    # Find all the div elements with the class 'views-row' within the parent div
    search_results = parent_div.find_all('div', class_='views-row')

    # List to hold the extracted data
    data = []

    # Extract the desired information from each search result
    for result in search_results:
        # You need to find the actual elements within each 'views-row' div
        # that contain the title and the link, for example:
        title = result.find('h2').text  # Assuming the title is in an 'h2' tag
        link = result.find('a')['href']  # Assuming the link is in an 'a' tag
        # Add the result to the data list as a dictionary
        data.append({
            'title': title,
            'link': link
        })

# Close the browser window
driver.quit()

data = ()

# Create a DataFrame with the extracted data
df = pd.DataFrame(data)

# Display the DataFrame
print(df)
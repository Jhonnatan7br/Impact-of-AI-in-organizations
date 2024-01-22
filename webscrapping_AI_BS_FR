from googlesearch import search
import csv
from bs4 import BeautifulSoup
import requests

# Function to scrape news articles from a Google search query
def scrape_news(search_query, num_results=250):
    articles = []

    # Perform a Google search and fetch the results
    search_results = search(search_query, num_results=num_results)

    for result in search_results:
        try:
            #response = requests.get(result, verify=False)  # Add verify=False here to ignore SSL verification to not being limited by quantity of results
            response = requests.get(result, timeout=10)  # Set a timeout value here (e.g., 10 seconds)
            response.raise_for_status()  # Check for HTTP request errors

            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.title.string if soup.title else ''
            description = soup.find('meta', attrs={'name': 'description'})
            description = description['content'] if description else ''
            pub_date = soup.find('time')  # Assuming the publication date is in a <time> element

            # Extract publication date if available
            if pub_date:
                pub_date = pub_date.get_text()
            else:
                pub_date = ''

            articles.append([title, result, description, pub_date])

        except requests.exceptions.RequestException as e:
            # Handle exceptions (e.g., SSL errors) and continue to the next URL
            print(f"Unable to fetch URL: {result}\nError: {e}")
            continue
    
    return articles

# Function to create a CSV file and store the scraped data
def create_and_save_csv(data):
    with open('Datasets/news_articles.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Link', 'Description', 'Publication Date']
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        for article in data:
            writer.writerow(article)

if __name__ == "__main__":
    # Enter your search query here
    search_query = 'AI Business France'

    # Scrape news articles from the Google search
    news_data = scrape_news(search_query)

    # Create and save a CSV file in the 'Datasets' folder
    create_and_save_csv(news_data)

    print(f"{len(news_data)} news articles scraped and saved to 'Datasets/news_articles.csv'")
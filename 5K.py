from scrapingbee import ScrapingBeeClient
from bs4 import BeautifulSoup
import csv

# Initialize the ScrapingBee client with your API key
client = ScrapingBeeClient(api_key='X431U37REXXQ2OBHB3GP32RSFZKTG88I8UAB2YUBG8ZVYHNQT4C5T3F0PL4UKKCYLNWJFOGY42GF2DZF')

def perform_google_search(query):
    # For demonstration purposes, this function returns a static list of URLs.
    # In a real application, replace this with your method of fetching search results.
    return [
        "https://www.google.com/search?q=AI%2BBusiness%2BFrance%2B(Only%2BEnglish%2Bnews)&rlz=1C1ALOY_esCO1061CO1062&oq=AI%2BBusiness%2BFrance%2B(Only%2BEnglish%2Bnews)&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBBzgwMmowajGoAgCwAgA&sourceid=chrome&ie=UTF-8",
        "https://www.google.com/search?q=Machine+Learning+Business+France(Only+English+news)&rlz=1C1ALOY_esCO1061CO1062&oq=Machine+Learning+Business+France(Only+English+news)&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBBzYwNmowajGoAgCwAgA&sourceid=chrome&ie=UTF-8",
        # Add more URLs as needed
    ]

def scrape_news(search_query, num_results=10):
    search_results = perform_google_search(search_query)

    articles = []
    for result in search_results:
        try:
            response = client.get(result, timeout=10000)  # Timeout of 10 seconds (10000 milliseconds)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                title = soup.title.string if soup.title else ''
                description = soup.find('meta', attrs={'name': 'description'})
                description = description['content'] if description else ''
                pub_date = '' # Extract the publication date using an appropriate method

                articles.append([title, result, description, pub_date])
            else:
                print(f"Error fetching URL: {result} with status code {response.status_code}")
        except Exception as e:
            print(f"Unable to fetch URL: {result} within 10 seconds\nError: {e}")
            continue

    return articles

def create_and_save_csv(data, filepath='BigDatasets/1news_articles.csv'):
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Title', 'Link', 'Description', 'Publication Date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for article in data:
            writer.writerow({'Title': article[0], 'Link': article[1], 'Description': article[2], 'Publication Date': article[3]})

if __name__ == "__main__":
    search_query = 'AI Business France (Only English news)'
    news_data = scrape_news(search_query)
    create_and_save_csv(news_data, 'BigDatasets/1news_articles.csv')
    print(f"{len(news_data)} news articles scraped and saved to 'BigDatasets/1news_articles.csv'")

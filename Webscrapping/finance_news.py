import requests
from bs4 import BeautifulSoup

def scrape_news(source_url, query):
    # Adjust the URL or parameters based on the news source and how it handles search queries
    search_url = f"{source_url}/search?q={query}"
    
    # Send a GET request to the URL
    response = requests.get(search_url)
    
    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find elements containing news articles, adjust the selector based on the website's structure
    articles = soup.findAll('article', class_='news-article-class')
    
    news_list = []
    
    for article in articles:
        title = article.find('h2').text
        link = article.find('a')['href']
        summary = article.find('p').text
        news_list.append({'title': title, 'link': link, 'summary': summary})
    
    return news_list

# Example usage
source_url = 'https://www.bloomberg.com/search?query=AI%20France'
query = 'technology'
news_articles = scrape_news(source_url, query)
for article in news_articles:
    print(f"Title: {article['title']}\nLink: {article['link']}\nSummary: {article['summary']}\n")

#%%
news_articles

import requests
from bs4 import BeautifulSoup

def fetch_search_results(query, page_number=1):
    url = f"https://www.google.com/search?q={query}&start={(page_number - 1) * 10}"
    response = requests.get(url)
    return response.text

def parse_results(html):
    soup = BeautifulSoup(html, 'html.parser')
    results = []
    for item in soup.find_all('div', class_='some-result-class'):  # Update with actual class
        title = item.find('h3').text
        link = item.find('a')['href']
        snippet = item.find('p', class_='some-snippet-class').text  # Update class
        results.append({'title': title, 'link': link, 'snippet': snippet})
    return results

def main():
    query = "apple"
    html = fetch_search_results(query)
    results = parse_results(html)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()

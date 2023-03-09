import requests

api_key = "bb7a3bdcd26443b8900b696fb92e3450"
url = "https://newsapi.org/v2/everything?q=tesla&from=2023-02-09&sortBy=publishedAt&" \
"apiKey=bb7a3bdcd26443b8900b696fb92e3450"

#make request 
request = requests.get(url)

#get dictionary with data
content = request.json()

#access articles with descriptions
for article in content["articles"]:
    print(article["title"])
    print(article["description"])
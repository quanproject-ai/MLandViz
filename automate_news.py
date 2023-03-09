import requests

api_key = "bb7a3bdcd26443b8900b696fb92e3450"
url = "https://newsapi.org/v2/everything?" \
    "q=tesla&from=2023-02-09&sortBy=publishedAt&apiKey=bb7a3bdcd26443b8900b696fb92e3450"
#Make request
request = requests.get(url)

#Get a dictionary of data
contents = request.json()

#Access the article titles and description
for i in contents['articles']:
    print (i["title"])
    print (i["description"])


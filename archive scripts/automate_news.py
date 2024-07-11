import requests
from send_emails import send_email
api_key = "bb7a3bdcd26443b8900b696fb92e3450"
url = "https://newsapi.org/v2/everything?q=tesla&from=2023-02-09&sortBy=publishedAt&" \
"apiKey=bb7a3bdcd26443b8900b696fb92e3450"

#make request 
request = requests.get(url)

#get dictionary with data
content = request.json()

#access articles with descriptions
body=""
for article in content["articles"]:
    if article["title"] is not None:
        body = body + article["title"] + "\n" + article["description"] + 2*"\n"
print (body)

body = body.encode("utf-8")
send_email(message=body)

##app may not run because the newsapi is on trial version; hence, GET function is not working#

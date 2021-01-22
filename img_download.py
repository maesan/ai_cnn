import settings
import os, sys, time
from flickrapi import FlickrAPI
from urllib.request import urlretrieve

interval = 1

# 保存するフォルダ
name = sys.argv[1]
savedir = './images/' + name

if not os.path.exists(savedir):
    os.mkdir(savedir)

flickr = FlickrAPI(settings.API_KEY, settings.SECRET_KEY, format='parsed-json')
result = flickr.photos.search(
    text = name,
    per_page = 500,
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence'
)

photos = result['photos']
print(photos)

for i, photo in enumerate(photos['photo']):
    url = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    print(filepath)
    if os.path.exists(filepath):
        continue
    urlretrieve(url, filepath)
    time.sleep(interval)


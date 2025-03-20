from google.cloud import vision
import os 
from tqdm import tqdm
import time
import sys 
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import *
from dataset_collection.scrape_utils import *
import argparse
import json
import boto3

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'/content/key.json'


def split_text(text, max_length=5000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def translate_dict(client, data_dict):
    translated_dict = {}
    for key, value in data_dict.items():
        if isinstance(value, str):
            chunks = split_text(value)
            for chunk in chunks:
                response = client.translate_text(
                    Text=chunk,
                    SourceLanguageCode="auto",
                    TargetLanguageCode="en"
                )

            translated_dict[key] = response["TranslatedText"]
        elif isinstance(value, list):
            translated_dict[key] = [translate_dict({"text": v})["text"] if isinstance(v, str) else v for v in value]
        else:
            translated_dict[key] = value

    return translated_dict

def compute_url_distance(url1,url2,threshold):
    distance = lev.distance(url1,url2)
    if distance < threshold:
        return True
    else:
        return False
def find_image_caption(soup, image_url,threshold=25):
    '''
    Retrieve the caption corresponding to an image url by searching the html in BeautifulSoup format.
    '''
    img_tag = None
    for img in soup.find_all('img'):
        src = img.get('src') or img.get('data-src') or img.get('data-original')
        if src and compute_url_distance(src, image_url, threshold):
            img_tag = img
            break
    if not img_tag:
        return "Image not found"
    figure = img_tag.find_parent('figure')
    if figure:
        figcaption = figure.find('figcaption')
        if figcaption:
            return figcaption.get_text().strip()
    for sibling in img_tag.find_next_siblings(['div', 'p','small']):
        if sibling.get_text().strip():
            return sibling.get_text().strip()
    title = img_tag.get('title')
    if title:
        return title.strip()
    # Strategy 4: Use the alt attribute of the image
    alt_text = img_tag.get('alt')
    if alt_text:
        return alt_text.strip()

    return "Caption not found"


def extract_info_trafilatura(page_url,image_url):
    try:
        headers= {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(page_url, headers=headers, timeout=(10,10))
        if response.status_code == 200:
            #Extract content with Trafilatura
            result = bare_extraction(response.text,
                                   include_images=True,
                                   include_tables=False)
            #Remove unnecessary content
            keys_to_keep = ['title','author','url',
                            'description',
                            'text','image']
            result = {key: result[key] for key in keys_to_keep if key in result}
            result['image_url'] = image_url
            # Finding the image caption
            image_caption = []
            soup = bs(response.text, 'html.parser')
            for img in image_url:
                image_caption.append(find_image_caption(soup, img))
            if result['image']:
              image_caption.append(find_image_caption(soup,result['image']))
            result['image_caption'] = image_caption
            result['url'] = page_url
            return result
        else:
            return "Failed to retrieve webpage"
    except Exception as e:
        return f"Error occurred: {e}"

def detect_web(image_path,
               how_many_queries=30):
    """
    Detects web annotations given an image.
    """
    
    client_web = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client_web.web_detection(image=image, max_results=15)
    if response.error.message:
        raise Exception(
            f"{response.error.message}\n"
            "For more info on error messages, check: https://cloud.google.com/apis/design/errors"
        )

    annotations = response.web_detection

    page_urls = []
    matching_image_urls = {}
    visual_entities = {}

    if annotations.pages_with_matching_images:
        print(
            "\n{} Pages with matching images found:".format(
                len(annotations.pages_with_matching_images)
            )
        )

        for page in annotations.pages_with_matching_images:
            page_urls.append(page.url)
            if page.full_matching_images:
                #List of image URLs for that webpage (the image can appear more than once)
                matching_image_urls[page.url] = [image.url for image in page.full_matching_images]
            else:
                matching_image_urls[page.url] = []
            if page.partial_matching_images:
                matching_image_urls[page.url] += [image.url for image in page.partial_matching_images]
    else:
        print('No matching images found for ' + image_path)
    if  annotations.web_entities:
        for entity in annotations.web_entities:
            #Collect web entities as entity-score dictionary pairs
            if entity.description:
                visual_entities[entity.description] = entity.score

    # if 'error' in annotations.keys():
    #     raise Exception(
    #         "{}\nFor more info on error messages, check: "
    #         "https://cloud.google.com/apis/design/errors".format(response['error']['message'])
    #     )

    return page_urls, matching_image_urls, visual_entities


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Collect evidence using Google Reverse Image Search.')
    parser.add_argument('--collect_google', type=int, default=0, 
                        help='Whether to collect evidence URLs with the google API. If 0, it is assumed that a file containing URLs already exists.')
    parser.add_argument('--evidence_urls', type=str, default='dataset/retrieval_results/evidence_urls.json',
                        help='Path to the list of evidence URLs to scrape. Needs to be a valid file if collect_google is set to 0.')
    parser.add_argument('--google_vision_api_key', type=str,  default= "", #Provide your own key here as default value
                        help='Your key to access the Google Vision services, including the web detection API. Only needed if collect_google is set to 1.')  
    parser.add_argument('--image_path', type=str, default='dataset/images/',
                        help='The folder where the images are stored.') 
    parser.add_argument('--raw_ris_urls_path', type=str, default='dataset/retrieval_results/ris_results.json',
                        help='The json file to store the raw RIS results.') 
    parser.add_argument('--scrape_with_trafilatura', type=int, default=1, 
                        help='Whether to scrape the evidence URLs with trafilatura. If 0, it is assumed that a file containing the scraped webpages already exists.') 
    parser.add_argument('--trafilatura_path', type=str, default='dataset/retrieval_results/trafilatura_data.json',
                        help='The json file to store the scraped trafilatura  content as a json file.')
    parser.add_argument('--apply_filtering', type=int, default=0,
                        help='If 1, remove evidence published after the source FC article. Not needed if using the default evidence set')
    parser.add_argument('--json_path', type=str, default='dataset/retrieval_results/evidence.json',
                        help='The json file to store the text evidence as a json file.')
    parser.add_argument('--max_results', type=int, default=50,
                        help='The maximum number of web-pages to collect with the web detection API.') 
    parser.add_argument('--sleep', type=int, default=3,
                        help='The waiting time between two web detection API calls') 
    

    args = parser.parse_args()
    key = os.getenv(args.google_vision_api_key)

    #Create directories if they do not exist yet
    if not 'retrieval_results'  in os.listdir('dataset/'):
        os.mkdir('dataset/retrieval_results/')

    #Google RIS
    raw_ris_results = []
    for path in tqdm(os.listdir(args.image_path)):
        urls, image_urls, vis_entities  = detect_web(args.image_path +path, args.max_results)
        
        raw_ris_results.append({'image_path':args.image_path + path, 
                                'urls': urls, 
                                'image_urls': image_urls,  
                                'visual_entities': vis_entities
                                }
        )
        time.sleep(args.sleep)
    with open(args.raw_ris_urls_path, 'w') as file:
        #Save raw results
        json.dump(raw_ris_results, file, indent=4)
    #Apply filtering to the URLs to remove content produced by FC organizations and content that is not scrapable

    
    # # selected_data = get_filtered_retrieval_results(args.raw_ris_urls_path)
    urls = [d['raw_url'] for d in raw_ris_results]
    images = [d['image_urls'] for d in raw_ris_results]

    #     #Collect results with Trafilatura
    output = []
    client = boto3.client(
    "translate",
    region_name="us-east-1",  # Change based on your AWS setup
    aws_access_key_id="AKIA2UC26RBGMWEWWKEJ",
    aws_secret_access_key="XiS1u0exNKOSrK8p/6Mow7odroSFWMNzrzggmFuI"
)
    for u in tqdm(range(len(urls)),  desc="Translating" ):
        res = extract_info_trafilatura(urls[u],images[u])
        if type(res) is dict:
            translated_dict = translate_dict(client, res)
            output.append(translated_dict)
    args.trafilatura_path
    with open("translated_data.json", "w", encoding="utf-8") as json_file:
        json.dump(output, json_file, indent=4, ensure_ascii=False)
    
    # #Save all results in a Pandas Dataframe
    # evidence_trafilatura = json.load_json(args.trafilatura_path)
    # dataset = json.load_json('dataset/train.json') + json.load_json('dataset/val.json')  + json.load_json('dataset/test.json')
    # evidence = merge_data(evidence_trafilatura, selected_data, dataset, apply_filtering=args.apply_filtering).fillna('').to_dict(orient='records')
    # # Save the list of dictionaries as a JSON file
    # with open(args.json_path, 'w') as file:
    #     json.dump(evidence, file, indent=4)

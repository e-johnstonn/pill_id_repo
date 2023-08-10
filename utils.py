import difflib
import os

import cv2
import easyocr
import numpy as np
from bs4 import BeautifulSoup


def normalize_string(s):
    return s.lower().replace(" ", "")


def create_master_dict(pills, new_images):
    for image in new_images:
        for pill in pills:
            if image.startswith(pill['image_url']):
                if 'image_urls' not in pill:
                    pill['image_urls'] = []
                pill['image_urls'].append(image)
                break

    return pills


def find_pill(imprint, pills_dict=pills):
    if imprint == '':
        return None

    normalized_imprint = normalize_string(imprint)

    pill_dict = {pill['imprint']: pill for pill in pills_dict}

    if normalized_imprint in pill_dict:
        print(normalized_imprint)
        return [pill_dict[normalized_imprint]]
    else:
        matches = difflib.get_close_matches(normalized_imprint, list(pill_dict.keys()), n=4, cutoff=.8)
        if matches:
            return [pill_dict[match] for match in matches]
        else:
            return None


def preprocess_image(image):
    image = np.array(image.convert('L'))

    inverted_img = cv2.bitwise_not(image)  # invert

    kernel = np.ones((3, 3), np.uint8)  # dilate
    dilated_img = cv2.dilate(image, kernel, iterations=1)

    kernel = np.array([[0, -1, 0],  # edge enhancement
                       [-1, 5,-1],
                       [0, -1, 0]])
    highpass_img = cv2.filter2D(image, -1, kernel)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # contrast enhancement
    contrast_inverted_img = clahe.apply(inverted_img)

    return inverted_img, dilated_img, highpass_img, contrast_inverted_img


def ocr_image(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    texts = []
    for result in results:
        text = result[1]
        texts.append(text)
    combined_text = ' '.join(texts)
    return combined_text


html_files = os.listdir('data')
html_files = [file for file in html_files if file.endswith('.html')]
joined_html_files = [f'data/{file}' for file in html_files]
pill_list = []

for html_doc in joined_html_files:

    with open(html_doc, 'r', errors='ignore') as f:
        html_doc = f.read()

    soup = BeautifulSoup(html_doc, 'html.parser')
    for div in soup.find_all('div', {'class': 'search-results-card'}):
        try:
            pill_dict = {}
            pill_dict['name'] = div.find('a', {'class': 'search-results-pillName'}).text.strip()
            pill_dict['generic'] = div.find('p', {'class': 'search-results-genericName'}).text.strip().replace(" GENERIC: ", "")

            side_container = div.find('div', {'class': 'Side-container'})
            pill_dict['strength'] = side_container.find('b', {'class': 'strength-value'}).text.strip()
            pill_dict['imprint'] = side_container.find_all('span', {'class': 'side-fields-data-left'})[0].text.strip()
            pill_dict['color'] = side_container.find('span', {'class': 'side-fields-data-right'}).text.strip()
            pill_dict['shape'] = side_container.find_all('span', {'class': 'side-fields-data-right'})[0].text.strip()
            pill_list.append(pill_dict)

        except Exception as e:
            print(e)
            continue


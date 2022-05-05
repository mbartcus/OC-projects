#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:00:21 2022
source: https://docs.microsoft.com/fr-fr/azure/cognitive-services/translator/quickstart-translator?tabs=python

@author: bartcus
"""
import argparse
import requests, uuid, json
from dotenv import load_dotenv
import os
import pandas as pd
import random
import utils as utl

parser = argparse.ArgumentParser()
parser.add_argument('--txt', help='text for detecting the language by Microsoft Azure', default='')
parser.add_argument('--labdir', help='the dataset containing the labels of the languages', default='data/Dataset_project_1_AI_Engineer/labels.csv')

parser.add_argument('--dir_txt', help='The directory for the data containing text with different languages paragraph', default='data/Dataset_project_1_AI_Engineer/x_train.csv')
parser.add_argument('--dir_lang', help='The directory for the data containing languages associated with different paragraph', default='data/Dataset_project_1_AI_Engineer/y_train.csv')

def configure():
    load_dotenv()

def get_lang_detection_azure(text_to_translate, languages_labels):
    """
    Get the language spoken by the user using Microsoft Azure
    Input:
        text_to_translate - the text to be translated
        languages_labels - data frame containing the languages with their labels - csv readed from wiki dataset

    Output:
        language - the language in English that is spoken by the user
    """
    # Add your key and endpoint
    key = os.getenv('API_KEY')

    endpoint = os.getenv('ENDPOINT')
    # Add your location, also known as region. The default is global.
    # This is required if using a Cognitive Services resource.
    location = os.getenv('LOCATION')

    path = os.getenv('PATH_DETECT')
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0'
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # You can pass more than one object in body.
    body = [{
        'text': text_to_translate
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    lang = json.loads(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': '))) # results a list of strings
    print(lang)
    lang = lang[0]['language']


    #Formating the English
    language = languages_labels.English[languages_labels['Wiki Code'] == lang].iloc[0]

    return language



def make_verification(language_data_dict):
    '''
    According to : https://1to1progress.fr/blog/2021/04/30/combien-de-langues-dans-le-monde/
    The 5 most spoken languages are:
    Anglais (1,348 milliard)
    Mandarin Chinese (1,120 milliard)
    Hindi (600 millions)
    Espagnol (543 millions)
    Arabe (247 millions)
    '''
    list_languages_most_spoken = {'zho': 'Standard Chinese', 'eng':'English',  'hin':'Hindi', 'spa':'Spanish', 'ara':'Arabic'}
    list_languages = list_languages_most_spoken
    #list_languages = {'eng':'English', 'fra':'French', 'jpn':'Japanese', 'spa':'Spanish', 'deu':'German'}
    for l in list_languages.keys():
        # randomly choise the line of the specific language form list_languages
        language_line = random.choice(utl.getKeysByValue(language_data_dict, l))
        langue = get_lang_detection_azure(text_data[language_line], languages_labels)
        print('Success: The spoken language is selected correctly') if langue==list_languages[l] else print('Fail: The spoken language is badly selected')
        print('The user is speaking {0} language!'.format(langue))


if __name__ == '__main__':
    global args
    configure()
    args = parser.parse_args()
    languages_labels = pd.read_csv(args.labdir, sep=',')

    text_to_translate = args.txt
    if len(text_to_translate)==0:
        # get the dataset from witch select languages
        print('No text is given by the user, 5 languages are selected to verify translator - Standard Chinese, English, Hindi, Espagnol, Arabe')

        text_data = utl.get_linenr_text(args.dir_txt)

        language_data_dict = utl.get_linenr_text(args.dir_lang)

        # verify if the data is well colected
        assert(len(language_data_dict)==len(language_data_dict))

        make_verification(language_data_dict)
    else:
        print('The user is speaking {0} language!'.format(get_lang_detection_azure(text_to_translate, languages_labels)))

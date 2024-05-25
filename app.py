from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests as rq
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import RegexpTokenizer
import spacy
import numpy as np
from PIL import Image
from wordcloud import WordCloud
from collections import Counter, defaultdict
import os
from tqdm import tqdm
import re

# Initialize Flask application
app = Flask(__name__)

# Load models and other resources
nltk.download("punkt")
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load("en_core_web_sm")
tokenizer = RegexpTokenizer(r'\w+')

POS_Dict = {
    'Null': 'Null', 'cont': 'cont', 'CC': 'conj', 'CD': 'n', 'DT': 'art',
    'EX': 'pron', 'FW': 'fw', 'IN': 'prep', 'JJ': 'adj', 'JJR': 'adj',
    'JJS': 'adj', 'LS': 'n', 'MD': 'v', 'NN': 'n', 'NNS': 'n', 'NNP': 'n',
    'NNPS': 'n', 'PDT': 'n', 'POS': 'pron', 'PRP': 'pron', 'PRP$': 'pron',
    'RB': 'adv', 'RBR': 'adv', 'RBS': 'adv', 'RP': 'prep', 'SYM': 'sym',
    'TO': 'to', 'UH': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v',
    'VBN': 'v', 'VBP': 'v', 'VBZ': 'v', 'WDT': 'art', 'WP': 'pronoun',
    'WP$': 'pronoun', 'WRB': 'adv'
}

POS_Dict_record = defaultdict(int)
text_Set = []

def Get_Pos_dict2(tokens, pos_feats, stopwords):
    poslist = []
    poss = nltk.pos_tag(tokens)
    for pos in poss:
        try:
            if pos[0] not in stopwords:
                if "all" in pos_feats:
                    poslist.append(f'{pos[0]}')
                    POS_Dict_record[POS_Dict[pos[1]]] += 1
                elif POS_Dict[pos[1]] in pos_feats:
                    poslist.append(f'{pos[0]}')
                    POS_Dict_record[POS_Dict[pos[1]]] += 1
        except:
            pass
    return poslist

def Get_wiki_Context(url):
    content_text = ""
    response = rq.get(url)
    bs = BeautifulSoup(response.text, 'html.parser')
    p_list = bs.find_all('p')
    for p in p_list:
        content_text += p.text.replace("\n", "")
    return content_text, "wiki"

def getWebText(web_url, pos_feats, stopwords):
    global urlfilename
    for url in tqdm(web_url, desc="Processing URLs"):
            context, filetype = Get_wiki_Context(url)
            urlfilename = url.split("/")[-1]
            sentences = nltk.sent_tokenize(context)
            for sentence in sentences:
                tokens1 = tokenizer.tokenize(sentence)
                text_Set.extend(Get_Pos_dict2(tokens1, pos_feats, stopwords))
    return context

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        urls = request.form.get('urls').split()
        image_file = request.form.get('image_file')
        stop_words = request.form.get('stop_words').split()
        pos_feats = request.form.get('pos_feats').split()
        
        text_Set.clear()
        POS_Dict_record.clear()
        
        getWebText(urls, pos_feats, stop_words)
        
        if sum(POS_Dict_record.values()) < 3:
            return render_template('index.html', message="詞彙數必須大於 3!!")
        
        mask = np.array(Image.open(os.path.join('static', 'images', image_file)))
        diction = Counter(text_Set)
        wordcloud = WordCloud(background_color="white", mask=mask)
        wordcloud.generate_from_frequencies(frequencies=diction)
        
        featfile = "_".join(pos_feats)
        savename = f"static/images/Wordcloud_Result_{urlfilename}_{featfile}.png"
        wordcloud.to_file(savename)
        
        return render_template('result.html', image_file=savename, new_window=True)
    
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file_upload' not in request.files:
        return jsonify(success=False, message='No file part')
    file = request.files['file_upload']
    if file.filename == '':
        return jsonify(success=False, message='No selected file')
    if file:
        filename = file.filename
        file_path = os.path.join('static', 'images', filename)
        file.save(file_path)
        return jsonify(success=True, filename=filename)
    return jsonify(success=False, message='File upload failed')

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(debug=True, port=80, host='0.0.0.0')

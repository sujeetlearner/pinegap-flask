from flask import Flask, render_template, request, redirect

import urllib.request
from PyPDF2 import PdfReader
import io
import re

#####

# @title Important functions and libraries installation.

import urllib.request
from PyPDF2 import PdfReader
import io


from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import pickle

import os
import requests
import json
from PyPDF2 import PdfReader
from nltk.tokenize import PunktSentenceTokenizer
import pandas as pd
import spacy
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer
import networkx as nx


import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello from Koyeb'
@app.route('/showTranscript', methods=['GET'])
def showTranscript():
    merge_algo = request.args.get('transcript')
    dependency = MergeAlgoDependency(merge_algo = merge_algo)
    transcript = dependency.merge()
    print(transcript['Sentences'][0])
    return transcript['Sentences'][0]


class MergeAlgoDependency():
    def __init__(self, merge_algo = 'windowTechnique', input_filename = "transcript.docx", output_filename = "transcript_merged.xlsx", threshold = 0.2, max_merge_len = 3, resolution = 10):
        self.merge_algo = merge_algo
        self.filename= input_filename
        self.output_filename= output_filename
        self.thres = threshold
        self.max_merge_len = max_merge_len
        self.resolution = resolution

    def merge(self):
        try:
            utilityObject = Utility(self.filename, self.output_filename, self.thres, self.max_merge_len)
            if self.merge_algo == 'windowTechnique':
                comDf = utilityObject.MergeDataFrame()
                print('windowTechnique merged')
                return comDf
            elif self.merge_algo == 'NetworkGraphMergeAlgo':
                comDf = utilityObject.MergeDataFrameUsingNetworkAlgo()
                print('NetworkGraphMergeAlgo merged')
                return comDf
        except Exception as e:
            print('Exception occurred while merging')
            return


superSet = ['Pricing','Volume','Organic Growth', 'Revenue',
'Gross Margin','Supply Chain','D&A', 'SG&A', 'Employee Expense', 'Cost Reduction', 'General Expenses', 'EBIT','Margin Change','Tax','Interest Expense', 'Leverage', 'Cash', 'Headwinds', 'Tailwinds', 'Product Innovation', 'New Product','Capital Allocation','Restructuring', 'Foreign Exchange', 'Economic','Outsourcing',
'Net Income','Free Cash Flow','Raise Guidance', 'Profitability',
'Management Change','Equity Issue','EBITDA',
'Macro','Single Country Challen', 'Geopolitical Risks','Legal/Regulatory',
'Regulatory','Tax Regulation', 'Accounting Practices', 'IP & Patents',
'ESG', 'Deflection', 'Margin Change', 'Inventory','Investments', 'Industry & Business', 'Store Openings/closing', 'Competition & Market',
'Seasonality', 'Business Drivers', 'Revenue Growth', 'EPS', 'Operating Margin', 'Partnerships', 'Marketing', 'Regional Trends', 'Outlook', 'Digital Transformation', 'Share Buybacks','Artificial Intelligence','Manufacturing', 'Data Center', 'ARM', 'X86', 'Gaming', 'CPU', 'GPU', 'Generative AI', 'Azure Cloud', 'Public Cloud', 'AWS Cloud', 'Google Cloud', 'Demand', 'Supply' , 'Share Count' ]

def TagUtility():
    def Tagger(df1, urlList1, filename):
        x = -1
        url = urlList1[0]
        y = 0
        sample1 = []
        for i in range(df1.shape[0]):
            x = x + 1
            YOUR_GENERATED_SECRET = "https://api.promptperfect.jina.ai/D4kHETzPfM50uLrFw6rV"
            if x % 20 == 0:
                url = urlList1.pop(0)
            headers = {
                "Content-Type": "application/json"
            }
            targetTask = '<!--start-input--> ' + '["' + str(
                df1['Sentences'][i]) + '"]' + '<!--end-input--> <!--start-output>'
            response = requests.post(url, headers=headers, json={
                "parameters": {"1": "1", "2": "2", "400": "400", "targetPrompt": targetTask}}, stream=True)
            if response.status_code == 200:
                samp = []
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        chunk = line.replace('data: ', '')
                        if len(chunk) > 0:
                            dict1 = chunk
                            samp.append(dict1)
                            print(chunk, end='', flush=True)
                        else:
                            print('')
                    else:
                        pass
                final_samp = "".join(samp)
                end_token = "<!--end-output-->"
                if end_token in final_samp:
                    sample1.append(final_samp)
                else:
                    final_samp = final_samp + end_token
                    sample1.append(final_samp)
            else:
                dict1 = response.text
                end_token = "<!--end-output-->"
                if end_token in dict1:
                    sample1.append(dict1)
                else:
                    dict1 = dict1 + end_token
                    sample1.append(dict1)
        string1 = "".join(sample1)
        list1 = string1.split('<!--end-output-->')
        amdDict = {
            "Sentence": [],
            "Tag": [],
        }
        i = 0
        for dict1 in list1:
            if len(dict1) == 0:
                continue
            if dict1 == 'null':
                continue
            keyval = dict1.split(':')
            if len(keyval) < 2:
                continue
            key = keyval[0]
            val = keyval[1]
            key = key.strip('{')
            val = val.strip('}')
            print(i)
            i = i + 1
            print(key, '----', val)
            amdDict['Sentence'].append(key)
            amdDict['Tag'].append(val)
        df = pd.DataFrame(amdDict)
        df['FinalTag'] = df['Tag']
        for i in range(df.shape[0]):
            tags = []
            text = df['Tag'][i]
            matches = re.findall(r'###(.*?)###', text)
            if not matches:
                for tag in superSet:
                    lowertext = text.lower()
                    lowertag = tag.lower()
                    if lowertag in lowertext:
                        tags.append(tag)
                df['FinalTag'][i] = tags
            else:
                for match in matches:
                    taglist = match.strip('[]')
                    taglist = taglist.replace('Management Change', '')
                    taglist = taglist.split(',')
                    for ta in taglist:
                        ta = ta.strip(' ')
                        print(ta, len(ta))
                        if len(ta) > 0:
                            tags.append(ta)
                df['FinalTag'][i] = tags
        print('Generate result for ', f'{filename}')
        print(len(urlList1))
        df.to_excel(f'{filename}_Processed_Final.xlsx')

class Utility():
    def __init__(self,input_filename, output_filename, thres, max_merge_len):
        self.filename= input_filename
        self.output_filename= output_filename
        self.thres = thres
        self.max_merge_len = max_merge_len
    def loadPdf(self,doc_reader):
        raw_text = ''
        for i, page in enumerate(doc_reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
        return raw_text
    def readDoc(self, link):
        URL = str(link)
        req = urllib.request.Request(URL, headers={'User-Agent': "Magic Browser"})
        remote_file = urllib.request.urlopen(req).read()
        remote_file_bytes = io.BytesIO(remote_file)
        doc_reader = PdfReader(remote_file_bytes)
        raw_text = self.loadPdf(doc_reader)
        sent_tokenizer = PunktSentenceTokenizer(raw_text)
        split_text = sent_tokenizer.tokenize(raw_text)
        df = pd.DataFrame(split_text)
        print(df.head())
        return raw_text

    def cosine_similarity(self, vector1, vector2):
        # Calculate the dot product of vector1 and vector2
        dot_product = np.dot(vector1, vector2)

        # Calculate the magnitudes (Euclidean norms) of vector1 and vector2
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        # Calculate the cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0  # To handle division by zero
        else:
            similarity = dot_product / (magnitude1 * magnitude2)
            return similarity

    def mergeSentence(self, thres, word_list, embeddings, max_merge_count):
        visited = {i: 0 for i in range(len(word_list))}
        curr_index = 0
        mark = 0
        for curr_index in range(len(word_list)):
            emb = embeddings[curr_index]
            merg = curr_index + 1
            if not visited[curr_index] == 0:
                continue
            mark = mark + 1
            visited[curr_index] = mark
            merge_count = 0
            while merg < len(word_list):
                merge_count = merge_count + 1
                emb_1 = embeddings[merg]
                score = self.cosine_similarity(emb_1, emb)
                score = score * 1.0
                if score > thres and merge_count < max_merge_count:
                    print(str(merg) + str(mark))
                    visited[merg] = mark
                else:
                    break
                merg = merg + 1
        merge_dict = {i: [] for i in range(len(word_list))}
        for key, val in visited.items():
            merge_dict[val].append(key)
        sentence_list = []
        for key, val in merge_dict.items():
            sent_val = [word_list[x] for x in val]
            merge_sent = ''.join(sent_val)
            if len(merge_sent) == 0:
                continue
            sentence_list.append(merge_sent)
        return sentence_list

    def read_docx(self, file_path):
        doc = Document(file_path)
        content = []
        for paragraph in doc.paragraphs:
            content.append(paragraph.text)
        return '\n'.join(content)

    def readLocalDOCX(self,file_path):
        try:
            docx = self.read_docx(file_path)
            raw_text = docx
            sent_tokenizer = PunktSentenceTokenizer(raw_text)
            split_text = sent_tokenizer.tokenize(raw_text)
            df = pd.DataFrame(split_text)
            print(df.head())
            return df
        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def MergeDataFrame(self):
        df = self.readLocalDOCX(f'{self.filename}')
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        wordlist = list(df[0][:])
        embeddings = embedding_model.encode(wordlist, show_progress_bar=True)
        data = {
            'Sentences': [],
            'tag': [],
        }
        sentence_list = self.mergeSentence(self.thres, wordlist, embeddings, self.max_merge_len)
        data['Sentences'] = sentence_list
        data['tag'] = sentence_list
        print(data['Sentences'][0])
        comDf = pd.DataFrame(data)
        print(comDf.head())
        return comDf

    def MergeDataFrameUsingNetworkAlgo(self):
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        df = self.readLocalDOCX(f'{self.filename}')
        # Tokenize text into sentences
        sentences = df[0][:].tolist()
        embeddings = embedding_model.encode(sentences, show_progress_bar=True)

        # Compute cosine similarity between sentences
        similarity_matrix = self.cosine_similarity(embeddings)
        print(similarity_matrix)
        # Create graph
        G = nx.Graph()

        # Add nodes
        for i, sentence in enumerate(sentences):
            G.add_node(i, text=sentence)

        # Add edges
        for i in range(len(sentences) - 1):
            similarity = similarity_matrix[i][i + 1]
            G.add_edge(i, i + 1, weight=similarity)

        # Detect communities using Louvain method
        partition = nx.community.louvain_communities(G, resolution=10, seed=123)

        print(partition)
        new_partition = []
        for x in partition:
            x = sorted(x)
            new_partition.append(x)
        partition = new_partition
        print(partition)
        # Group sentences into communities
        communities = defaultdict(list)
        for community_id, node in enumerate(partition):
            for item in node:
                communities[community_id].append(sentences[item])

        comDf = pd.DataFrame(columns= 'Sentences')
        for community_id, sentences in communities.items():
            print(f"Community {community_id + 1}:")
            t1 = []
            t1.append(f"\nCommunity {community_id + 1}:")
            for sentence in sentences:
                print("- ", sentence)
                t1.append(sentence)
        comDf = comDf.append({'Sentences': "".join(t1)}, ignore_index=True)
        return comDf



if __name__ == "__main__":
    app.run()

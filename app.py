from sentence_transformers import SentenceTransformer, util
import torch
import csv
import pickle


embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    global model
    global quote_list
    global stored_data
    global stored_sentences
    global stored_embeddings
    global list_of_csv

    device = 0 if torch.cuda.is_available() else -1

    with open('embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['quote_list']
        stored_embeddings = stored_data['corpus_embeddings']

    with open('quote3.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        list_of_csv = list(csv_reader)
        quote_list = []


def findQuotes(text, resultCount):
    global quote_list
    global stored_data
    global stored_sentences
    global stored_embeddings

    for x in list_of_csv:
        quote_list.append(x[0] + ", Author: " + x[1] + ", Tags: " + x[2])

    top_k = min(resultCount, len(quote_list))

    quotes = []

    query_embedding = embedder.encode(text, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, stored_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", text)
    print("\nMost similar quotes in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        quotes.append([quote_list[idx], "(Score: {:.4f})".format(score)])
        print(quote_list[idx], "(Score: {:.4f})".format(score))

    return quotes

# Inference is ran for every server call
# Reference your preloaded global model variable here.


def inference(model_inputs: dict) -> dict:
    global model
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    result = findQuotes(prompt, 5)

    # Return the results as a dictionary
    return result

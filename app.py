from sentence_transformers import SentenceTransformer
import torch




# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"


def init():
    # dummy change
    global model

    model = SentenceTransformer('all-MiniLM-L6-v2')

    device = 0 if torch.cuda.is_available() else -1


def getEmbeddings(text):
    query_embedding = model.encode(text, convert_to_tensor=True)
    embedding_json = {"embedding": [
        float(i) for i in list(query_embedding.cpu().numpy())]}

    return embedding_json

# Inference is ran for every server call
# Reference your preloaded global model variable here.


def inference(model_inputs: dict) -> dict:
    global model
    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    # Run the model
    embedding = getEmbeddings(prompt)

    # Return the results as a dictionary
    return embedding

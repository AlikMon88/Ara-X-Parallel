
import json
import os
import numpy as np
from data.load import *
from rag.embed import *
from model.model_arch.simple_nn import *
import shap
from sklearn.metrics import classification_report
from pprint import pprint
import inspect
from collections import Counter
import pickle as pkl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def route_query(query: str, llm) -> str:
    ## Categorizes the user's query
    
    system_prompt = """You are an intelligent routing agent for an ML system.
    Categorize the user's issue into exactly ONE of these categories:
    - [COMPUTE] (e.g., OOM, CUDA errors, hardware)
    - [DATA] (e.g., NaN loss, data drift, distribution shift)
    -[CODE] (e.g., syntax errors, tensor shape mismatches)
    
    Respond with ONLY the category tag (e.g., [DATA])."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])
    
    ## LCEL Chain: Prompt -> LLM -> String Output
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip()


### Tooling (JSON-model-Schema) / MCP-like

@tool
def read_training_logs(filepath: str = "model/logs/training_logs.json"):
    """Reads the JSON log file from the recent training run to check metrics like loss and gradients."""
    if not os.path.exists(filepath):
        return "Error: Log file not found."
    with open(filepath, "r") as f:
        data = json.load(f)
    return json.dumps(data)


@tool
def main_run_shap_analysis(model_path: str = "model/models_save/model.pth"):
    """Runs actual SHAP DeepExplainer to identify which image regions drive model predictions."""
    try:
        
        model = SimpleNN()
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        # Load Sample Data (100 background samples, 10 test samples to keep it fast)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        background = torch.stack([test_dataset[i][0] for i in range(100)])
        test_samples = torch.stack([test_dataset[i][0] for i in range(100, 110)])

        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(test_samples)

        # TRANSLATE TO LLM TEXT: Aggregate spatial importance
        # shap_values shape for PyTorch: list of length 10 (classes). Each is (10_samples, 1, 28, 28)
        # We take absolute values and average across classes, samples, and channels to get a 28x28 heat map
        shap_numpy = np.abs(np.array(shap_values)) 
        ## avg-across-the-classes too
        shap_numpy = np.mean(shap_numpy, axis=-1)
        global_importance = np.mean(shap_numpy, axis=(0, 1)) # Shape becomes (28, 28)

        # Let's map this: Does the model care about the center (the digit) or the edges (background noise)?
        center_mask = np.zeros((28, 28))
        center_mask[7:21, 7:21] = 1  # The 14x14 center pixels
        
        center_importance = np.sum(global_importance * center_mask)
        edge_importance = np.sum(global_importance * (1 - center_mask))
        total_importance = center_importance + edge_importance
        
        center_pct = (center_importance / total_importance) * 100
        edge_pct = (edge_importance / total_importance) * 100
        
        return (f"SHAP Spatial Execution Complete. "
                f"Feature Importance Distribution: "
                f"Center Region (14x14): {center_pct:.1f}% | Edge Region: {edge_pct:.1f}%. "
                f"(If edges hold high importance > 20%, the model is memorizing background noise instead of the object).")
                
    except Exception as e:
        return f"SHAP Analysis failed due to: {str(e)}"

    
def get_framework_docs_vector():
    
    # You can add any PyTorch or TensorFlow documentation URLs here.
    # We are targeting common debugging pain points (Optimization, Dropout, Loss functions, Imbalance)
    # urls =[
    #     "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",
    #     "https://pytorch.org/docs/stable/optim.html",
    #     "https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html",
    #     "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
    #     "https://pytorch.org/docs/stable/notes/randomness.html",
    #     "https://pytorch.org/docs/stable/data.html" # Covers DataLoaders and Samplers
    # ]
    
    urls = [

        # Core
        "https://docs.pytorch.org/docs/stable/index.html",

        # Optimization
        "https://pytorch.org/docs/stable/optim.html",
        "https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html",

        # Autograd
        "https://pytorch.org/docs/stable/autograd.html",
        "https://pytorch.org/docs/stable/notes/autograd.html",

        # Modules
        "https://pytorch.org/docs/stable/nn.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.Module.html",
        "https://pytorch.org/docs/stable/nn.functional.html",
        "https://pytorch.org/docs/stable/nn.init.html",

        # Losses
        "https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html",

        # Data
        "https://pytorch.org/docs/stable/data.html",

        # CUDA
        "https://pytorch.org/docs/stable/cuda.html",
        "https://pytorch.org/docs/stable/notes/cuda.html",

        # AMP
        "https://pytorch.org/docs/stable/amp.html",

        # Reproducibility
        "https://pytorch.org/docs/stable/notes/randomness.html",
        "https://pytorch.org/docs/stable/notes/reproducibility.html",

        # Serialization
        "https://pytorch.org/docs/stable/notes/serialization.html",

        # Numerical
        "https://pytorch.org/docs/stable/notes/numerical_accuracy.html",

        # Broadcasting
        "https://pytorch.org/docs/stable/notes/broadcasting.html",

        # Profiling
        "https://pytorch.org/docs/stable/profiler.html",

        # Distributed
        "https://pytorch.org/docs/stable/distributed.html",

        # Tensorboard
        "https://pytorch.org/docs/stable/tensorboard.html",

    ]
    
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, # Overlap prevents cutting sentences in half
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    splits = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    return vector_store


@tool
def main_search_framework_docs(query: str):
    """Searches PyTorch/TensorFlow documentation for fixes to ML issues."""
    # create a FAISS vector DB retrieval
    vs_frameworks = get_framework_docs_vector()
    ret_framework_vs = vs_frameworks.as_retriever(search_kwargs={"k": 2})
    ret_framework_docs = ret_framework_vs.invoke(query)
    return "\n\n".join(doc.page_content for doc in ret_framework_docs) 

def get_db_files_vector():
    cate_dict = {
    '[DATA]': 'data/incident_spaces/data_incidents.json',
    '[COMPUTE]': 'data/incident_spaces/compute_incidents.json',
    '[CODE]': 'data/incident_spaces/code_incidents.json'
    }
    
    vector_embed_cache = {}
    
    documents = load_documents(cate_dict['[DATA]'])
            
    for key in cate_dict.keys():
        documents = load_documents(cate_dict[key])
        vector_embed_cache[key] = create_vector_store(documents) ## BERT-based embedding-vector (encoder-architechture)
        print('vector-embed-shape: ', vector_embed_cache[key].index.ntotal, vector_embed_cache[key].index.d)

    return vector_embed_cache

@tool
def search_db_files(query): ##for categorical-routing
    """ Searches Local Database Files/JSONs/Data for causes and fixes to ML issues."""

    vector_embed_cache = get_db_files_vector()

    llm = load_llm(model_name='openai')
    
    category = route_query(query, llm)
    selected_vs = vector_embed_cache[category]
    retriever = selected_vs.as_retriever(search_kwargs={"k": 2})
    ret_docs = retriever.invoke(query) ## LangChain-doc-datatype
    
    return "\n\n".join(doc.page_content for doc in ret_docs)
    
### Tool in data-distribution-info / model-architechture info
@tool 
def model_arch_info():
    """Provides the Model Architecture for model contextualization"""
    
    model = SimpleNN()
    code_str = inspect.getsource(inspect.getmodule(model))
    return f"Model-Architecture: \n{code_str}"


@tool
def evaluate_model_per_class(model_path: str = "model/models_save/model.pth"):
    """Loads the saved PyTorch model, runs inference on the balanced MNIST test set, and generates a per-class precision/recall report."""
    
    model = SimpleNN()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    all_preds = []
    all_targets =[]

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(target.numpy())

    report = classification_report(all_targets, all_preds, zero_division=0)
    return f"Model Evaluation Report (Scikit-Learn):\n{report}"

@tool
## retrieve train/test-data distribution information every tooling call
def data_distribution_understand(plot=True):
    """
    Loads training and test distributions and optionally plots them.
    Useful for detecting class imbalance or distribution drift.
    """

    print("retrieving train/test-distrib information")

    with open('./data/train_dist.pkl', 'rb') as f:
        train_dist = pkl.load(f)

    with open('./data/test_dist.pkl', 'rb') as f:
        test_dist = pkl.load(f)

    if plot:

        train_keys = list(train_dist.keys())
        train_values = list(train_dist.values())

        test_keys = list(test_dist.keys())
        test_values = list(test_dist.values())

        plt.figure()

        plt.bar(train_keys, train_values)
        plt.title("Training Data Distribution")

        plt.figure()

        plt.bar(test_keys, test_values)
        plt.title("Test Data Distribution")

        plt.show()

    ret_str = (
        f"Train-Distribution: {train_dist}\n\n"
        f"Test-Distribution: {test_dist}"
    )

    return ret_str
if __name__ == "__main__":
    print('Running__Main__Tooling__')
    # def main_run_shap_analysis_nontool(model_path: str = "model/models_save/model.pth"):
    #     """Runs actual SHAP DeepExplainer to identify which image regions drive model predictions."""
    #     try:
            
    #         model = SimpleNN()
    #         model.load_state_dict(torch.load(model_path, weights_only=True))
    #         model.eval()

    #         # Load Sample Data (100 background samples, 10 test samples to keep it fast)
    #         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    #         test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            
    #         background = torch.stack([test_dataset[i][0] for i in range(100)])
    #         test_samples = torch.stack([test_dataset[i][0] for i in range(100, 120)])
    #         print(background.shape, test_samples.shape)

    #         explainer = shap.DeepExplainer(model, background)
    #         shap_values = explainer.shap_values(test_samples)

    #         # TRANSLATE TO LLM TEXT: Aggregate spatial importance
    #         # shap_values shape for PyTorch: list of length 10 (classes). Each is (n_samples, 1, 28, 28, n_class) // (i, :, :, :, j) >> i-th sample n-th class SHAP-values
    #         # We take absolute values and average across classes, samples, and channels to get a 28x28 heat map
    #         shap_numpy = np.abs(np.array(shap_values)) 
    #         ## avg-across-the-classes too
    #         shap_numpy = np.mean(shap_numpy, axis=-1)

    #         global_importance = np.mean(shap_numpy, axis=(0, 1)) # Shape becomes (28, 28) 
            
    #         # Let's map this: Does the model care about the center (the digit) or the edges (background noise)?
    #         center_mask = np.zeros((28, 28))
    #         center_mask[7:21, 7:21] = 1  # The 14x14 center pixels
            
    #         center_importance = np.sum(global_importance * center_mask)
    #         edge_importance = np.sum(global_importance * (1 - center_mask))
    #         total_importance = center_importance + edge_importance
            
    #         center_pct = (center_importance / total_importance) * 100
    #         edge_pct = (edge_importance / total_importance) * 100
            
    #         return (f"SHAP Spatial Execution Complete. "
    #                 f"Feature Importance Distribution: "
    #                 f"Center Region (14x14): {center_pct:.1f}% | Edge Region: {edge_pct:.1f}%. "
    #                 f"(If edges hold high importance > 20%, the model is memorizing background noise instead of the object).")
                    
    #     except Exception as e:
    #         return f"SHAP Analysis failed due to: {str(e)}"
        
    # print('main-shap-test-START')
    # pprint(main_run_shap_analysis_nontool())
    # print('main-shap-test-END')
    
    # def m_arch():
    #     model = SimpleNN()
    #     code_str_1 = inspect.getsource(model.__class__)
    #     print(code_str_1)
        
    #     code_str_2 = inspect.getsource(inspect.getmodule(model))
    #     print(code_str_2)
    
    # pprint(m_arch())
        
import requests
from bs4 import BeautifulSoup
import streamlit as sl
from langchain_ollama import OllamaLLM
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# load a model
llm = OllamaLLM(model="mistral")

# web scrapper func


def web_scrapper(url):
    try:
        sl.write(f"scrapping website :{url}")
        headers = {"user-agent": "chrome"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return f"fail to fetch {url}"
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:2000]
    except Exception as e:
        return f"error: {str(e)}"


# ai summerize func
def summerize_content(content):
    sl.write("summerizing ...")
    return llm.invoke(f"summerize the following content\n\n{content[:1000]}")


# web UI
sl.title("ai web scrapper")
sl.write("enter url")
url = sl.text_input("url :")

if url:
    content = web_scrapper(url)
    if "failed" in content or "stop" in content:
        sl.write(content)
    else:
        summary = summerize_content(content)
        sl.subheader("website summary")
        sl.write(summary)

# # main.py
# from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
# from pydantic import BaseModel
# from PIL import Image
# import pytesseract
# import pdfplumber
# import shutil
# import nltk
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords
# from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# import numpy as np
# import warnings
#
# from milvus_handler import connect_to_milvus, insert_data, query_collection, embedding_model
#
# app = FastAPI()
#
# @app.on_event("startup")
# async def startup_event():
#     connect_to_milvus()
#
# # Initialize models
# chat_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', api_key="AIzaSyBF3755EbGuaFzXRwNNaICVhaDfr2-raRg", convert_system_message_to_human=True)
#
# def extract_text_from_pdf(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         text = ''
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text
#
# def extract_text_from_image(image_path):
#     try:
#         image = Image.open(image_path)
#         pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'
#         text = pytesseract.image_to_string(image)
#         return text
#     except Exception as e:
#         print(f"Error extracting text from image: {e}")
#         return ""
#
# def generate_embeddings(text, embedding_model):
#     sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
#     embeddings = embedding_model.encode(sentences)
#     return sentences, embeddings
#
# def get_neighboring_sentences(top_n_results, sentences, num_neighbors=2):
#     indices = [sentences.index(result) for result in top_n_results]
#     neighboring_sentences = []
#     for index in indices:
#         start = max(0, index - num_neighbors)
#         end = min(len(sentences), index + num_neighbors + 1)
#         neighboring_sentences.extend(sentences[start:end])
#     return neighboring_sentences
#
# def generate_insights_with_llm(retrieved_sentences, keywords, chat_model):
#     context = "\n".join(retrieved_sentences)
#     prompt = (
#         f"You are a financial expert. Based on the following context:\n"
#         f"{context}\n\n"
#         f"Please provide detailed financial insights to answer the question: {keywords}\n"
#         "Make sure your answer is specific, insightful, and based on the context given, don't give the limitations in context make the best with what you have, do not say context or information given is incomplete"
#     )
#     response = chat_model.invoke(prompt)
#     return response.content
#
# def query_and_generate_insights(keywords, sentences, top_n=5, num_neighbors=5):
#     retrieved_sentences = query_collection(keywords, top_k=top_n)
#     neighboring_sentences = get_neighboring_sentences(retrieved_sentences, sentences, num_neighbors=num_neighbors)
#     insights = generate_insights_with_llm(neighboring_sentences, keywords, chat_model)
#     return insights
#
# def extract_keywords(query):
#     stop_words = set(stopwords.words('english'))
#     tokenizer = RegexpTokenizer(r'\w+|\w+[-/]\w+')
#     words = tokenizer.tokenize(query.lower())
#     filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
#     tagged_words = nltk.pos_tag(filtered_words)
#     important_words = [word for word, pos in tagged_words if pos.startswith('N') or pos.startswith('V') or pos.startswith('J')]
#     bigram_measures = BigramAssocMeasures()
#     finder = BigramCollocationFinder.from_words(filtered_words)
#     bigrams = finder.nbest(bigram_measures.pmi, 5)
#     bigram_phrases = [' '.join(bigram) for bigram in bigrams]
#     bigram_phrases = [word for phrase in bigram_phrases for word in phrase.split()]
#     combined_keywords = set(important_words + bigram_phrases)
#     final_keywords = sorted(combined_keywords, key=lambda x: query.lower().find(x))
#     return final_keywords
#
# @app.get("/healthcheck")
# def healthcheck():
#     return {"status": "ok"}
# @app.post("/process/")
# async def process_document(query: str = None, pdf_file: UploadFile = File(None), image_file: UploadFile = File(None), background_tasks: BackgroundTasks = BackgroundTasks()):
#     if not (pdf_file or image_file):
#         raise HTTPException(status_code=400, detail="At least one of pdf_file or image_file must be provided")
#
#     pdf_path = None
#     image_path = None
#     if pdf_file:
#         pdf_path = f"./{pdf_file.filename}"
#         with open(pdf_path, "wb") as buffer:
#             shutil.copyfileobj(pdf_file.file, buffer)
#     if image_file:
#         image_path = f"./{image_file.filename}"
#         with open(image_path, "wb") as buffer:
#             shutil.copyfileobj(image_file.file, buffer)
#
#     pdf_text = extract_text_from_pdf(pdf_path) if pdf_path else ""
#     image_text = extract_text_from_image(image_path) if image_path else ""
#     combined_text = pdf_text + "\n" + image_text
#
#     sentences, embeddings = generate_embeddings(combined_text, embedding_model)
#     insert_data(sentences, embeddings)
#
#     keywords = extract_keywords(query) if query else ""
#     if keywords:
#         insights = query_and_generate_insights(" ".join(keywords), sentences)
#         return {"insights": insights}
#     return {"message": "No query provided or keywords could not be extracted."}
#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8002)


# main.py
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from pydantic import BaseModel
from PIL import Image
import pytesseract
import pdfplumber
import shutil
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from langchain_google_genai import ChatGoogleGenerativeAI
from milvus_handler import connect_to_milvus, insert_data, query_collection, embedding_model
import os

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    connect_to_milvus()


# Initialize models
chat_model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    api_key=os.getenv("GOOGLE_API_KEY"),  # Use environment variable for API key
    convert_system_message_to_human=True
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_text_from_image(image_path: str) -> str:
    """Extract text from an image file using OCR."""
    try:
        image = Image.open(image_path)
        pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR/tesseract.exe'  # Adjust path as needed
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""


def generate_embeddings(text: str):
    """Generate embeddings for the given text."""
    sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    embeddings = embedding_model.encode(sentences)
    return sentences, embeddings


def get_neighboring_sentences(top_n_results, sentences, num_neighbors=2):
    """Get neighboring sentences for the retrieved results."""
    indices = [sentences.index(result) for result in top_n_results]
    neighboring_sentences = []
    for index in indices:
        start = max(0, index - num_neighbors)
        end = min(len(sentences), index + num_neighbors + 1)
        neighboring_sentences.extend(sentences[start:end])
    return neighboring_sentences


def generate_insights_with_llm(retrieved_sentences, keywords):
    """Generate insights using the language model."""
    context = "\n".join(retrieved_sentences)
    prompt = (
        f"You are a financial expert. Based on the following context:\n"
        f"{context}\n\n"
        f"Please provide detailed financial insights to answer the question: {keywords}\n"
        "Make sure your answer is specific, insightful, and based on the context given."
    )
    response = chat_model.invoke(prompt)
    return response.content


def query_and_generate_insights(keywords, sentences, top_n=5, num_neighbors=5):
    """Query the collection and generate insights based on retrieved sentences."""
    retrieved_sentences = query_collection(keywords, top_k=top_n)
    neighboring_sentences = get_neighboring_sentences(retrieved_sentences, sentences, num_neighbors=num_neighbors)
    insights = generate_insights_with_llm(neighboring_sentences, keywords)
    return insights


def extract_keywords(query: str):
    """Extract keywords from the user's query."""
    stop_words = set(stopwords.words('english'))
    tokenizer = RegexpTokenizer(r'\w+|\w+[-/]\w+')
    words = tokenizer.tokenize(query.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    tagged_words = nltk.pos_tag(filtered_words)
    important_words = [word for word, pos in tagged_words if
                       pos.startswith('N') or pos.startswith('V') or pos.startswith('J')]
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(filtered_words)
    bigrams = finder.nbest(bigram_measures.pmi, 5)
    bigram_phrases = [' '.join(bigram) for bigram in bigrams]
    combined_keywords = set(important_words + bigram_phrases)
    final_keywords = sorted(combined_keywords, key=lambda x: query.lower().find(x))
    return final_keywords


@app.get("/healthcheck")
def healthcheck():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/process/")
async def process_document(
        query: str = None,
        pdf_file: UploadFile = File(None),
        image_file: UploadFile = File(None),
        background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Process the uploaded document and generate insights based on the query."""
    if not (pdf_file or image_file):
        raise HTTPException(status_code=400, detail="At least one of pdf_file or image_file must be provided")

    pdf_path = None
    image_path = None
    try:
        if pdf_file:
            pdf_path = f"./{pdf_file.filename}"
            with open(pdf_path, "wb") as buffer:
                shutil.copyfileobj(pdf_file.file, buffer)
        if image_file:
            image_path = f"./{image_file.filename}"
            with open(image_path, "wb") as buffer:
                shutil.copyfileobj(image_file.file, buffer)

        pdf_text = extract_text_from_pdf(pdf_path) if pdf_path else ""
        image_text = extract_text_from_image(image_path) if image_path else ""
        combined_text = pdf_text + "\n" + image_text

        sentences, embeddings = generate_embeddings(combined_text)
        insert_data(sentences, embeddings)

        keywords = extract_keywords(query) if query else ""
        if keywords:
            insights = query_and_generate_insights(" ".join(keywords), sentences)
            return {"insights": insights}
        return {"message": "No query provided or keywords could not be extracted."}
    finally:
        # Clean up uploaded files
        if pdf_path and os.path.exists(pdf_path):
            os.remove(pdf_path)
        if image_path and os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8002)
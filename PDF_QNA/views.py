from django.shortcuts import render
from django.core.cache import cache

from PDF_QNA.forms import Qform
from PDF_QNA.apis import keys

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.document_loaders import PyPDFLoader

from transformers import AutoTokenizer
import transformers
import torch

model_id = 'ericzzz/falcon-rw-1b-instruct-openorca'

def load_pipeline():

    pipeline  = cache.get('pipeline')
    torch.cuda.empty_cache()
    if not pipeline :
        device, dtype = ('cuda', torch.bfloat16) if torch.cuda.is_available() else ('cpu',torch.float16) 
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipeline = transformers.pipeline(
            'text-generation',
            model=model_id,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            device_map=device,
            )
        cache.set('pipeline', pipeline, None)  

    return pipeline


def pdf(request):
    form = Qform()
    return render(request, "pdf.html", context={"form" : form})

def upload_pdf(request):
    if request.method == 'POST':
        form = Qform(request.POST)
        if 'pdf_file' in request.FILES :
            f = request.FILES['pdf_file']
            with open("temp.pdf", "wb+") as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
            data = PyPDFLoader('temp.pdf')
            pdf = data.load()

            # Step 2: Transform (Split)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                                        "\n\n", "\n", "(?<=\. )", " "], length_function=len)
            docs = text_splitter.split_documents(pdf)

            gpt4all_embd = GPT4AllEmbeddings()

            # Step 4: Store
            # Initialize MongoDB python client
            client = MongoClient(keys['MONGO_STR'], server_api=ServerApi('1'))
            collection = client['try']['vec']

            # Reset w/out deleting the Search Index 
            collection.delete_many({})

            docsearch = MongoDBAtlasVectorSearch.from_documents(
                docs, gpt4all_embd, collection=collection, index_name = "vector_index"
            )
    return render(request, "pdf.html", context={"form" : form})


def qna(request):
    form = Qform(request.POST)
    if form.is_valid():
        pipeline = load_pipeline()
        query = form.cleaned_data['text_input']

        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            keys['MONGO_STR'],
            "try" + "." + "vec",
            GPT4AllEmbeddings(),
            index_name="vector_index",
        )

        results = vector_search.similarity_search_with_score(
            query=query, k=2
        )

        context = ''

        for result in results:
            context += result[0].page_content


        system_message = 'You are a helpful assistant. Give answers only if the information is present in the context, if information is not present answer with "Information is not present."'
        prompt = f'<SYS> {system_message} <CONTEXT> {context} <INST> {query} <RESP> '

        response = pipeline(
        prompt, 
        max_length=1024,
        repetition_penalty=1.05
        )
        response = response[0]['generated_text'].split("<RESP>")[-1]
    
        return render(request, 'pdf.html', context={"response" : response, "form" : form})

    

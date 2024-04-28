from django.shortcuts import render

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.document_loaders import PyPDFLoader

from transformers import AutoTokenizer
import transformers
import torch

#from apis import keys


# Create your views here.
def pdf(request):
    return render(request, "pdf.html")

def qna(request):
    if request.method == 'POST':
        print(request.FILES)
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
        #print('Split into ' + str(len(docs)) + ' docs')

        # Step 3: Embed
        # https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html

        gpt4all_embd = GPT4AllEmbeddings()

        # Step 4: Store
        # Initialize MongoDB python client
        client = MongoClient(keys['MONGO_STR'], server_api=ServerApi('1'))
        collection = client['try']['vec']

        # # Reset w/out deleting the Search Index 
        collection.delete_many({})

        docsearch = MongoDBAtlasVectorSearch.from_documents(
            docs, gpt4all_embd, collection=collection, index_name = "vector_index"
        )


    if query: 

        if 'uploaded' not in st.session_state :
            st.error('Please upload a PDF', icon="🚨")
        
        else :
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

            st.write(response)
    return render(request, "pdf.html")
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


def initialize_splitter(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
                        # Set a really small chunk size, just to show.
                        chunk_size = chunk_size,
                        chunk_overlap  = chunk_overlap,
                        length_function = len,
                        is_separator_regex = False,
                    )
    return text_splitter

def load_split_html_file(html_file, text_splitter):
    loader = UnstructuredHTMLLoader(html_file)
    data = loader.load_and_split(text_splitter)
    return data

def load_split_pdf_file(pdf_file, text_splitter):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator='/n',
                                          chunk_size = 1000,
                                            chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    # documents = loader.load_and_split()
    

    print("Pages in the original document: ", len(documents))
    print("Length of chunks after splitting pages: ", len(text_chunks))
    print("================")
    print("lenght: ", len(text_chunks))
    return text_chunks




import openai
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from core.settings import BASE_DIR


PROMPT_TEXT = (
    "Use the following pieces of context to truthfully answer the question at the end."
    "To be sure answer consistent with the Context and doesn't require external knowledge.(use bullet-points in separate lines) "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."
    "Always reply in the same language you are being asked."
)


class EmbedDoc:
    def __init__(self, model: str, chunk_size: int = 64):
        self.chunk_size = chunk_size
        self.model = SentenceTransformer(model)

    def _embedding_func(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint with exponential backoff."""
        # replace newlines, which can negatively affect performance.
        if isinstance(text, list):
            text = [t.replace("\n", " ") for t in text]
        else:
            text = text.replace("\n", " ")
        return self.model.encode(text)

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        results = []
        _chunk_size = chunk_size or self.chunk_size
        for i in range(0, len(texts), _chunk_size):
            embeddings = self._embedding_func(texts[i : i + _chunk_size])
            results += [list(embedding) for embedding in embeddings]
        return results

    def embed_query(self, query: str) -> List[float]:
        """Embed a query string.

        Args:
            query: The query string to embed.

        Returns:
            The embedding of the query as a list of floats.
        """
        return list(self._embedding_func(query))


class DocIndex:
    """
    Index Document using faiss

    Args:
            document_path (str): path of the document
            model_name (str): model name which help to create embedings of document
    """

    def __init__(self, document_path: str, model_name: str):
        self.document_path = os.path.join(BASE_DIR, document_path.lstrip("/"))
        self.document_embedder = EmbedDoc(model_name)
        self.faiss_index = None

    def similarity_search(self, query_str: str, k: int = 5) -> List[Document]:
        """Search for documents similar to the given query string using the FAISS index.

        Args:
         query_str (str): The query string to search for.
         k (int, optional): The number of documents to return. Defaults to 5.
        Returns:
         List[Document]: A list of the top-k most similar documents to the query.
        """
        docs = self.faiss_index.similarity_search(query_str, k=k)
        return docs

    def save_index(self, dir_name: str):
        """Save the FAISS index to disk in the specified directory.
        The index is saved as a binary file with the same name as the input document.

        Args:
            dir_name (str): The directory where the index file will be saved.
        """
        loader = PyPDFLoader(self.document_path)
        pages = loader.load_and_split()
        self.faiss_index = FAISS.from_documents(pages, self.document_embedder)
        self.faiss_index.save_local(
            f"{dir_name}/{os.path.basename(self.document_path)}"
        )

    def load_local_index(self, dir_name):
        """Load the FAISS index from a binary file on disk.

        The index file is expected to be located in the specified directory and have the same
        name as the input document.

        Args:
            dir_name (_type_): The directory where the index file is located
        """
        self.faiss_index = FAISS.load_local(
            f"{dir_name}/{os.path.basename(self.document_path)}",
            self.document_embedder,
        )


def get_prompt(text: str = None):
    """Returns a PromptTemplate object for generating prompts with optional text.

    Args:
        text (str, optional): Optional text to include in the prompt. Defaults to None.

    Returns:
        PromptTemplate: A PromptTemplate object containing a template string that can be filled in with a context
        and question to create a prompt for answering the question."""
    prompt_template = """{text}

    {context}

    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        partial_variables={"text": text if text is not None else PROMPT_TEXT},
    )
    return prompt

    # task = (
    #     "Answer the question truthfully based on the text below. "
    #     "Include at least one verbatim quote (marked with quotation marks) and a comment where to find it in the text (ie name of the section and page number). "
    #     "Use ellipsis in the quote to omit irrelevant parts of the quote. "
    #     "After the quote write (in the new paragraph) a step by step explanation to be sure we have the right answer "
    #     "(use bullet-points in separate lines)"  # , adjust the language for a young reader). "
    #     "After the explanation check if the Answer is consistent with the Context and doesn't require external knowledge. "
    #     "In a new line write 'SELF-CHECK OK' if the check was successful and 'SELF-CHECK FAILED' if it failed. "
    # )
    # prompt = f"""


# {task or 'Task: Answer question based on context.'}

# Context:
# {chosen_sections}

# Question: {self.query}

# Answer:"""  # TODO: move to prompts.py

# return prompt

import csv
import openai
from PyPDF2 import PdfReader
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Use load_env to trace the path of .env:
load_dotenv(".env")

COMPLETIONS_MODEL = "gpt-3.5-turbo"
openai.api_key = os.environ["OPENAI_API_KEY"]
separator_len = 3
MAX_SECTION_LEN = 500
SEPARATOR = "\n* "
COMPLETIONS_API_PARAMS = {
    # We use temperature of 0.0 because it gives the most predictable, factual answer.
    "temperature": 0.0,
    "max_tokens": 150,
    "model": COMPLETIONS_MODEL,
}


class PDFTextExtractor:
    def __init__(self, file):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.file = file

    def count_tokens(self, text):
        """count the number of tokens in a string"""
        return len(self.tokenizer.encode(text))

    def extract_pages(
        self,
        page_text: str,
        index: int,
    ) -> str:
        """
        Extract the text from the page
        """
        if len(page_text) == 0:
            return []

        content = " ".join(page_text.split())
        outputs = [("Page " + str(index), content, self.count_tokens(content) + 4)]

        return outputs

    def extract_text_from_pdf(self):
        reader = PdfReader(self.file)
        res = []
        i = 1
        for page in reader.pages:
            res += self.extract_pages(page.extract_text(), i)
            i += 1
        df = pd.DataFrame(res, columns=["title", "content", "tokens"])
        df = df[df.tokens < 2046]
        df = df.reset_index().drop("index", axis=1)  # reset index
        return df

    def save_to_csv(self, df: pd.DataFrame):
        df.to_csv(f"{self.file}.pages.csv", index=False)


class DocEmbeddings:
    """
    Class for computing and writing document embeddings using the OpenAI Embeddings API.
    """

    def __init__(self, model: str):
        """
        Initializes the class with the OpenAI API key and the name of the model to use for embeddings.

        :param api_key: The OpenAI API key.
        :param model_name: The name of the OpenAI model to use for embeddings.
        """
        # openai.api_key = api_key
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_embedding(self, text: str) -> List[float]:
        """
        Gets the embedding for a single string using the OpenAI Embeddings API.

        :param text: The text to compute the embedding for.
        :return: The embedding as a list of floats.
        """

        embedding = self.model.encode(text)
        return embedding.tolist()
        # result = openai.Embedding.create(model=self.model_name, input=text)
        # return result["data"][0]["embedding"]

    def compute_doc_embeddings(self, df: pd.DataFrame) -> Dict[Tuple[str], List[float]]:
        """
        Computes the embeddings for each row in the dataframe using the OpenAI Embeddings API.

        :param df: The dataframe to compute the embeddings for.
        :return: A dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {idx: self.get_embedding(r.content) for idx, r in df.iterrows()}

    def write_embeddings_to_csv(
        self, filename: str, doc_embeddings: Dict[Tuple[str], List[float]]
    ) -> None:
        """
        Writes the embeddings to a CSV file.

        :param filename: The name of the CSV file to write the embeddings to.
        :param doc_embeddings: The dictionary of embeddings to write.
        """

        with open(f"{filename}.embeddings.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["title"] + list(range(self.model.get_sentence_embedding_dimension()))
            )
            for i, embedding in list(doc_embeddings.items()):
                writer.writerow(["Page " + str(i + 1)] + embedding)

    @staticmethod
    def load_embeddings(fname: str) -> Dict[Tuple[str, str], List[float]]:
        """
        Read the document embeddings and their keys from a CSV.

        fname is the path to a CSV with exactly these named columns:
            "title", "0", "1", ... up to the length of the embedding vectors.
        """

        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c) for c in df.columns if c != "title"])

        return {
            (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }


class Answer:
    def __init__(
        self,
        query: str,
        df: pd.DataFrame,
        document_embeddings: Dict[Tuple[str, str], List[float]],
    ):
        self.df = df
        self.doc_embeddings = document_embeddings
        self.query = query

    def construct_answer_with_openai(self):
        most_relevant_document_sections = self.order_document()
        prompt = self.construct_prompt(most_relevant_document_sections)
        # print("===\n", prompt)

        # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        # model = GPT2LMHeadModel.from_pretrained("gpt2")

        # input_ids = tokenizer.encode(prompt, return_tensors='pt')
        # output = model.generate(input_ids,max_length=50, do_sample=False, temperature=0.1)
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # print(generated_text)

        # return generated_text

        response = openai.ChatCompletion.create(
            messages=[{"role": "user", "content": prompt}], model=COMPLETIONS_MODEL
        )
        return response["choices"][0]["message"]["content"].strip(" \n")

    def construct_prompt(self, most_relevant_document_sections):
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for _, section_index in most_relevant_document_sections:
            document_section = self.df.loc[self.df["title"] == section_index].iloc[0]

            chosen_sections_len += document_section.tokens + separator_len
            if chosen_sections_len > MAX_SECTION_LEN:
                space_left = MAX_SECTION_LEN - chosen_sections_len - len(SEPARATOR)
                chosen_sections.append(
                    SEPARATOR + document_section.content[:space_left]
                )
                chosen_sections_indexes.append(str(section_index))
                break

            chosen_sections.append(SEPARATOR + document_section.content)
            chosen_sections_indexes.append(str(section_index))

        task = (
            "Answer the question truthfully based on the text below. "
            "Include at least one verbatim quote (marked with quotation marks) and a comment where to find it in the text (ie name of the section and page number). "
            "Use ellipsis in the quote to omit irrelevant parts of the quote. "
            "After the quote write (in the new paragraph) a step by step explanation to be sure we have the right answer "
            "(use bullet-points in separate lines)"  # , adjust the language for a young reader). "
            "After the explanation check if the Answer is consistent with the Context and doesn't require external knowledge. "
            "In a new line write 'SELF-CHECK OK' if the check was successful and 'SELF-CHECK FAILED' if it failed. "
        )
        prompt = f"""
		{task or 'Task: Answer question based on context.'}
		
		Context:
		{chosen_sections}
		
		Question: {self.query}
		
		Answer:"""  # TODO: move to prompts.py

        return prompt

    @staticmethod
    def vector_similarity(x: List[float], y: List[float]) -> float:
        """
        We could use cosine similarity or dot product to calculate the similarity between vectors.
        In practice, we have found it makes little difference.
        """

        # calculate cosine similarity using numpy

        return np.dot(np.array(x), np.array(y))

    def order_document(self) -> List[Tuple[float, Tuple[str, str]]]:
        embedding = DocEmbeddings("model")
        query_embedding = embedding.get_embedding(self.query)

        document_similarities = sorted(
            [
                (self.vector_similarity(query_embedding, doc_embedding), doc_index)
                for doc_index, doc_embedding in self.doc_embeddings.items()
            ],
            reverse=True,
        )

        return document_similarities

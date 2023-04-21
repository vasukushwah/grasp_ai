from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from .serializers import FileUploadSerializer
from app.models import FileUpload
from .utils import DocIndex, EmbedDoc, get_prompt
import pandas as pd
from rest_framework.views import exception_handler
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_protect
import traceback
import time
import os
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


class FilesView(APIView):
    def post(self, request, format=None):
        serializer = FileUploadSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            file_path = serializer.data["file"]
            print(file_path)

            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(
            {"success": False, "error": serializer.errors, "data": None},
            status=status.HTTP_400_BAD_REQUEST,
        )

    def get(self, request, format=None):
        try:
            file_uploads = FileUpload.objects.all()
            serializer = FileUploadSerializer(file_uploads, many=True)

            # Check if the serializer is valid before accessing the errors attribute

            return Response(
                {"success": True, "data": serializer.data}, status=status.HTTP_200_OK
            )
        except Exception as e:
            print(e)
            return Response(
                {"success": False, "error": str(e), "data": None},
                status=status.HTTP_400_BAD_REQUEST,
            )

    def delete(self, request, id):
        try:
            # get the file by id and delete it from the database
            file = FileUpload.objects.get(id=id)
            file.delete()
            return Response({"success": True}, status=status.HTTP_200_OK)
        except FileUpload.DoesNotExist:
            return Response(
                {"success": False, "error": "File not found", "data": None},
                status=status.HTTP_400_BAD_REQUEST,
            )


class FileEmbedingView(APIView):
    def get(self, request, id):
        try:
            # get the file by id and delete it from the database
            file = FileUpload.objects.get(id=id)
            serializer = FileUploadSerializer(file)

            file_path = serializer.data.get("file")
            index = DocIndex(
                file_path,
                "bert-base-nli-mean-tokens",
            )
            index.save_index("faiss-indexs")
            # extractor = PDFTextExtractor(file.file)
            # df = extractor.extract_text_from_pdf()
            # extractor.save_to_csv(df)
            # embeddings = DocEmbeddings("all-MiniLM-L6-v2")
            # doc_embeddings = embeddings.compute_doc_embeddings(df)
            # embeddings.write_embeddings_to_csv(file.file, doc_embeddings)
            return Response({"success": True}, status=status.HTTP_200_OK)

        except FileUpload.DoesNotExist:
            return Response(
                {"success": False, "error": "File not found.", "data": None},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            print(e)
            return Response(
                {"success": False, "error": str(e), "data": None},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


@api_view(["POST"])
@csrf_protect
def ask_question_view(request):
    try:
        body = dict(request.data)
        id = body.get("id")
        question = body.get("question")
        if not question.endswith("?"):
            question += "?"
        file = FileUpload.objects.get(id=id)
        file_path = str(file.file)
        index = DocIndex(
            file_path,
            "bert-base-nli-mean-tokens",
        )
        index.load_local_index("faiss-indexs")
        prompt = get_prompt()
        chain_type_kwargs = {"prompt": prompt}
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=index.faiss_index.as_retriever(),
            chain_type_kwargs=chain_type_kwargs,
        )
        answer = qa.run(question)
        data = {
            "question": question,
            "answer": answer,
        }
        return Response({"success": True, "data": data}, status=status.HTTP_200_OK)
    except Exception as e:
        traceback_str = traceback.format_exc()
        return Response(
            {
                "success": False,
                "error": str(e),
                "traceback": traceback_str,
                "data": None,
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

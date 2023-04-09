from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from .serializers import FileUploadSerializer
from app.models import FileUpload
from .utils import PDFTextExtractor, DocEmbeddings, Answer
import pandas as pd
from rest_framework.views import exception_handler
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_protect
import traceback
import time


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
            extractor = PDFTextExtractor(file.file)
            df = extractor.extract_text_from_pdf()
            extractor.save_to_csv(df)
            embeddings = DocEmbeddings("all-MiniLM-L6-v2")
            doc_embeddings = embeddings.compute_doc_embeddings(df)
            embeddings.write_embeddings_to_csv(file.file, doc_embeddings)
            return Response({"success": True}, status=status.HTTP_200_OK)

        except FileUpload.DoesNotExist:
            return Response(
                {"success": False, "error": "File not found", "data": None},
                status=status.HTTP_400_BAD_REQUEST,
            )


@method_decorator(csrf_protect, name="dispatch")
class AskQuestionView(APIView):
    def post(
        self,
        request,
    ):
        try:
            body = dict(request.data)
            id = body.get("id")
            print(id)
            question = body.get("question")
            if not question.endswith("?"):
                question += "?"
            file = FileUpload.objects.get(id=id)
            df = pd.read_csv(f"{file.file}.pages.csv")
            document_embeddings = DocEmbeddings.load_embeddings(
                f"{file.file}.embeddings.csv"
            )
            answerObj = Answer(question, df, document_embeddings)
            answer = answerObj.construct_answer_with_openai()
            # answer = f"hello! this is heading.\nthe given below points are valid.\n - one \n- two \n-three"
            # time.sleep(4)
            # print(answer)
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

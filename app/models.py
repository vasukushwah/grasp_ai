from django.db import models


class FileUpload(models.Model):
    file = models.FileField(upload_to="uploads/%Y/%m/%d/")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name

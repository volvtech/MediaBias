from django.db import models

# Create your models here.

class ArticlesSummary(models.Model):
    article_summary = models.TextField(max_length=1024*2)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
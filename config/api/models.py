from django.db import models

class ModelVersion(models.Model):
    """Track deployed model versions"""
    version = models.CharField(max_length=50, unique=True)
    deployed_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    model_path = models.CharField(max_length=500)
    metadata = models.JSONField(default=dict)
    
    class Meta:
        ordering = ['-deployed_at']

class RecommendationLog(models.Model):
    """Log API recommendations for monitoring"""
    query_type = models.CharField(max_length=50)
    query_data = models.JSONField()
    results_count = models.IntegerField()
    response_time_ms = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    model_version = models.CharField(max_length=50)
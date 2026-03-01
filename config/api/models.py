"""
api/models.py
-------------
Django ORM models for tracking deployed versions and logging API requests.

After editing run:
    python manage.py makemigrations api
    python manage.py migrate
"""

from django.db import models


class ModelVersion(models.Model):
    """
    Records every model version that has been trained and deployed.
    Populated automatically by the training pipeline or manually via admin.
    """

    version      = models.CharField(max_length=100, unique=True)
    deployed_at  = models.DateTimeField(auto_now_add=True)
    is_active    = models.BooleanField(default=True)
    model_path   = models.CharField(max_length=500)
    n_laptops    = models.IntegerField(default=0)
    metadata     = models.JSONField(default=dict)

    class Meta:
        ordering = ["-deployed_at"]

    def __str__(self):
        flag = "✅" if self.is_active else "⬜"
        return f"{flag} {self.version} ({self.n_laptops} laptops)"

    @classmethod
    def set_active(cls, version: str) -> "ModelVersion":
        """Mark *version* as active and deactivate all others."""
        cls.objects.exclude(version=version).update(is_active=False)
        obj, _ = cls.objects.get_or_create(version=version)
        obj.is_active = True
        obj.save(update_fields=["is_active"])
        return obj


class RecommendationLog(models.Model):
    """
    Logs every API recommendation request for monitoring, analytics,
    and debugging.  Written by views.py on each successful response.
    """

    QUERY_TYPES = [
        ("text_search",   "Text Search"),
        ("similar",       "Similar Items"),
        ("preferences",   "Preferences"),
        ("personalized",  "Personalized"),
    ]

    query_type      = models.CharField(max_length=50, choices=QUERY_TYPES)
    query_data      = models.JSONField()                 # the raw request body
    results_count   = models.IntegerField(default=0)
    response_time_ms= models.FloatField(default=0.0)
    timestamp       = models.DateTimeField(auto_now_add=True)
    model_version   = models.CharField(max_length=100, blank=True)

    class Meta:
        ordering = ["-timestamp"]
        indexes  = [
            models.Index(fields=["query_type"]),
            models.Index(fields=["timestamp"]),
            models.Index(fields=["model_version"]),
        ]

    def __str__(self):
        return f"[{self.timestamp:%Y-%m-%d %H:%M}] {self.query_type} → {self.results_count} results ({self.response_time_ms:.0f}ms)"

    @classmethod
    def record(
        cls,
        query_type:       str,
        query_data:       dict,
        results_count:    int,
        response_time_ms: float,
        model_version:    str = "",
    ) -> "RecommendationLog":
        """
        Convenience factory used by views.py to log each request.

        Usage in views.py
        -----------------
        RecommendationLog.record(
            query_type="text_search",
            query_data={"query": "gaming laptop"},
            results_count=len(results),
            response_time_ms=elapsed_ms,
            model_version=ModelManager.get_version() or "",
        )
        """
        return cls.objects.create(
            query_type=query_type,
            query_data=query_data,
            results_count=results_count,
            response_time_ms=response_time_ms,
            model_version=model_version,
        )
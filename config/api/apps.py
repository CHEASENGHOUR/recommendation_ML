from django.apps import AppConfig


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    def ready(self):
        """
        Called once after Django has fully initialised all apps.
        We pre-load the recommendation model here.

        The `RUN_MAIN` guard prevents double-loading in Django's
        development server which spawns two processes.
        """
        import os
        if os.environ.get("RUN_MAIN") != "true":
            return   # skip in the file-watcher process

        try:
            from api.views import ModelManager
            ModelManager.get_engine()
        except FileNotFoundError:
            print("[api] ⚠  No trained model found — run `python main.py` first.")
        except Exception as exc:
            print(f"[api] ⚠  Model pre-load failed: {exc}")

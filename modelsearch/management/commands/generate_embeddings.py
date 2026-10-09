"""Generate embeddings for indexed objects that don't have them yet.

This command is useful when:
- Enabling semantic search on an existing index
- Updating embeddings after changing the embedding provider
- Backfilling embeddings for objects that were indexed before
  semantic search was enabled

Usage::

    python manage.py generate_embeddings
    python manage.py generate_embeddings --model search.SearchTerm
    python manage.py generate_embeddings --force  # Regenerate all embeddings
    python manage.py generate_embeddings --chunk-size 500
"""

import logging

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from modelsearch.backends import get_search_backend
from modelsearch.conf import get_app_config
from modelsearch.embeddings import get_embedding_provider
from modelsearch.index import SemanticField, get_indexed_models


logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 100


class Command(BaseCommand):
    help = "Generate embeddings for indexed objects (semantic search)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--model",
            action="store",
            dest="model",
            default=None,
            help="Specify a model to process (format: app_label.ModelName). "
            "If not specified, all indexed models with SemanticFields are processed.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            dest="force",
            default=False,
            help="Regenerate embeddings even for objects that already have them",
        )
        parser.add_argument(
            "--chunk-size",
            action="store",
            dest="chunk_size",
            default=DEFAULT_CHUNK_SIZE,
            type=int,
            help="Number of objects to process at once",
        )
        parser.add_argument(
            "--backend",
            action="store",
            dest="backend_name",
            default="default",
            help="Search backend to use (default: 'default')",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            dest="dry_run",
            default=False,
            help="Show what would be done without making changes",
        )

    def handle(self, **options):
        self.verbosity = options["verbosity"]
        model_filter = options["model"]
        force = options["force"]
        chunk_size = options["chunk_size"]
        backend_name = options["backend_name"]
        dry_run = options["dry_run"]

        # Get the backend
        try:
            backend = get_search_backend(backend_name)
        except Exception as e:
            raise CommandError(
                f"Could not get search backend '{backend_name}': {e}"
            ) from e

        # Check if semantic search is enabled on this backend
        if not getattr(backend, "semantic_enabled", False):
            self.stdout.write(
                self.style.WARNING(
                    f"Warning: Semantic search is not enabled on backend '{backend_name}'. "
                    "Set SEMANTIC_ENABLED=True in MODELSEARCH_BACKENDS config."
                )
            )

        # Get models to process
        models = get_indexed_models()
        if model_filter:
            try:
                app_label, model_name = model_filter.split(".")
                models = [
                    m
                    for m in models
                    if m._meta.app_label == app_label
                    and m.__name__ == model_name
                ]
                if not models:
                    raise CommandError(f"Model '{model_filter}' not found or not indexed")
            except ValueError as e:
                raise CommandError(
                    f"Invalid model format '{model_filter}'. Use: app_label.ModelName"
                ) from e

        # Filter to models that have SemanticFields
        models_with_semantic = []
        for model in models:
            semantic_fields = [
                f for f in model.get_search_fields() if isinstance(f, SemanticField)
            ]
            if semantic_fields:
                models_with_semantic.append((model, semantic_fields))

        if not models_with_semantic:
            self.stdout.write(
                self.style.WARNING(
                    "No indexed models with SemanticFields found. "
                    "Add index.SemanticField(...) to a model's search_fields."
                )
            )
            return

        # Get the embedding provider
        try:
            provider = get_embedding_provider()
            self.stdout.write(
                f"Using embedding provider: {type(provider).__name__} "
                f"(dimensions={provider.dimensions})"
            )
        except Exception as e:
            raise CommandError(
                f"Could not initialize embedding provider: {e}"
            ) from e

        total_processed = 0
        total_updated = 0

        for model, semantic_fields in models_with_semantic:
            self.stdout.write(
                f"\nProcessing {model._meta.app_label}.{model.__name__} "
                f"(semantic fields: {', '.join(f.field_name for f in semantic_fields)})"
            )

            # Get the index for this model
            index = backend.get_index_for_model(model)

            # Get objects that need embeddings
            queryset = model.get_indexed_objects().order_by("pk")

            # Get existing index entries to check which ones need embeddings
            IndexEntry = get_app_config().get_model("IndexEntry")
            from modelsearch.utils import get_descendants_content_types_pks

            content_type_pks = get_descendants_content_types_pks(model)

            if not force:
                # Only process objects without embeddings
                existing_with_embeddings = set(
                    IndexEntry._default_manager.filter(
                        content_type_id__in=content_type_pks,
                        embedding__isnull=False,
                    ).values_list("object_id", flat=True)
                )
                # Filter out objects that already have embeddings
                queryset = [
                    obj
                    for obj in queryset
                    if str(obj.pk) not in existing_with_embeddings
                ]
                self.stdout.write(
                    f"  Skipping {len(existing_with_embeddings)} objects with existing embeddings"
                )
            else:
                queryset = list(queryset)

            if not queryset:
                self.stdout.write("  No objects to process")
                continue

            self.stdout.write(f"  Processing {len(queryset)} objects...")

            # Process in chunks
            for i in range(0, len(queryset), chunk_size):
                chunk = queryset[i : i + chunk_size]

                if dry_run:
                    total_processed += len(chunk)
                    continue

                # Generate embeddings for this chunk
                with transaction.atomic():
                    # Use the backend's index to add items (which generates embeddings)
                    index.add_items(model, chunk)

                total_processed += len(chunk)
                total_updated += len(chunk)

                if self.verbosity > 1:
                    self.stdout.write(
                        f"    Processed {min(i + chunk_size, len(queryset))}/{len(queryset)}"
                    )

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone! Processed {total_processed} objects, "
                f"updated {total_updated} embeddings."
                + (" (DRY RUN - no changes made)" if dry_run else "")
            )
        )

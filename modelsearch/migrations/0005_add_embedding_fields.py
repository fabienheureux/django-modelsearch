"""Add embedding fields for semantic search support."""

from django.db import connection, migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("modelsearch", "0004_rename_modelsear_autocom_476c89_gin_modelsearch_autocom_ea8581_gin_and_more"),
    ]

    if connection.vendor == "postgresql":
        operations = [
            migrations.AddField(
                model_name="indexentry",
                name="embedding",
                field=models.JSONField(blank=True, default=None, null=True),
            ),
            migrations.AddField(
                model_name="indexentry",
                name="embedding_text",
                field=models.TextField(blank=True, default=""),
            ),
        ]
    else:
        operations = []

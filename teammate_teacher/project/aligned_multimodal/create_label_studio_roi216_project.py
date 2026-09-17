from __future__ import annotations

import json
from pathlib import Path

from django.db import transaction
from io_storages.localfiles.models import LocalFilesImportStorage
from projects.models import Project
from tasks.models import Prediction, Task
from users.models import User


REPO_DIR = Path.cwd()
PROJECT_ASSETS = (
    REPO_DIR
    / "aligned_multimodal"
    / "data"
    / "local_roi_annotation_v2"
    / "label_studio_roi216"
)
TITLE = "ROI216 Local定位审计 v1"


def load_contract() -> tuple[str, list[dict[str, object]]]:
    label_config = (PROJECT_ASSETS / "label_config.xml").read_text(encoding="utf-8")
    tasks = json.loads((PROJECT_ASSETS / "tasks.json").read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or len(tasks) != 186:
        raise ValueError("ROI216 project must import exactly 186 new tasks")
    modes = [str(task["data"]["annotation_mode"]) for task in tasks]
    if modes.count("blind") != 60 or modes.count("correction") != 126:
        raise ValueError("Expected 60 blind and 126 correction tasks")
    if any("predictions" in task for task in tasks if task["data"]["annotation_mode"] == "blind"):
        raise ValueError("Blind tasks must not contain predictions")
    if any("predictions" not in task for task in tasks if task["data"]["annotation_mode"] == "correction"):
        raise ValueError("Every correction task must contain one automatic-box prediction")
    return label_config, tasks


def ensure_local_image_storage(project: Project) -> LocalFilesImportStorage:
    image_dir = (PROJECT_ASSETS / "images").resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"ROI216 image directory not found: {image_dir}")
    storage, _ = LocalFilesImportStorage.objects.get_or_create(
        project=project,
        path=str(image_dir),
        defaults={
            "title": "ROI216 static images",
            "use_blob_urls": True,
        },
    )
    changed_fields: list[str] = []
    if storage.title != "ROI216 static images":
        storage.title = "ROI216 static images"
        changed_fields.append("title")
    if not storage.use_blob_urls:
        storage.use_blob_urls = True
        changed_fields.append("use_blob_urls")
    if changed_fields:
        storage.save(update_fields=changed_fields)
    storage.validate_connection()
    return storage


def create_or_reuse() -> tuple[int, bool]:
    label_config, task_specs = load_contract()
    matches = list(Project.objects.filter(title=TITLE))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple Label Studio projects named {TITLE}")
    if matches:
        project = matches[0]
        task_count = Task.objects.filter(project=project).count()
        prediction_count = Prediction.objects.filter(project=project).count()
        if task_count != 186 or prediction_count != 126:
            raise RuntimeError(
                f"Existing project {project.id} is incomplete: "
                f"tasks={task_count}, predictions={prediction_count}"
            )
        if project.label_config != label_config:
            raise RuntimeError(
                f"Existing project {project.id} has a different label config"
            )
        ensure_local_image_storage(project)
        return int(project.id), False

    template = Project.objects.order_by("-id").first()
    if template is None:
        user = User.objects.order_by("id").first()
        if user is None or user.active_organization is None:
            raise RuntimeError("No Label Studio user/organization found")
        created_by = user
        organization = user.active_organization
    else:
        created_by = template.created_by
        organization = template.organization

    with transaction.atomic():
        project = Project.objects.create(
            title=TITLE,
            description=(
                "Kaggle CUHK-X Small Model Track：60条盲标定位评估 + "
                "126条自动框修正；旧Pilot30的30条标注单独复用。"
            ),
            label_config=label_config,
            created_by=created_by,
            organization=organization,
            show_skip_button=False,
            enable_empty_annotation=False,
            show_collab_predictions=True,
            reveal_preannotations_interactively=False,
            maximum_annotations=1,
            sampling=Project.SEQUENCE,
            is_draft=False,
            is_published=True,
        )
        for inner_id, spec in enumerate(task_specs, start=1):
            task = Task.objects.create(
                project=project,
                data=spec["data"],
                meta={},
                overlap=1,
                inner_id=inner_id,
            )
            predictions = spec.get("predictions", [])
            if len(predictions) > 1:
                raise ValueError(f"Task {inner_id} contains multiple predictions")
            if predictions:
                prediction = predictions[0]
                Prediction.objects.create(
                    task=task,
                    project=project,
                    result=prediction["result"],
                    score=prediction.get("score"),
                    model_version=prediction.get("model_version", ""),
                )

    task_count = Task.objects.filter(project=project).count()
    prediction_count = Prediction.objects.filter(project=project).count()
    if task_count != 186 or prediction_count != 126:
        raise RuntimeError(
            f"Post-import mismatch: tasks={task_count}, predictions={prediction_count}"
        )
    ensure_local_image_storage(project)
    return int(project.id), True


project_id, created = create_or_reuse()
print(
    "ROI216_PROJECT_RESULT="
    + json.dumps(
        {
            "project_id": project_id,
            "created": created,
            "tasks": 186,
            "blind": 60,
            "correction": 126,
            "predictions": 126,
        },
        ensure_ascii=False,
    )
)

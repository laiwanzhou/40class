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
    / "label_studio_blind60_review_v2"
)
TITLE = "ROI60 机器原框复核 v2"
MODEL_VERSION = "motion_bbox_all_frames_v1_review"


def load_contract() -> tuple[str, list[dict[str, object]]]:
    config = (PROJECT_ASSETS / "label_config.xml").read_text(encoding="utf-8")
    tasks = json.loads((PROJECT_ASSETS / "tasks.json").read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or len(tasks) != 60:
        raise ValueError("ROI60 review must contain exactly 60 tasks")
    if any(len(task.get("predictions", [])) != 1 for task in tasks):
        raise ValueError("Every ROI60 review task must contain one machine pre-box")
    return config, tasks


def ensure_storage(project: Project) -> LocalFilesImportStorage:
    image_dir = (PROJECT_ASSETS / "images").resolve()
    storage, _ = LocalFilesImportStorage.objects.get_or_create(
        project=project,
        path=str(image_dir),
        defaults={
            "title": "ROI60 review images",
            "description": "Machine-box and first-blind-box comparison assets.",
            "use_blob_urls": True,
        },
    )
    storage.validate_connection()
    return storage


def create_or_reuse() -> tuple[int, bool]:
    config, task_specs = load_contract()
    matches = list(Project.objects.filter(title=TITLE))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple projects named {TITLE}")
    if matches:
        project = matches[0]
        task_count = Task.objects.filter(project=project).count()
        prediction_count = Prediction.objects.filter(project=project).count()
        if task_count != 60 or prediction_count != 60:
            raise RuntimeError(
                f"Existing project {project.id} is incomplete: "
                f"tasks={task_count}, predictions={prediction_count}"
            )
        if project.label_config != config:
            raise RuntimeError(f"Existing project {project.id} has another config")
        if project.model_version != MODEL_VERSION:
            project.model_version = MODEL_VERSION
            project.save(update_fields=["model_version"])
        ensure_storage(project)
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
                "对首次60条盲画框做第二次复核：显示机器原框与首次人工盲框，"
                "不覆盖首次盲标结果。"
            ),
            label_config=config,
            created_by=created_by,
            organization=organization,
            show_skip_button=False,
            enable_empty_annotation=False,
            show_collab_predictions=True,
            model_version=MODEL_VERSION,
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
            prediction = spec["predictions"][0]
            Prediction.objects.create(
                task=task,
                project=project,
                result=prediction["result"],
                score=prediction.get("score"),
                model_version=prediction.get("model_version", ""),
            )
    ensure_storage(project)
    task_count = Task.objects.filter(project=project).count()
    prediction_count = Prediction.objects.filter(project=project).count()
    if task_count != 60 or prediction_count != 60:
        raise RuntimeError(
            f"Post-import mismatch: tasks={task_count}, predictions={prediction_count}"
        )
    return int(project.id), True


project_id, created = create_or_reuse()
print(
    "ROI60_REVIEW_PROJECT_RESULT="
    + json.dumps(
        {
            "project_id": project_id,
            "created": created,
            "tasks": 60,
            "predictions": 60,
        },
        ensure_ascii=False,
    )
)

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
    / "hard_local_v1"
    / "label_studio_annotation500"
)
TITLE = "困难动作 Local ROI 500 v1"
MODEL_VERSION = "hard_action_roi_locator_v1_all216"


def load_contract() -> tuple[str, list[dict[str, object]]]:
    config = (PROJECT_ASSETS / "label_config.xml").read_text(encoding="utf-8")
    tasks = json.loads((PROJECT_ASSETS / "tasks.json").read_text(encoding="utf-8"))
    if not isinstance(tasks, list) or len(tasks) != 500:
        raise ValueError("Hard Local project must contain exactly 500 tasks")
    if any(len(task.get("predictions", [])) != 1 for task in tasks):
        raise ValueError("Every task must contain one pre-annotation prediction")
    if any(
        len(task["predictions"][0].get("result", [])) != 2 for task in tasks
    ):
        raise ValueError("Every prediction must contain Depth and Thermal rectangles")
    return config, tasks


def ensure_storage(project: Project) -> LocalFilesImportStorage:
    image_dir = (PROJECT_ASSETS / "images").resolve()
    storage, _ = LocalFilesImportStorage.objects.get_or_create(
        project=project,
        path=str(image_dir),
        defaults={
            "title": "Hard Local 500 Depth Thermal assets",
            "description": "Depth/Thermal full, mapped ROI, and Local crop sheets.",
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
        if task_count != 500 or prediction_count != 500:
            raise RuntimeError(
                f"Existing project {project.id} incomplete: "
                f"tasks={task_count}, predictions={prediction_count}"
            )
        if project.label_config != config:
            raise RuntimeError(f"Existing project {project.id} uses another config")
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
                "困难动作条件 Local 分支的500条滚动ROI标注。"
                "同时复核Depth定位器v1与Depth到Thermal的归一化映射；"
                "原216条已全部用于定位器v1训练。"
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
                model_version=prediction.get("model_version", MODEL_VERSION),
            )
    ensure_storage(project)
    task_count = Task.objects.filter(project=project).count()
    prediction_count = Prediction.objects.filter(project=project).count()
    if task_count != 500 or prediction_count != 500:
        raise RuntimeError(
            f"Post-import mismatch: tasks={task_count}, predictions={prediction_count}"
        )
    return int(project.id), True


project_id, created = create_or_reuse()
print(
    "HARD_LOCAL500_PROJECT_RESULT="
    + json.dumps(
        {
            "project_id": project_id,
            "created": created,
            "tasks": 500,
            "predictions": 500,
        },
        ensure_ascii=False,
    )
)

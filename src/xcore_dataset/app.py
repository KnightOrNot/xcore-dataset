"""Read-only data browsing. No collection or motion is started by the UI."""

import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .integrity import verify


def create_app(root, *, lelab=True):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="Xcore remote datasets")

    @app.get("/api/datasets")
    def datasets():
        return {
            "datasets": [
                str(p.parent.relative_to(root))
                for p in sorted(root.rglob("READY.json"))
            ]
        }

    @app.get("/api/verify")
    def check(dataset: str):
        path = (root / dataset).resolve()
        if not path.is_relative_to(root):
            raise HTTPException(400, "invalid dataset path")
        try:
            verify(path)
        except (OSError, ValueError) as e:
            raise HTTPException(409, str(e)) from e
        return {"verified": True}

    def sample(dataset, index):
        path = (root / dataset).resolve()
        if not path.is_relative_to(root) or not (path / "READY.json").is_file():
            raise HTTPException(404, "published dataset not found")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset("local/" + path.name, root=path, depth_output_unit="m")
        if not 0 <= index < len(ds):
            raise HTTPException(400, "sample index out of range")
        return ds[index]

    @app.get("/api/sample")
    def sample_info(dataset: str, index: int = 0):
        row = sample(dataset, index)
        result = {
            key: row[key].tolist()
            for key in (
                "observation.state",
                "action",
                "observation.wrench",
                "observation.wrench_valid",
            )
        }
        result["instruction"] = row["task"]
        result["depth"] = {}
        for camera in ("cam1", "cam2"):
            depth = row["observation.images." + camera + "_depth"].numpy()[0]
            valid = depth[depth > 0]
            result["depth"][camera] = dict(
                unit="m",
                shape=list(depth.shape),
                valid_fraction=float((depth > 0).mean()),
                min_m=float(valid.min()) if valid.size else None,
                max_m=float(valid.max()) if valid.size else None,
            )
        return result

    @app.get("/api/depth-preview.png")
    def depth_preview(dataset: str, camera: str, index: int = 0):
        if camera not in ("cam1", "cam2"):
            raise HTTPException(400, "unknown camera")
        import numpy as np
        from PIL import Image

        row = sample(dataset, index)
        depth = row["observation.images." + camera + "_depth"].numpy()[0]
        # Display only; metric float32 depth in the dataset is never rewritten.
        preview = (np.clip(depth / 3, 0, 1) * 255).astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(preview).save(stream, format="PNG")
        return Response(stream.getvalue(), media_type="image/png")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (
            "<h1>Xcore 数据集</h1><p>远端数据浏览，不启动硬件。</p>"
            '<a href="/api/datasets">数据集列表</a> · '
            '<a href="/edit-dataset">LeLab</a>'
            "<p>状态／力／深度数值：/api/sample?dataset=数据集ID/版本&amp;index=0。</p>"
            "<p>深度预览：/api/depth-preview.png?dataset=数据集ID/版本&amp;camera=cam1&amp;index=0；"
            "预览按 0～3m 映射为灰度，训练数据仍为原始米制深度。</p>"
        )

    if lelab:
        os.environ["HF_LEROBOT_HOME"] = str(root)
        from lerobot.datasets import dataset_metadata

        dataset_metadata.HF_LEROBOT_HOME = root
        from lelab.server import FRONTEND_DIST
        from lelab.server import app as official_app

        @app.get("/edit-dataset")
        def browser(request: Request):
            origin = str(request.base_url).rstrip("/")
            if request.query_params.get("api") != origin:
                return RedirectResponse(
                    str(request.url.include_query_params(api=origin))
                )
            return FileResponse(FRONTEND_DIST / "index.html")

        @app.middleware("http")
        async def read_only_browser(request, call_next):
            if request.method not in ("GET", "HEAD", "OPTIONS"):
                raise_error = False
                # Official browsing may POST read-only metadata queries.
                if request.url.path not in ("/dataset-info",):
                    raise_error = True
                if raise_error:
                    from fastapi.responses import JSONResponse

                    return JSONResponse(
                        {"error": "This deployment only enables browsing"},
                        status_code=403,
                    )
            return await call_next(request)

        allowed_get = {
            "/datasets",
            "/dataset-episodes",
            "/dataset-episode",
            "/dataset-motion",
            "/dataset-thumbnails",
            "/dataset-frame",
            "/dataset-video",
            "/hf-auth-status",
            "/health",
        }
        for route in official_app.routes:
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", set()) or set()
            if (path in allowed_get and methods == {"GET"}) or (
                path == "/dataset-info" and methods == {"POST"}
            ):
                app.router.routes.append(route)
            elif getattr(route, "name", None) == "frontend":
                app.router.routes.append(route)
    return app

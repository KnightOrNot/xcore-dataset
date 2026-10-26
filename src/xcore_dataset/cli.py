import argparse
import json
import tomllib
from pathlib import Path

from .integrity import verify


def main(argv=None):
    p = argparse.ArgumentParser(
        description="CR7 remote dataset tools; never controls hardware"
    )
    p.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("default.toml")
    )
    subs = p.add_subparsers(dest="command", required=True)
    d = subs.add_parser("demo", help="create explicitly synthetic raw")
    d.add_argument("path", type=Path)
    d.add_argument("--frames", type=int, default=12)
    c = subs.add_parser("convert")
    c.add_argument("source", type=Path)
    c.add_argument("staging", type=Path)
    c.add_argument("destination", type=Path)
    c.add_argument("--fps", type=int)
    c.add_argument("--max-skew-ms", type=float)
    c.add_argument("--representation", choices=("pose10", "joints"), default="pose10")
    v = subs.add_parser("verify")
    v.add_argument("path", type=Path)
    v.add_argument("--readback", action="store_true")
    s = subs.add_parser("serve")
    s.add_argument("--root", type=Path)
    s.add_argument("--port", type=int)
    s.add_argument("--without-lelab", action="store_true")
    q = subs.add_parser("queue", help="enqueue a sealed session for conversion")
    q.add_argument("session")
    q.add_argument("--root", type=Path)
    q.add_argument("--dataset-id")
    q.add_argument("--representation", choices=("pose10", "joints"), default="pose10")
    j = subs.add_parser("jobs", help="list or run conversion jobs")
    j.add_argument("--root", type=Path)
    j.add_argument("--run", action="store_true")
    j.add_argument("--retry", metavar="JOB", help="requeue one failed job")
    ds = subs.add_parser("datasets", help="list published revisions")
    ds.add_argument("--root", type=Path)
    a = p.parse_args(argv)
    config = tomllib.loads(a.config.read_text())
    if a.command == "demo":
        from .demo import make_demo

        result = str(make_demo(a.path, a.frames, config["dataset"]["fps"]))
    elif a.command == "convert":
        from .convert import convert

        result = str(
            convert(
                a.source,
                a.staging,
                a.destination,
                fps=a.fps if a.fps is not None else config["dataset"]["fps"],
                max_skew_ms=a.max_skew_ms
                if a.max_skew_ms is not None
                else config["dataset"]["max_skew_ms"],
                representation=a.representation,
            )
        )
    elif a.command == "queue":
        from .jobs import enqueue

        root = a.root or config["dataset"].get("root", "~/project/xcore-data")
        result = enqueue(
            root,
            a.session,
            fps=config["dataset"]["fps"],
            max_skew_ms=config["dataset"]["max_skew_ms"],
            dataset_id=a.dataset_id,
            representation=a.representation,
        )
    elif a.command == "jobs":
        from .jobs import list_jobs, retry, run_pending

        root = a.root or config["dataset"].get("root", "~/project/xcore-data")
        if a.retry:
            retry(root, a.retry)
        result = run_pending(root) if (a.run or a.retry) else list_jobs(root)
    elif a.command == "datasets":
        from .published import published_datasets

        root = a.root or config["dataset"].get("root", "~/project/xcore-data")
        result = published_datasets(root / "lerobot")
    elif a.command == "verify":
        m = verify(a.path)
        result = {"verified": True, "files": len(m["files"])}
        if a.readback:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(
                "local/" + a.path.name, root=a.path, depth_output_unit="m"
            )
            if not len(ds):
                raise ValueError("empty LeRobot dataset")
            ds[0]
            ds[len(ds) - 1]
            result["readback_frames"] = len(ds)
    else:
        import uvicorn

        from .app import create_app

        uvicorn.run(
            create_app(
                a.root or config["server"]["data_root"], lelab=not a.without_lelab
            ),
            host=config["server"]["host"],
            port=a.port or config["server"]["port"],
        )
        return
    print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))


if __name__ == "__main__":
    main()

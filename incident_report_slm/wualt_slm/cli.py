import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="WUALT incident-report SLM wrapper")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the incident REST API and preview UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", type=Path)
    generate = commands.add_parser("generate", help="Generate from a JSON source snapshot")
    generate.add_argument("input", type=Path)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--pdf", type=Path)
    generate.add_argument("--require-model", action="store_true", help="Fail when the model service is unavailable")
    interpret = commands.add_parser("interpret", help="Return only an interpretation from compact backend signals")
    interpret.add_argument("input", type=Path)
    interpret.add_argument("--output", type=Path)
    setup = commands.add_parser("setup-model", help="Download pinned model assets with checksum verification")
    setup.add_argument("--directory", type=Path, default=Path.cwd())
    setup.add_argument("--model-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.data_dir:
            os.environ["WUALT_DATA_DIR"] = str(args.data_dir.resolve())
        # __init__ imports the public wrapper; refresh the setting before API import.
        from . import settings
        if args.data_dir:
            settings.DATA_ROOT = args.data_dir.resolve()
        if not os.getenv("WUALT_ALLOWED_ORIGINS"):
            os.environ["WUALT_ALLOWED_ORIGINS"] = f"http://127.0.0.1:{args.port},http://localhost:{args.port}"
        import uvicorn
        uvicorn.run("wualt_slm.api:app", host=args.host, port=args.port, workers=1, access_log=False)
    elif args.command == "setup-model":
        from .setup_model import main as setup_main
        setup_main(["--directory", str(args.directory)] + (["--model-only"] if args.model_only else []))
    elif args.command == "interpret":
        from .signal_interpretation import SignalInterpreter
        result = SignalInterpreter().interpret(json.loads(args.input.read_text(encoding="utf-8")))
        encoded = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded)
    else:
        from .wrapper import IncidentReporter
        reporter = IncidentReporter()
        if args.require_model and not reporter.backend.ready():
            parser.error("The configured model service is unavailable.")
        report = reporter.generate(json.loads(args.input.read_text(encoding="utf-8")))
        pdf = reporter.pdf(report) if args.pdf else None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.pdf:
            args.pdf.parent.mkdir(parents=True, exist_ok=True)
            args.pdf.write_bytes(pdf)
        print(json.dumps({"report_id": report["id"], "generation_mode": report["generation_mode"], "output": str(args.output)}))

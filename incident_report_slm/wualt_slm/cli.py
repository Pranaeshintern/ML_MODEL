import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="WUALT signal interpretation SLM wrapper")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the interpretation REST API and preview UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    interpret = commands.add_parser("interpret", help="Return separate audio and distress interpretations from backend signals")
    interpret.add_argument("input", type=Path)
    interpret.add_argument("--output", type=Path)
    setup = commands.add_parser("setup-model", help="Download pinned model assets with checksum verification")
    setup.add_argument("--directory", type=Path, default=Path.cwd())
    setup.add_argument("--model-only", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "serve":
        if not os.getenv("WUALT_ALLOWED_ORIGINS"):
            os.environ["WUALT_ALLOWED_ORIGINS"] = f"http://127.0.0.1:{args.port},http://localhost:{args.port}"
        import uvicorn
        uvicorn.run("wualt_slm.api:app", host=args.host, port=args.port, workers=1, access_log=False)
    elif args.command == "setup-model":
        from .setup_model import main as setup_main
        setup_main(["--directory", str(args.directory)] + (["--model-only"] if args.model_only else []))
    else:
        from .signal_interpretation import SignalInterpreter
        result = SignalInterpreter().interpret(json.loads(args.input.read_text(encoding="utf-8")))
        encoded = json.dumps(result, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        else:
            print(encoded)

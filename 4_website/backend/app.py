import json
import threading
import queue

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from engine import RecommendationEngine, validate_params, preload, _static

app = Flask(__name__)
CORS(app)

_preload_lock = threading.Lock()
_preload_done = False
_preload_error = None


def _ensure_preloaded():
    global _preload_done, _preload_error
    with _preload_lock:
        if _preload_done or _preload_error:
            return
        try:
            preload()
            _preload_done = True
        except Exception as e:
            _preload_error = str(e)


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.route("/api/status", methods=["GET"])
def status():
    return jsonify({
        "loaded": _static.loaded,
        "load_error": _static.load_error,
        "repo_count": len(_static.repo_names),
        "tag_count": len(_static.tag2idx),
    })


@app.route("/api/recommend", methods=["POST"])
def recommend():
    body = request.get_json(silent=True) or {}

    errors = validate_params(body)
    if errors:
        return jsonify({"errors": errors}), 400

    username = body["username"].strip()
    top_k = int(body.get("top_k", 30))
    model_type = body.get("model", "nn")
    include_forked = bool(body.get("include_forked", True))
    include_user_repos = bool(body.get("include_user_repos", True))
    clusters = int(body.get("clusters", 0))
    clustering_type = body.get("clustering_type", "k-means")

    def generate():
        q: queue.Queue = queue.Queue()

        def progress_cb(msg: str, event: str = "progress", **data):
            payload = {"message": msg, **data}
            q.put((event, payload))

        def run_engine():
            try:
                engine = RecommendationEngine(progress_cb=progress_cb)

                if not _static.loaded:
                    progress_cb("Preloading model data (first request only)...", event="progress")
                    try:
                        preload(progress_cb)
                    except Exception as e:
                        progress_cb(f"Failed to preload: {e}", event="error")
                        q.put(("done", {"success": False, "error": str(e)}))
                        return

                engine.run(
                    username=username,
                    top_k=top_k,
                    model_type=model_type,
                    include_forked=include_forked,
                    include_user_repos=include_user_repos,
                    clusters=clusters,
                    clustering_type=clustering_type,
                )
                q.put(("done", {"success": True}))
            except Exception as e:
                progress_cb(f"Error: {e}", event="error")
                q.put(("done", {"success": False, "error": str(e)}))

        thread = threading.Thread(target=run_engine, daemon=True)
        thread.start()

        while True:
            try:
                event, data = q.get(timeout=120)
                yield _sse_event(event, data)
                if event == "done":
                    break
            except queue.Empty:
                yield _sse_event("error", {"message": "Timeout: no response from engine"})
                break

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Preloading static data...")
    try:
        preload()
    except Exception as e:
        print(f"WARNING: Preload failed: {e}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

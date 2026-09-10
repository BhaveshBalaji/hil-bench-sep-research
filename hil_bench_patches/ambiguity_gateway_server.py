"""
ambiguity_gateway_server.py
---------------------------
Flask server that sits between the agent's tools and the ask_human oracle.
It provides five responsibilities:

1. /ask — resolves one explicit ambiguity ID through the upstream ask_human
          server. No fuzzy attribution is used.

2. /record_business_info — records a knowledge-base lookup for one ID.

3. /resolve_with_business_info — resolves one ID only with evidence present in
                                  a recorded knowledge-base response.

4. /submit_check  — gateway_submit_sql calls this before emitting the
                    <<SWE_AGENT_SUBMISSION>> sentinel. Returns:
                    {"allowed": true}  or
                    {"allowed": false, "reason": "...", "unresolved": [...]}

5. /ledger  — read current ledger state (used by view_ambiguities tool)

The server reads LEDGER_BASE_PATH and TASK_INSTANCE_ID from env to locate
the correct ledger. It forwards /ask requests to ASK_HUMAN_SERVER_URL
(the existing ask_human server) unchanged.

Ports:
    Gateway server:    GATEWAY_SERVER_PORT  (default 9532)
    Upstream ask_human: ASK_HUMAN_SERVER_URL (default http://localhost:9521)

Environment variables:
    LEDGER_BASE_PATH        — base directory for ledger files (required)
    GATEWAY_SERVER_PORT     — port for this server (default 9532)
    ASK_HUMAN_SERVER_URL    — upstream ask_human server URL (default http://localhost:9521/ask)
    GATEWAY_SERVER_URL      — URL agents use to reach this server (set by harness)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

CANT_ANSWER  = "can't answer"
IRRELEVANT   = "irrelevant question"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_ledger_base() -> str:
    base = os.environ.get("LEDGER_BASE_PATH", "")
    if not base:
        raise RuntimeError("LEDGER_BASE_PATH environment variable is not set")
    return base


def _normalize_task_id(raw_id: str) -> str:
    """Strip model/mode/pass suffixes — keep original task ID only."""
    return raw_id.split("__")[0]


def _load(task_id: str):
    """Load ledger for task_id (strips suffixes)."""
    from hil_bench.ambiguity_ledger import load_or_create_ledger
    return load_or_create_ledger(_get_ledger_base(), _normalize_task_id(task_id))


def _save(ledger) -> None:
    from hil_bench.ambiguity_ledger import save_ledger
    save_ledger(_get_ledger_base(), ledger)


def _ask_human_url() -> str:
    return os.environ.get("ASK_HUMAN_SERVER_URL", "http://localhost:9521/ask")


def _proxy_to_ask_human(payload: dict) -> dict:
    """Forward the payload to the upstream ask_human server and return parsed response."""
    url = _ask_human_url()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
            return body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        logger.error(f"Upstream ask_human HTTP {e.code}: {body}")
        return {"response": CANT_ANSWER}
    except Exception as e:
        logger.error(f"Upstream ask_human error: {e}")
        return {"response": CANT_ANSWER}


def _update_ledger_from_response(
    task_id: str,
    ambiguity_id: str,
    oracle_response: str,
) -> str:
    """Apply an oracle response to exactly one caller-selected ambiguity ID."""
    ledger = _load(task_id)
    if ledger.get(ambiguity_id) is None:
        raise KeyError(f"Unknown ambiguity ID: {ambiguity_id}")

    response_lower = oracle_response.lower()
    if CANT_ANSWER in response_lower:
        ledger.mark_failed(ambiguity_id)
        status = "failed"
    elif IRRELEVANT in response_lower:
        ledger.mark_rejected(ambiguity_id)
        status = "unresolved"
    else:
        ledger.mark_resolved(ambiguity_id, oracle_response, source="human")
        status = "resolved"

    _save(ledger)
    logger.info(f"[{task_id}] Ambiguity {ambiguity_id} -> {status}")
    return status


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ambiguity_gateway"})


@app.route("/ask", methods=["POST"])
def ask():
    """
    Resolve one explicit ledger entry through the human oracle.

    The ambiguity ID is required; the gateway never guesses which entry a
    question belongs to.
    """
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()
    instance_id = str(data.get("instance_id", "")).strip()
    ambiguity_id = str(data.get("ambiguity_id", "")).strip()

    if not question or not instance_id or not ambiguity_id:
        return jsonify({
            "response": CANT_ANSWER,
            "status": "error",
            "error": "instance_id, ambiguity_id, and question are required",
        }), 400

    try:
        ledger = _load(instance_id)
        if ledger.get(ambiguity_id) is None:
            return jsonify({
                "response": CANT_ANSWER,
                "status": "error",
                "error": f"Unknown ambiguity ID: {ambiguity_id}",
            }), 404

        if not ledger.has_business_info_check(ambiguity_id):
            return jsonify({
                "response": CANT_ANSWER,
                "status": "error",
                "error": (
                    f"Business information has not been checked for {ambiguity_id}. "
                    "Call get_business_info for this ID before ask_human."
                ),
            }), 409

        # ASKED remains blocking if the upstream request is interrupted.
        ledger.mark_asked(ambiguity_id, question)
        _save(ledger)

        upstream_resp = _proxy_to_ask_human({
            "question": question,
            "instance_id": instance_id,
        })
        oracle_response = upstream_resp.get("response", CANT_ANSWER)
        status = _update_ledger_from_response(
            instance_id, ambiguity_id, oracle_response
        )
        return jsonify({
            "response": oracle_response,
            "ambiguity_id": ambiguity_id,
            "status": status,
        })
    except Exception as exc:
        logger.exception("Explicit ask_human ledger update failed")
        return jsonify({
            "response": CANT_ANSWER,
            "status": "error",
            "error": str(exc),
        }), 500


@app.route("/record_business_info", methods=["POST"])
def record_business_info():
    """Record a gateway-aware knowledge-base lookup against one ledger ID."""
    data = request.get_json(silent=True) or {}
    instance_id = str(data.get("instance_id", "")).strip()
    ambiguity_id = str(data.get("ambiguity_id", "")).strip()
    search_string = str(data.get("search_string", "")).strip()
    result = str(data.get("result", "")).strip()
    found = data.get("found") is True
    if not instance_id or not ambiguity_id or not search_string or not result:
        return jsonify({"status": "error", "error": "Missing required fields"}), 400

    try:
        ledger = _load(instance_id)
        if not ledger.record_business_info_check(
            ambiguity_id, search_string, result, found
        ):
            return jsonify({
                "status": "error",
                "error": f"Unknown ambiguity ID: {ambiguity_id}",
            }), 404
        _save(ledger)
        logger.info(f"[{instance_id}] Business-info check recorded for {ambiguity_id}")
        return jsonify({"status": "recorded", "ambiguity_id": ambiguity_id})
    except Exception as exc:
        logger.exception("record_business_info failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/resolve_with_business_info", methods=["POST"])
def resolve_with_business_info():
    """Resolve one ID only when evidence came from its recorded KB results."""
    data = request.get_json(silent=True) or {}
    instance_id = str(data.get("instance_id", "")).strip()
    ambiguity_id = str(data.get("ambiguity_id", "")).strip()
    evidence = str(data.get("evidence", "")).strip()
    if not instance_id or not ambiguity_id or not evidence:
        return jsonify({"status": "error", "error": "Missing required fields"}), 400

    try:
        ledger = _load(instance_id)
        if ledger.get(ambiguity_id) is None:
            return jsonify({
                "status": "error",
                "error": f"Unknown ambiguity ID: {ambiguity_id}",
            }), 404
        if not ledger.business_info_supports(ambiguity_id, evidence):
            return jsonify({
                "status": "error",
                "error": "Evidence was not found in a recorded business-info result",
            }), 409
        ledger.mark_resolved(ambiguity_id, evidence, source="business_info")
        _save(ledger)
        logger.info(f"[{instance_id}] Ambiguity {ambiguity_id} resolved by business info")
        return jsonify({
            "status": "resolved",
            "ambiguity_id": ambiguity_id,
            "message": f"Ledger [{ambiguity_id}] resolved using recorded business information.",
        })
    except Exception as exc:
        logger.exception("resolve_with_business_info failed")
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/submit_check", methods=["POST"])
def submit_check():
    """
    Called by gateway_submit_sql before emitting the submission sentinel.

    Request body: {"instance_id": "...", "query": "SELECT ..."}
    Response:
        {"allowed": true}
        {"allowed": false, "reason": "...", "unresolved": [...]}
    """
    data        = request.get_json(silent=True) or {}
    instance_id = data.get("instance_id", "")
    query       = data.get("query", "")

    if not instance_id:
        return jsonify({"allowed": False, "reason": "Missing instance_id"}), 400

    try:
        ledger = _load(instance_id)
    except Exception as e:
        logger.error(f"submit_check ledger load error: {e}")
        # The ledger is the enforcement state.  Allowing a missing/corrupt ledger
        # would make an infrastructure failure indistinguishable from a pass.
        return jsonify({"allowed": False, "reason": "Ledger unavailable"}), 503

    if ledger.is_submit_allowed():
        logger.info(f"[{instance_id}] submit_check ALLOWED — {ledger.summary()}")
        return jsonify({"allowed": True})

    unresolved = [
        {
            "id":          a.id,
            "description": a.description,
            "type":        a.type,
            "status":      a.status.value,
        }
        for a in ledger.unresolved_critical()
    ]

    reason = (
        f"Submission blocked: {len(unresolved)} critical ambiguit"
        f"{'y' if len(unresolved) == 1 else 'ies'} unresolved. "
        "Resolve every critical ambiguity with recorded business evidence or by "
        "calling ask_human with its exact ambiguity ID before submitting."
    )

    logger.info(f"[{instance_id}] submit_check BLOCKED — {ledger.summary()}")
    return jsonify({
        "allowed":    False,
        "reason":     reason,
        "unresolved": unresolved,
        "summary":    ledger.summary(),
    })


@app.route("/ledger", methods=["GET", "POST"])
def get_ledger():
    """
    Return the current ledger state for a task.
    Accepts instance_id as query param or JSON body field.
    """
    instance_id = request.args.get("instance_id") or (
        (request.get_json(silent=True) or {}).get("instance_id", "")
    )
    if not instance_id:
        return jsonify({"error": "Missing instance_id"}), 400

    try:
        ledger = _load(instance_id)
        return jsonify(ledger.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/register_ambiguity", methods=["POST"])
def register_ambiguity():
    """
    Agent-initiated ambiguity registration.
    Called by the agent when it discovers a new ambiguity not in the ledger.

    Request body:
        {
            "instance_id": "...",
            "description": "What does 'senior' mean?",
            "type": "threshold",      (optional)
            "critical": true          (optional, default true)
        }
    """
    from hil_bench.ambiguity_ledger import AmbiguityEntry, AmbiguityStatus

    data        = request.get_json(silent=True) or {}
    instance_id = data.get("instance_id", "")
    description = data.get("description", "").strip()

    if not instance_id or not description:
        return jsonify({"error": "Missing instance_id or description"}), 400

    try:
        ledger = _load(instance_id)

        # Generate a new unique ID
        existing_ids = {a.id for a in ledger.ambiguities}
        i = len(ledger.ambiguities) + 1
        while f"A{i}" in existing_ids:
            i += 1
        new_id = f"A{i}"

        entry = AmbiguityEntry(
            id=new_id,
            description=description,
            type=data.get("type", "business_definition"),
            critical=bool(data.get("critical", True)),
            status=AmbiguityStatus.UNRESOLVED,
            source="agent",
        )
        ledger.add(entry)
        _save(ledger)

        logger.info(
            f"[{instance_id}] Agent registered new ambiguity {new_id}: {description[:60]}"
        )
        return jsonify({"id": new_id, "status": "registered"})

    except Exception as e:
        logger.error(f"register_ambiguity error: {e}")
        return jsonify({"error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ambiguity Gateway Server")
    parser.add_argument(
        "--port", type=int,
        default=int(os.environ.get("GATEWAY_SERVER_PORT", 9532)),
    )
    parser.add_argument(
        "--ask-human-url", type=str,
        default=os.environ.get("ASK_HUMAN_SERVER_URL", "http://localhost:9521/ask"),
    )
    args = parser.parse_args()

    # Override env var so _ask_human_url() picks it up
    os.environ["ASK_HUMAN_SERVER_URL"] = args.ask_human_url

    if not os.environ.get("LEDGER_BASE_PATH"):
        logger.warning(
            "LEDGER_BASE_PATH is not set — ledger operations will fail. "
            "Set LEDGER_BASE_PATH before starting the gateway server."
        )

    logger.info(
        f"Starting Ambiguity Gateway on port {args.port}, "
        f"forwarding to {args.ask_human_url}"
    )
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()

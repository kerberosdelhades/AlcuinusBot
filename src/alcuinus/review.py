"""
Phase 10 — Review cycle

Re-audits existing decay classifications and detects stale content.
Designed to be run as a cron job with any frequency (monthly, quarterly,
biannually — whatever schedule fits the data volume).

Checks performed:
    1. Semi-stable clusters older than their retention window → flag
    2. Ephemeral clusters past retention → flag for removal
    3. Evergreen clusters with no recent activity → flag for review
    4. Generates a review report (data/review_report.json)
    5. Optionally regenerates digest + syllabus if changes detected

Usage (standalone):
    uv run python -m alcuinus.review

Usage (cron):
    Sets up a Hermes cron job that runs review on a schedule.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from alcuinus.decay import DECAY_PROFILES
from alcuinus.output import run_output as regenerate_digest
from alcuinus.syllabus import run_syllabus as regenerate_syllabus

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DECAY_PATH = "data/decay_profiles.json"
DEFAULT_CLUSTERS_PATH = "data/clusters.json"
DEFAULT_OUTPUT_PATH = "data/review_report.json"
DEFAULT_SNAPSHOT_PATH = "data/review_snapshot.json"

# ---------------------------------------------------------------------------
# Review logic
# ---------------------------------------------------------------------------


def load_current_state(
    decay_path: str = DEFAULT_DECAY_PATH,
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
) -> tuple[dict, dict]:
    """Load current decay profiles and cluster data."""
    with open(decay_path, encoding="utf-8") as f:
        decay = json.load(f)
    with open(clusters_path, encoding="utf-8") as f:
        clusters = json.load(f)
    return decay, clusters


def check_staleness(
    decay: dict,
    clusters: dict,
    snapshot: dict | None = None,
) -> dict:
    """Check all clusters for staleness based on their decay profile.

    Returns a report dict with:
        - flagged: list of clusters needing attention
        - summary: counts per action
        - compared_to: previous snapshot date (if any)
    """
    now = datetime.now()
    classified_at = decay.get("classified_at", "")
    if classified_at:
        try:
            last_classified = datetime.strptime(classified_at, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            last_classified = now
    else:
        last_classified = now

    months_since = (now - last_classified).days / 30.0
    flagged = []
    actions = {"promote_evergreen": 0, "demote_semi_stable": 0, "remove_ephemeral": 0, "no_change": 0}

    for key, info in decay.get("classifications", {}).items():
        profile = info.get("decay_profile", "semi-stable")
        retention_months = DECAY_PROFILES.get(profile, {}).get("retention_months")
        cluster_info = clusters.get("clusters", {}).get(key, {})
        size = cluster_info.get("size", 0)
        bundle_count = len(cluster_info.get("bundle_ids", []))
        keywords = cluster_info.get("keywords", [])

        flag = {
            "cluster_id": key,
            "current_profile": profile,
            "keywords": keywords,
            "size": size,
            "bundle_count": bundle_count,
        }

        # Ephemeral: past retention → flag for removal
        if profile == "ephemeral" and retention_months is not None and months_since > retention_months:
            flag["action"] = "remove_ephemeral"
            flag["reason"] = f"Past retention window ({months_since:.0f} months > {retention_months})"
            flagged.append(flag)
            actions["remove_ephemeral"] += 1
            continue

        # Semi-stable: past retention → flag for demotion to ephemeral
        if profile == "semi-stable" and retention_months is not None and months_since > retention_months:
            flag["action"] = "demote_semi_stable"
            flag["reason"] = f"Past retention window ({months_since:.0f} months > {retention_months})"
            flagged.append(flag)
            actions["demote_semi_stable"] += 1
            continue

        # Evergreen: no new activity in a long time → flag for human review
        if profile == "evergreen" and months_since > 24:
            flag["action"] = "review_evergreen"
            flag["reason"] = f"No reclassification in {months_since:.0f} months"
            flagged.append(flag)
            actions["promote_evergreen"] += 1
            continue

        actions["no_change"] += 1

    # Detect changes from snapshot
    changes = []
    if snapshot:
        prev = snapshot.get("profiles_snapshot", {})
        for key, info in decay.get("classifications", {}).items():
            old = prev.get(key, {})
            new_profile = info.get("decay_profile")
            old_profile = old.get("decay_profile")
            if old_profile and new_profile and old_profile != new_profile:
                changes.append({
                    "cluster_id": key,
                    "from": old_profile,
                    "to": new_profile,
                })

    return {
        "reviewed_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "classified_at": classified_at,
        "months_since_classification": round(months_since, 1),
        "clusters_total": len(decay.get("classifications", {})),
        "flagged": flagged,
        "profile_changes": changes,
        "summary": actions,
    }


def save_snapshot(
    decay: dict,
    snapshot_path: str = DEFAULT_SNAPSHOT_PATH,
) -> None:
    """Save a snapshot of current decay profiles for future comparison."""
    snapshot = {
        "taken_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "profiles_snapshot": {
            key: info
            for key, info in decay.get("classifications", {}).items()
        },
        "total_clusters": len(decay.get("classifications", {})),
    }
    os.makedirs(os.path.dirname(snapshot_path) or ".", exist_ok=True)
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)


def load_snapshot(
    snapshot_path: str = DEFAULT_SNAPSHOT_PATH,
) -> dict | None:
    """Load previous snapshot, or None if no snapshot exists."""
    try:
        with open(snapshot_path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_review(
    decay_path: str = DEFAULT_DECAY_PATH,
    clusters_path: str = DEFAULT_CLUSTERS_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    snapshot_path: str = DEFAULT_SNAPSHOT_PATH,
    regenerate: bool = False,
) -> str:
    """Run the review cycle: check staleness, save snapshot, optionally regenerate.

    Returns path to the review report JSON file.
    """
    decay, clusters = load_current_state(decay_path, clusters_path)
    snapshot = load_snapshot(snapshot_path)

    report = check_staleness(decay, clusters, snapshot)

    # Always save a new snapshot for next comparison
    save_snapshot(decay, snapshot_path)

    # Write report
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Optionally regenerate output if there are flagged items
    if regenerate and report["flagged"]:
        print(f"  {len(report['flagged'])} items flagged — regenerating digest + syllabus...")
        regenerate_digest(generate_insight=True)
        regenerate_syllabus()

    return output_path


if __name__ == "__main__":
    output = run_review()
    with open(output) as f:
        report = json.load(f)

    print(f"Review written to: {output}")
    print(f"  Reviewed at: {report['reviewed_at']}")
    print(f"  Months since classification: {report['months_since_classification']}")
    print(f"  Clusters: {report['clusters_total']}")
    print(f"  Summary: {report['summary']}")
    if report["flagged"]:
        print(f"  Flagged: {len(report['flagged'])}")
        for item in report["flagged"]:
            print(f"    [{item['action']}] {item['reason']} — {item['keywords'][:3]}")
    if report["profile_changes"]:
        print(f"  Profile changes: {len(report['profile_changes'])}")
        for change in report["profile_changes"]:
            print(f"    Cluster {change['cluster_id']}: {change['from']} → {change['to']}")

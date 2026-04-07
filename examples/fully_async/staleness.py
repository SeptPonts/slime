from dataclasses import dataclass

from slime.utils.types import Sample

UNKNOWN_VERSION_SENTINEL = -1


@dataclass(frozen=True)
class StalenessDecision:
    keep: bool
    group_weight_version: int | None
    reason: str | None = None


@dataclass
class StalenessTracker:
    latest_seen_weight_version: int | None = None


def parse_weight_version(raw_version) -> int | None:
    if isinstance(raw_version, int):
        return raw_version
    if isinstance(raw_version, str):
        raw_version = raw_version.strip()
        if not raw_version:
            return None
        try:
            return int(raw_version)
        except ValueError:
            return None
    return None


def get_group_weight_version(group: list[Sample]) -> int | None:
    versions = []
    for sample in group:
        for raw_version in sample.weight_versions:
            parsed_version = parse_weight_version(raw_version)
            if parsed_version is not None:
                versions.append(parsed_version)
    return max(versions) if versions else None


def evaluate_group_staleness(
    group: list[Sample],
    tracker: StalenessTracker,
    *,
    max_staleness: int | None,
    drop_unknown_version: bool,
) -> StalenessDecision:
    group_weight_version = get_group_weight_version(group)

    if group_weight_version is not None:
        current_latest = tracker.latest_seen_weight_version
        if current_latest is None or group_weight_version > current_latest:
            tracker.latest_seen_weight_version = group_weight_version

    # Preserve backward compatibility when no staleness threshold is configured.
    if max_staleness is None:
        return StalenessDecision(keep=True, group_weight_version=group_weight_version)

    if group_weight_version is None:
        if drop_unknown_version:
            return StalenessDecision(keep=False, group_weight_version=None, reason="unknown_version")
        return StalenessDecision(keep=True, group_weight_version=None)

    latest_seen = tracker.latest_seen_weight_version
    if latest_seen is not None and latest_seen - group_weight_version > max_staleness:
        return StalenessDecision(keep=False, group_weight_version=group_weight_version, reason="stale")

    return StalenessDecision(keep=True, group_weight_version=group_weight_version)


def build_staleness_metrics(
    *,
    stale_drop_count: int,
    unknown_version_drop_count: int,
    inspected_group_count: int,
    accepted_versions: list[int],
    latest_seen_weight_version: int | None,
) -> dict[str, float | int]:
    stale_drop_ratio = stale_drop_count / inspected_group_count if inspected_group_count > 0 else 0.0
    min_accepted_weight_version = min(accepted_versions) if accepted_versions else UNKNOWN_VERSION_SENTINEL
    max_seen_weight_version = (
        latest_seen_weight_version if latest_seen_weight_version is not None else UNKNOWN_VERSION_SENTINEL
    )
    return {
        "fully_async/stale_drop_count": stale_drop_count,
        "fully_async/stale_drop_ratio": stale_drop_ratio,
        "fully_async/unknown_version_drop_count": unknown_version_drop_count,
        "fully_async/max_seen_weight_version": max_seen_weight_version,
        "fully_async/min_accepted_weight_version": min_accepted_weight_version,
    }

"""L3b — sessions: bouts become sessions, and each session gets a label.

The classifier knows five classes, one per 30 s window. It has no idea what a
*session* is, and no class for a workout that mixes walking and running — interval
running, football, HIIT. Those only become visible at session scale.

So this layer does two things after the bouts are built:

1. Groups bouts into sessions. A bout is one unbroken stretch of a single label, so
   one continuous workout arrives as walking, running, walking, running. Only
   exercise (walking, running, cycling) can start a session; a short pause is
   absorbed into it, and a long one ends it.
2. Labels each session. If it contains a stretch of at least 10 minutes made of
   walking and running — both genuinely present — plus only short standing (`static`)
   or `other` pauses, the session is `other_cardio` and flagged `needs_confirmation`:
   the accelerometer cannot say which sport it was, and the user can. Otherwise it
   takes its dominant exercise class.

Pauses are allowed because real interval and field sessions have them — standing
between reps, a moment the classifier reads as `other`. They are allowed only while
short: a pause long enough to end the session (standing 2 min, `other` 5 min) ends the
stretch with it. Any other class — cycling, or an abstained `unknown` window — breaks
the stretch, because it is no longer a walk/run session.

Both classes must each fill a real share of the stretch, judged by time rather than by
counting switches. The decoder cannot follow walk/run alternation faster than about a
minute: 30 s intervals come back as one walking block and one running block, and 15 s
intervals as running throughout. A switch count would miss exactly these sessions.
"""

from __future__ import annotations

# Classes that can start a session. Only recognised exercise — otherwise ironing
# (`other`) or sitting (`static`) would open one.
OPENS = frozenset({"walking", "running", "cycling"})

# A non-exercise bout at least this long ends the session; shorter ones are absorbed.
# Stillness is strong evidence a session ended; `other` is the catch-all and only weak
# evidence of anything, so it is allowed to run longer before it counts.
BREAKS_AFTER_S = {"static": 120.0, "unknown": 120.0, "other": 300.0}
FALLBACK_BREAK_S = 120.0

MIXED_CARDIO_LABEL = "other_cardio"
MIXED_CARDIO_CLASSES = ("walking", "running")  # both must be present
MIXED_CARDIO_PAUSES = ("static", "other")      # allowed inside, only as short pauses
MIXED_CARDIO_MIN_S = 600.0     # the stretch must last at least 10 minutes
MIXED_CARDIO_MIN_SHARE = 0.20  # walking and running must each fill 20% of the stretch


def build_sessions(bouts: list[dict]) -> list[dict]:
    """Group a bout timeline into labelled sessions.

    `bouts` is the list ActivityDetector.predict() returns: dicts with `label_name`,
    `t_start_s` and `t_end_s`. Returns one dict per session:

        t_start_s, t_end_s, duration_s   the session's span
        labels                           bout labels in order, absorbed pauses included
        label_seconds                    seconds spent in each label
        session_label                    dominant exercise class, or other_cardio
        needs_confirmation               True only for other_cardio
        cardio_stretch                   the walk/run stretch that made it other_cardio
                                         ({t_start_s, t_end_s, duration_s}), or None

    Time is credited to a bout only until the next bout begins. Neighbouring bouts
    overlap by one 15 s hop, because the last window of one and the first of the next
    cover the same seconds; the switch is where the next bout starts, so the overlap
    belongs to the later bout. Counting both would inflate every switch.
    """
    ordered = sorted(bouts, key=lambda b: float(b["t_start_s"]))
    sessions: list[dict] = []

    start: float | None = None
    end = 0.0
    parts: list[tuple[str, float, float]] = []    # (label, t_start, credited seconds)
    pending: list[tuple[str, float, float]] = []

    def close() -> None:
        nonlocal start, end, parts, pending
        if start is not None:
            seconds: dict[str, float] = {}
            for name, _, sec in parts:
                seconds[name] = seconds.get(name, 0.0) + sec
            label, confirm, stretch = label_session(parts, seconds)
            sessions.append({
                "t_start_s": start,
                "t_end_s": end,
                "duration_s": end - start,
                "labels": [name for name, _, _ in parts],
                "label_seconds": seconds,
                "session_label": label,
                "needs_confirmation": confirm,
                "cardio_stretch": stretch,
            })
        start, end, parts, pending = None, 0.0, [], []

    for i, b in enumerate(ordered):
        name = str(b["label_name"])
        t0, t1 = float(b["t_start_s"]), float(b["t_end_s"])
        t_next = float(ordered[i + 1]["t_start_s"]) if i + 1 < len(ordered) else t1
        span = max(0.0, min(t1, t_next) - t0)

        if name not in OPENS:
            if (t1 - t0) >= BREAKS_AFTER_S.get(name, FALLBACK_BREAK_S):
                close()  # a long pause: the session is over
            elif start is not None:
                # Absorbed only if exercise resumes, so a session never ends on a pause.
                pending.append((name, t0, span))
            continue

        if start is None:
            start = t0
        parts.extend(pending)
        pending = []
        parts.append((name, t0, span))
        end = t1

    close()
    return sessions


def label_session(
    parts: list[tuple[str, float, float]],
    label_seconds: dict[str, float],
) -> tuple[str, bool, dict | None]:
    """Name one session from its bouts, in order.

    Looks for stretches of walking, running and short pauses. Pauses only reach this
    point if they were short enough to be absorbed into the session, so their length is
    already bounded. The longest stretch lasting at least MIXED_CARDIO_MIN_S, with
    walking and running each filling at least MIXED_CARDIO_MIN_SHARE of it, makes the
    session `other_cardio`.
    """
    best: dict | None = None
    run: list[tuple[str, float, float]] = []

    def consider(group):
        nonlocal best
        # A stretch is measured from its first to its last walking/running bout; a
        # pause at either edge belongs to whatever is next to it, not to the workout.
        while group and group[0][0] not in MIXED_CARDIO_CLASSES:
            group = group[1:]
        while group and group[-1][0] not in MIXED_CARDIO_CLASSES:
            group = group[:-1]
        if not group:
            return
        t_start = group[0][1]
        t_end = group[-1][1] + group[-1][2]
        total = sum(sec for _, _, sec in group)
        share = {c: sum(sec for n, _, sec in group if n == c) / total if total else 0.0
                 for c in MIXED_CARDIO_CLASSES}
        if (t_end - t_start) >= MIXED_CARDIO_MIN_S and all(
                v >= MIXED_CARDIO_MIN_SHARE for v in share.values()):
            if best is None or (t_end - t_start) > best["duration_s"]:
                best = {"t_start_s": t_start, "t_end_s": t_end,
                        "duration_s": t_end - t_start}

    for part in parts:
        if part[0] in MIXED_CARDIO_CLASSES or part[0] in MIXED_CARDIO_PAUSES:
            run.append(part)
        else:
            consider(run)   # cycling or unknown breaks the stretch
            run = []
    consider(run)

    if best is not None:
        return MIXED_CARDIO_LABEL, True, best

    exercise = {k: v for k, v in label_seconds.items() if k in OPENS}
    if exercise:
        return max(exercise, key=exercise.get), False, None
    return "", False, None

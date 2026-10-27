"""Review-only capture of rejected drafts during evaluation runs.

The service never returns rejected prose. Evaluation scripts wrap the validator so each
rejected draft and its issues can be saved next to the final result for human review.
"""
import wualt_slm.signal_interpretation as interpretation


def capture_rejections(store):
    """Record every rejected draft in `store`; returns the original validator."""
    original = getattr(interpretation.draft_issues, "_original", interpretation.draft_issues)

    def recording(draft, facts, *, section="distress"):
        issues = original(draft, facts, section=section)
        if issues:
            store.append({"section": section, "draft": " ".join(s.text for s in draft.sentences), "issues": issues})
        return issues

    recording._original = original
    interpretation.draft_issues = recording
    return original

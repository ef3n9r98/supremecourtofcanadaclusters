#!/usr/bin/env python3
"""
extract_opinion_groups.py
=========================

Regex/structure-based extraction of Supreme Court of Canada opinion groups from
raw SCC JSONL records. No LLMs are used.

Input
-----
A JSONL file where each row is one SCC case and contains at least:
  - citation_en
  - name_en
  - document_date_en
  - unofficial_text_en

Default expected input name:
  scc_cases_2016-2026.jsonl

Outputs
-------
By default, written into ./opinion_group_outputs/:

  case_level.csv
      One row per case. Tracks case metadata, outcome, subjects, raw Coram,
      participating judges, panel size, opinion-group count, and extraction confidence.

  judge_case_opinions.csv
      One row per judge per case. This is the canonical wide/long table.
      It preserves outcome vote, disposition side, opinion group, opinion role,
      author, subjects, non-participation exclusions, and extraction notes.

  judge_pair_case.csv
      One row per judge pair per case. Contains multiple agreement measures:
      same_outcome_vote, same_disposition_side, same_opinion_group.

  judge_pair_summary.csv
      One row per judge pair. Summarizes agreement rates.

  extraction_warnings.csv
      Cases where not all judges were assigned cleanly, no opinion group was
      found, or other warnings were generated.

Usage
-----
  python extract_opinion_groups.py scc_cases_2016-2026.jsonl \
      --outdir opinion_group_outputs --start-year 2020 --end-year 2026

Design notes
------------
This script intentionally separates extraction from interpretation.
It does not decide that one agreement measure is "correct". Instead, it gives
you enough structured columns to calculate outcome agreement, majority/minority
agreement, and opinion-group agreement later.

The most important field for your current project is:
  same_opinion_group

That means two judges joined the same set of reasons, whether that group was
majority, dissent, concurrence, or unanimous.
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set


# ---------------------------------------------------------------------------
# Judge name normalization
# ---------------------------------------------------------------------------

LAST_TO_CANONICAL = {
    "Abella": "Abella, Rosalie Silberman",
    "Brown": "Brown, Russell",
    "Cromwell": "Cromwell, Thomas Albert",
    "Côté": "Côté, Suzanne",
    "Gascon": "Gascon, Clément",
    "Jamal": "Jamal, Mahmud",
    "Karakatsanis": "Karakatsanis, Andromache",
    "Kasirer": "Kasirer, Nicholas",
    "LeBel": "LeBel, Louis",
    "McLachlin": "McLachlin, Beverley",
    "Martin": "Martin, Sheilah L.",
    "Moldaver": "Moldaver, Michael J.",
    "Moreau": "Moreau, Mary T.",
    "O'Bonsawin": "O'Bonsawin, Michelle",
    "O’Bonsawin": "O'Bonsawin, Michelle",
    "OʼBonsawin": "O'Bonsawin, Michelle",
    "Rowe": "Rowe, Malcolm",
    "Wagner": "Wagner, Richard",
    # Older names that can still appear in citations or pre-2016 edge cases
    "Bastarache": "Bastarache, Michel",
    "Binnie": "Binnie, William Ian Corneil",
    "Charron": "Charron, Louise",
    "Deschamps": "Deschamps, Marie",
    "Fish": "Fish, Morris J.",
    "Iacobucci": "Iacobucci, Frank",
    "L'Heureux-Dubé": "L'Heureux-Dubé, Claire",
    "Lamer": "Lamer, Antonio",
    "La Forest": "La Forest, Gérard Vincent",
    "Major": "Major, John C.",
    "Rothstein": "Rothstein, Marshall",
}

ALIAS_MAP = {
    "Cote": "Côté",
    "Côte": "Côté",
    "O’Bonsawin": "O'Bonsawin",
    "OʼBonsawin": "O'Bonsawin",
    "O Bonsawin": "O'Bonsawin",
    "OBonsawin": "O'Bonsawin",
    "Lebel": "LeBel",
    "Le Bel": "LeBel",
    "Mc Lachlin": "McLachlin",
    "L’Heureux-Dubé": "L'Heureux-Dubé",
    "L’Heureux-Dube": "L'Heureux-Dubé",
    "L'Heureux-Dube": "L'Heureux-Dubé",
    "LaForest": "La Forest",
    "Laforest": "La Forest",
}

HEADERS = {
    "Notes", "Decision Content", "Collection", "Date", "Judges",
    "On appeal from", "Report", "Neutral citation", "Case number",
    "Subjects", "Counsel", "Cases Cited", "Statutes and Regulations Cited",
    "Authors Cited", "Solicitors",
}


def normalize_text(s: str) -> str:
    """Normalize punctuation variants that commonly appear in SCC text."""
    if s is None:
        return ""
    return (
        s.replace("\u2019", "'")
         .replace("\u02bc", "'")
         .replace("\u2011", "-")
         .replace("\u2010", "-")
         .replace("\u2013", "-")
         .replace("\u2014", "-")
         .replace("\xa0", " ")
    )


def canonical_from_last(last: str) -> Optional[str]:
    last = normalize_text(last).strip()
    last = re.sub(r"\s+", " ", last)
    last = ALIAS_MAP.get(last, last)
    return LAST_TO_CANONICAL.get(last)


def canonicalize_full_name(raw: str) -> Optional[str]:
    """Convert 'Wagner, Richard' or 'O’Bonsawin, Michelle' to canonical name."""
    raw = normalize_text(raw).strip()
    if not raw:
        return None
    last = raw.split(",", 1)[0].strip()
    return canonical_from_last(last) or raw


def parse_judges_field(text: str) -> List[str]:
    """Extract canonical judge names from the structured 'Judges' metadata field."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if line == "Judges" and i + 1 < len(lines):
            judges = []
            for part in lines[i + 1].split(";"):
                canon = canonicalize_full_name(part)
                if canon and canon not in judges:
                    judges.append(canon)
            return judges
    return []


def parse_coram_field(text: str) -> List[str]:
    """Extract judge names from the first English Coram line.

    Some modern SCC records, especially translations or unusual releases, have
    a usable ``Coram:`` line but no structured ``Judges`` metadata field. The
    voting panel should then come from the Coram line.
    """
    lines = [normalize_text(l.strip()) for l in text.splitlines() if l.strip()]
    for line in lines:
        if line.startswith("Coram:"):
            return extract_judge_names(line)
    return []


def detect_non_participating_judges(text: str, candidate_judges: List[str]) -> List[str]:
    """Find judges expressly excluded from the final disposition.

    Example SCC footnote:
        * Brown J. did not participate in the final disposition of the judgment.

    Important performance note:
        Do NOT run broad ``[^.\n]* ...`` regexes over the entire judgment body.
        SCC judgments contain very long lines/paragraphs, and that can become
        extremely slow. This function scans only short lines/sentences that
        contain the relevant trigger phrases.
    """
    text_norm = normalize_text(text)
    non_participating: List[str] = []

    trigger_re = re.compile(
        r"did\s+not\s+participate|did\s+not\s+take\s+part|took\s+no\s+part",
        flags=re.I,
    )
    disposition_re = re.compile(
        r"did\s+not\s+participate\s+in\s+the\s+(?:final\s+)?disposition\s+of\s+the\s+judgment"
        r"|did\s+not\s+take\s+part\s+in\s+the\s+judgment"
        r"|took\s+no\s+part\s+in\s+the\s+judgment",
        flags=re.I,
    )

    # First pass: normal line-based scan. The SCC footnote is usually its own line.
    for raw_line in text_norm.splitlines():
        line = raw_line.strip()
        if not line or not trigger_re.search(line):
            continue
        if not disposition_re.search(line):
            continue
        names = extract_judge_names(line, candidate_judges or None)
        for name in names:
            if name not in non_participating:
                non_participating.append(name)

    # Second pass: some scraped lines can be glued together. Inspect short windows
    # around trigger phrases rather than the whole judgment.
    if not non_participating:
        for m in trigger_re.finditer(text_norm):
            start = max(0, m.start() - 160)
            end = min(len(text_norm), m.end() + 220)
            window = text_norm[start:end]
            if not disposition_re.search(window):
                continue
            names = extract_judge_names(window, candidate_judges or None)
            for name in names:
                if name not in non_participating:
                    non_participating.append(name)

    return non_participating


# Names in attribution strings: "Wagner C.J.", "Karakatsanis, Côté and Rowe JJ."
JUDGE_REF_RE = re.compile(
    r"\b([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+)?)\s+(?:C\.J\.|J\.)"
)

JJ_CHUNK_RE = re.compile(
    r"([A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+)?"
    r"(?:\s*,\s*[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+)?)*"
    r"(?:\s+and\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+(?:\s+[A-ZÀ-Ÿ][A-Za-zÀ-ÿ'\-]+)?)?)\s+JJ\."
)


def split_name_list(chunk: str) -> List[str]:
    chunk = normalize_text(chunk)
    chunk = re.sub(r"\s+", " ", chunk.strip())
    parts = re.split(r"\s*,\s*|\s+and\s+", chunk)
    return [p.strip() for p in parts if p.strip()]


def extract_judge_names(fragment: str, sitting: Optional[List[str]] = None) -> List[str]:
    """Extract canonical judge names from an attribution fragment.

    For SCC header attributions, a robust last-name scan is safer than trying to
    parse every possible comma/and/JJ. grammar. We scan for known SCC judge last
    names and return them in text order. This is intentionally used only on the
    structured attribution lines, not on the full judgment body.
    """
    fragment = normalize_text(fragment)
    hits: List[Tuple[int, str]] = []

    # SCC attributions sometimes say "The Chief Justice" instead of "Wagner C.J."
    # or "McLachlin C.J." In the structured Judges field, the Chief Justice is
    # conventionally listed first, so use sitting[0] when that phrase appears.
    if sitting and re.search(r"\b(?:The\s+)?Chief\s+Justice\b", fragment, re.I):
        hits.append((0, sitting[0]))

    # Include aliases as searchable surface forms, but map them to canonical names.
    surface_to_canon: Dict[str, str] = {}
    for last, canon in LAST_TO_CANONICAL.items():
        surface_to_canon[normalize_text(last)] = canon
    for alias, last in ALIAS_MAP.items():
        canon = canonical_from_last(last)
        if canon:
            surface_to_canon[normalize_text(alias)] = canon

    # Longest surface forms first helps with names like L'Heureux-Dubé / La Forest.
    for surface, canon in sorted(surface_to_canon.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r"(?<![A-Za-zÀ-ÿ'])" + re.escape(surface) + r"(?![A-Za-zÀ-ÿ'])"
        for m in re.finditer(pattern, fragment):
            hits.append((m.start(), canon))

    hits.sort(key=lambda x: x[0])
    found: List[str] = []
    for _, canon in hits:
        if canon not in found:
            found.append(canon)

    if sitting:
        sitting_set = set(sitting)
        found = [j for j in found if j in sitting_set]
    return found


def get_year(case: Dict[str, Any]) -> Optional[int]:
    date = case.get("document_date_en") or ""
    if len(date) >= 4:
        try:
            return int(date[:4])
        except ValueError:
            return None
    return None


def get_case_id(case: Dict[str, Any]) -> str:
    return case.get("citation_en") or case.get("citation2_en") or case.get("url_en") or ""


def extract_subjects(text: str) -> List[str]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if line == "Subjects":
            tags = []
            j = i + 1
            while j < len(lines) and lines[j] not in HEADERS:
                tags.append(lines[j])
                j += 1
            return tags
    return []


def extract_appeal_from(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if line == "On appeal from" and i + 1 < len(lines):
            return lines[i + 1]
    return ""


def extract_case_number(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i, line in enumerate(lines):
        if line == "Case number" and i + 1 < len(lines):
            return lines[i + 1]
    m = re.search(r"Docket:\s*([0-9A-Za-z\-]+)", text)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# Outcome extraction
# ---------------------------------------------------------------------------


def normalize_outcome_phrase(s: str) -> Tuple[str, str]:
    """Return (outcome, method_detail)."""
    lower = normalize_text(s).lower()

    # Prefer more specific partial outcomes before broad ones.
    if re.search(r"appeal\s+(?:should be\s+)?allowed\s+in\s+part", lower):
        return "allowed_in_part", "allowed_in_part_phrase"
    if re.search(r"appeal\s+(?:should be\s+)?dismissed\s+in\s+part", lower):
        return "dismissed_in_part", "dismissed_in_part_phrase"
    if re.search(r"appeal\s+(?:is\s+|was\s+|should be\s+)?allowed", lower):
        return "allowed", "allowed_phrase"
    if re.search(r"appeal\s+(?:is\s+|was\s+|should be\s+)?dismissed", lower):
        return "dismissed", "dismissed_phrase"
    return "unknown", "no_outcome_phrase"


def extract_case_outcome(text: str) -> Tuple[str, str]:
    """Best-effort case-level appeal outcome from Held/Judgment/disposition text."""
    text_norm = normalize_text(text)
    lines = [l.strip() for l in text_norm.splitlines() if l.strip()]

    # 1) Held line is usually the cleanest.
    for line in lines:
        if line.startswith("Held:"):
            out, detail = normalize_outcome_phrase(line)
            if out != "unknown":
                return out, "held_" + detail

    # 2) Oral judgment / judgment block early in document.
    early = "\n".join(lines[:220])
    out, detail = normalize_outcome_phrase(early)
    if out != "unknown":
        return out, "early_text_" + detail

    # 3) End-of-document disposition line.
    tail = "\n".join(lines[-80:])
    out, detail = normalize_outcome_phrase(tail)
    if out != "unknown":
        return out, "tail_text_" + detail

    return "unknown", "not_found"


def opposite_outcome(outcome: str) -> str:
    if outcome == "allowed":
        return "dismissed"
    if outcome == "dismissed":
        return "allowed"
    if outcome == "allowed_in_part":
        return "partial_or_unknown"
    if outcome == "dismissed_in_part":
        return "partial_or_unknown"
    return "unknown"


# ---------------------------------------------------------------------------
# Opinion-group extraction
# ---------------------------------------------------------------------------

OPINION_HEADER_PATTERNS = [
    (re.compile(r"^Unanimous Judgment Read By:?$", re.I), "unanimous"),
    (re.compile(r"^Unanimous Reasons:?$", re.I), "unanimous"),
    (re.compile(r"^Reasons for Judgment:?$", re.I), "main"),
    (re.compile(r"^Reasons for the Judgment:?$", re.I), "main"),
    (re.compile(r"^Joint Reasons:?$", re.I), "main"),
    (re.compile(r"^Majority Reasons:?$", re.I), "main"),
    (re.compile(r"^Judgment Read By:?$", re.I), "main"),
    (re.compile(r"^Judgment Delivered By:?$", re.I), "main"),

    # Mixed/partial labels are treated as distinct opinion groups.
    # Their outcome vote is left partial/unknown because regex alone cannot
    # always tell which part of the disposition they accepted.
    (re.compile(r"^Reasons Concurring in Part and Dissenting in Part:?$", re.I), "mixed_partial"),
    (re.compile(r"^Reasons Dissenting in Part and Concurring in Part:?$", re.I), "mixed_partial"),
    (re.compile(r"^Reasons Concurring in Part:?$", re.I), "concurrence_in_part"),
    (re.compile(r"^Reasons Dissenting in Part:?$", re.I), "dissent_in_part"),

    (re.compile(r"^Concurring Reasons:?$", re.I), "concurrence"),
    (re.compile(r"^Reasons Concurring(?: in (?:the )?Result)?:?$", re.I), "concurrence"),
    (re.compile(r"^Dissenting Reasons:?$", re.I), "dissent"),
    (re.compile(r"^Reasons Dissenting:?$", re.I), "dissent"),
]

PARA_LINE_RE = re.compile(r"^\(?\s*paras?\.\s*[^)]*\)?$", re.I)


def classify_header(line: str) -> Optional[str]:
    line = normalize_text(line.strip())
    for pat, label in OPINION_HEADER_PATTERNS:
        if pat.match(line):
            return label
    return None


def opinion_type_to_disposition_side(opinion_type: str) -> str:
    if opinion_type == "unanimous":
        return "unanimous"
    if opinion_type == "dissent":
        return "minority"
    if opinion_type in ("dissent_in_part", "mixed_partial"):
        return "partial_dissent"
    if opinion_type in ("concurrence", "concurrence_in_part"):
        return "concurrence"
    if opinion_type in ("main", "main_inferred"):
        return "majority_or_main"
    return "unknown"


def outcome_for_group(opinion_type: str, case_outcome: str) -> Tuple[str, str]:
    if opinion_type in ("main", "main_inferred", "concurrence", "unanimous"):
        return case_outcome, "same_as_case_outcome"
    if opinion_type == "dissent":
        return opposite_outcome(case_outcome), "opposite_case_outcome_assumption"
    if opinion_type in ("dissent_in_part", "concurrence_in_part", "mixed_partial"):
        return "partial_or_unknown", opinion_type
    return "unknown", "unknown_opinion_type"


def find_header_region(lines: List[str]) -> List[str]:
    """Return the structured header region after the first Coram line and before Note/Indexed/Present.

    SCC cases usually place the opinion summary directly after the first Coram line:
      Coram: ...
      Reasons for Judgment:
      (paras. 1 to 154)
      Wagner C.J. (... JJ. concurring)
      Concurring Reasons:
      ...
      Note: ...
    """
    start = None
    for i, line in enumerate(lines):
        if line.startswith("Coram:"):
            start = i + 1
            break
    if start is None:
        return lines[:160]

    end = len(lines)
    for j in range(start, min(len(lines), start + 100)):
        if lines[j].startswith("Note:") or lines[j].startswith("Indexed as:") or lines[j] == "Counsel:":
            end = j
            break
    return lines[start:end]


def attribution_with_continuation(region: List[str], start_idx: int) -> Tuple[str, int]:
    """Return an attribution line plus wrapped continuation lines.

    SCC header attributions sometimes wrap like:
        Martin J.
        (Karakatsanis, O'Bonsawin and Moreau JJ. concurring)

    The first script missed those continuations, creating false unassigned
    judges. This function appends short parenthetical/continuation lines until
    the attribution is complete or another opinion header begins.
    """
    attribution = region[start_idx]
    j = start_idx + 1

    def paren_balance(x: str) -> int:
        return x.count("(") - x.count(")")

    appended = 0
    while j < len(region) and appended < 4:
        nxt = region[j]
        if classify_header(nxt) or PARA_LINE_RE.match(nxt):
            break
        if nxt.startswith("Note:") or nxt.startswith("Indexed as:") or nxt == "Counsel:":
            break

        should_append = False
        if paren_balance(attribution) > 0:
            should_append = True
        if nxt.startswith("("):
            should_append = True
        if re.search(r"\b(?:concurring|dissenting)\b", nxt, re.I):
            should_append = True

        if not should_append:
            break

        attribution = attribution + " " + nxt
        j += 1
        appended += 1

    return attribution, j - 1


def opinion_type_from_attribution(attribution: str, idx: int) -> str:
    low = normalize_text(attribution).lower()
    if "dissenting in part" in low and "concurring in part" in low:
        return "mixed_partial"
    if "concurring in part" in low and "dissenting in part" in low:
        return "mixed_partial"
    if "dissenting in part" in low:
        return "dissent_in_part"
    if "concurring in part" in low:
        return "concurrence_in_part"
    if "dissenting" in low:
        return "dissent"
    if "concurring" in low:
        return "concurrence"
    # First unlabelled Per group is normally the main/majority reasons.
    return "main" if idx == 0 else "main"


def make_group(group_id: str, opinion_type: str, header: str, paras: str,
               attribution: str, authors: List[str], members: List[str],
               case_outcome: str) -> Dict[str, Any]:
    group_outcome, group_outcome_method = outcome_for_group(opinion_type, case_outcome)
    return {
        "opinion_group": group_id,
        "opinion_type": opinion_type,
        "header": header,
        "paras": paras,
        "attribution": attribution,
        "authors": authors,
        "members": members,
        "outcome_vote": group_outcome,
        "outcome_vote_method": group_outcome_method,
        "disposition_side": opinion_type_to_disposition_side(opinion_type),
    }


def coverage(groups: List[Dict[str, Any]], sitting: List[str]) -> int:
    assigned: Set[str] = set()
    for g in groups:
        if g.get("opinion_group") == "UNASSIGNED":
            continue
        for judge in g.get("members", []):
            if judge in sitting:
                assigned.add(judge)
    return len(assigned)


def normalised_group_signature(g: Dict[str, Any]) -> Tuple[str, Tuple[str, ...], str]:
    """Signature used to drop exact duplicate groups produced by repeated headnote text."""
    members = tuple(sorted(g.get("members", [])))
    opinion_type = g.get("opinion_type", "")
    attribution = re.sub(r"\s+", " ", g.get("attribution", "")).strip()
    return (opinion_type, members, attribution)


def dedupe_exact_groups(groups: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Remove exact duplicate groups and renumber remaining groups G1, G2, ..."""
    seen = set()
    out: List[Dict[str, Any]] = []
    changed = False
    for g in groups:
        sig = normalised_group_signature(g)
        if sig in seen:
            changed = True
            continue
        seen.add(sig)
        out.append(g)
    for idx, g in enumerate(out, start=1):
        if g.get("opinion_group") != "UNASSIGNED":
            g["opinion_group"] = f"G{idx}"
    return out, changed


def has_overlapping_members(groups: List[Dict[str, Any]], sitting: List[str]) -> bool:
    """True if a judge appears in more than one non-UNASSIGNED group."""
    counts: Counter[str] = Counter()
    sitting_set = set(sitting)
    for g in groups:
        if g.get("opinion_group") == "UNASSIGNED":
            continue
        for judge in g.get("members", []):
            if judge in sitting_set:
                counts[judge] += 1
    return any(v > 1 for v in counts.values())


def candidate_groups_are_usable(groups: List[Dict[str, Any]], sitting: List[str]) -> bool:
    """Opinion-group candidates must not assign the same judge to multiple groups."""
    if not groups:
        return False
    return not has_overlapping_members(groups, sitting)


def group_assignment_status(groups: List[Dict[str, Any]], sitting: List[str]) -> Dict[str, Any]:
    """Validate whether a candidate extraction cleanly accounts for the panel.

    A clean opinion-group extraction for the judge-case table must satisfy:
      - at least one non-UNASSIGNED group
      - no judge assigned to more than one non-UNASSIGNED group
      - no extracted judge outside the sitting panel
      - all sitting judges assigned exactly once

    This is the key rule that prevents mixing structured SCC header groups with
    later headnote ``Per ...`` groups. If the formal header already covers the
    panel exactly once, it is accepted and lower-priority sources are ignored.
    """
    sitting_set = set(sitting)
    counts: Counter[str] = Counter()
    extras: Set[str] = set()
    real_groups = [g for g in groups if g.get("opinion_group") != "UNASSIGNED"]

    for g in real_groups:
        for judge in g.get("members", []):
            if judge in sitting_set:
                counts[judge] += 1
            else:
                extras.add(judge)

    duplicates = sorted([j for j, n in counts.items() if n > 1])
    missing = sorted([j for j in sitting if counts[j] == 0])
    assigned = sorted(counts.keys())

    return {
        "has_groups": bool(real_groups),
        "complete": bool(real_groups) and not missing and not duplicates and not extras,
        "usable": bool(real_groups) and not duplicates and not extras,
        "assigned_count": len(assigned),
        "missing": missing,
        "duplicates": duplicates,
        "extras": sorted(extras),
    }


def candidate_groups_are_complete(groups: List[Dict[str, Any]], sitting: List[str]) -> bool:
    return group_assignment_status(groups, sitting)["complete"]


def clean_candidate_groups(groups: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Clean parser output before comparing it to another candidate extraction."""
    return dedupe_exact_groups(groups)


def parse_per_headnote_groups(lines: List[str], sitting: List[str], case_outcome: str) -> List[Dict[str, Any]]:
    """Parse headnote groups introduced by 'Per X JJ.:'.

    This catches many 2016-2024 cases where the structured SCC header is sparse
    but the headnote says:
        Per Abella and Karakatsanis JJ.:
        Per Moldaver J. (dissenting):
    """
    groups: List[Dict[str, Any]] = []
    # Use the pre-body region. Stop before detailed reasons/citations/counsel.
    region = []
    for line in lines[:500]:
        if (line.startswith("Cases Cited") or line.startswith("Statutes and Regulations Cited")
                or line.startswith("Authors Cited") or line.startswith("APPEAL from")
                or line.startswith("APPLICATION for")):
            break
        region.append(line)

    for line in region:
        if not line.startswith("Per "):
            continue
        m = re.match(r"^Per\s+(.+?):", line)
        if not m:
            continue
        attribution = "Per " + m.group(1)
        if "The Court" in attribution:
            members = list(sitting)
            authors: List[str] = []
        else:
            members = extract_judge_names(attribution, sitting)
            authors = extract_judge_names(attribution.split("(", 1)[0], sitting)

        if not members:
            continue
        opinion_type = opinion_type_from_attribution(attribution, len(groups))
        groups.append(make_group(
            "G%d" % (len(groups) + 1),
            opinion_type,
            "headnote_per_group",
            "",
            attribution,
            authors,
            members,
            case_outcome,
        ))

    groups, _ = clean_candidate_groups(groups)
    return groups


def parse_held_fallback_groups(lines: List[str], sitting: List[str], case_outcome: str) -> List[Dict[str, Any]]:
    """Fallback to the Held line.

    This is not full opinion-group extraction, but it is better than assigning no
    judges. It creates disposition/opinion groups from the Held line when the
    structured opinion header is unavailable.
    """
    held_line = next((l for l in lines if l.startswith("Held")), "")
    if not held_line:
        return []

    low = normalize_text(held_line).lower()
    if "dissent" not in low:
        return [make_group(
            "G1", "unanimous", "held_fallback_unanimous", "", held_line,
            [], list(sitting), case_outcome
        )]

    # Pull dissenters/partial dissenters from parenthetical if present.
    paren_bits = re.findall(r"\(([^)]*dissent[^)]*)\)", held_line, flags=re.I)
    target = " ".join(paren_bits) if paren_bits else held_line
    dissenters = extract_judge_names(target, sitting)
    dissenting_set = set(dissenters)
    majority = [j for j in sitting if j not in dissenting_set]

    groups: List[Dict[str, Any]] = []
    if majority:
        groups.append(make_group("G1", "main_inferred", "held_fallback_majority", "", held_line, [], majority, case_outcome))
    if dissenters:
        opinion_type = "dissent_in_part" if "dissenting in part" in low else "dissent"
        groups.append(make_group("G%d" % (len(groups) + 1), opinion_type, "held_fallback_dissent", "", held_line, [], dissenters, case_outcome))
    return groups


def parse_appeal_dissent_fallback_groups(lines: List[str], sitting: List[str], case_outcome: str) -> List[Dict[str, Any]]:
    """Fallback for short/oral cases with a disposition sentence.

    Example:
        Appeal dismissed, Côté and Brown JJ. dissenting.

    This mirrors the old cat4 strategy, but emits opinion groups.
    """
    early = "\n".join(lines[:260])
    m = re.search(r"Appeals?\s+(?:allowed|dismissed)[^.]*?,\s*(.+?)\s+dissenting\.", early, re.I)
    if not m:
        return []

    dissenters = extract_judge_names(m.group(0), sitting)
    if not dissenters:
        return []

    dissenting_set = set(dissenters)
    majority = [j for j in sitting if j not in dissenting_set]
    groups: List[Dict[str, Any]] = []
    if majority:
        groups.append(make_group("G1", "main_inferred", "appeal_sentence_fallback_majority", "", m.group(0), [], majority, case_outcome))
    groups.append(make_group("G%d" % (len(groups) + 1), "dissent", "appeal_sentence_fallback_dissent", "", m.group(0), [], dissenters, case_outcome))
    return groups


def add_inferred_complement_group(groups: List[Dict[str, Any]], sitting: List[str], case_outcome: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """If only minority/partial groups were found, infer the unassigned complement as main.

    This is a conservative repair for cases where the header exposes a dissenting
    or partial-dissenting group but omits/parses poorly the main group. It should
    be marked medium confidence, not high.
    """
    assigned: Set[str] = set()
    for g in groups:
        if g.get("opinion_group") == "UNASSIGNED":
            continue
        assigned.update([j for j in g.get("members", []) if j in sitting])

    missing = [j for j in sitting if j not in assigned]
    real_groups = [g for g in groups if g.get("opinion_group") != "UNASSIGNED"]
    if not missing or not real_groups:
        return groups, None

    has_mainish = any(g.get("opinion_type") in ("main", "main_inferred", "unanimous") for g in real_groups)
    all_found_are_nonmain = all(g.get("opinion_type") in ("dissent", "dissent_in_part", "mixed_partial", "concurrence", "concurrence_in_part") for g in real_groups)
    if (not has_mainish) and all_found_are_nonmain:
        new_groups = list(real_groups)
        new_groups.insert(0, make_group("G_INFERRED_MAIN", "main_inferred", "inferred_main_from_complement", "",
                                        "inferred from judges not assigned to captured dissent/concurrence groups",
                                        [], missing, case_outcome))
        # Renumber for cleaner outputs.
        for idx, g in enumerate(new_groups, start=1):
            g["opinion_group"] = "G%d" % idx
        return new_groups, "inferred_main_group_from_unassigned_complement"

    return groups, None


def parse_structured_header_groups(lines: List[str], sitting: List[str], case_outcome: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse only the formal SCC opinion table immediately after the first Coram line.

    This is the gold-source parser for modern SCC cases. For example:

        Reasons for Judgment:
        (paras. 1 to 120)
        Abella J. (Karakatsanis and Martin JJ. concurring)
        Concurring Reasons:
        Wagner C.J. (Moldaver, Brown and Rowe JJ. concurring)

    If this parser covers all sitting judges exactly once, the main parser should
    accept it immediately and should NOT also add headnote ``Per ...`` groups.
    """
    warnings: List[str] = []
    region = find_header_region(lines)
    groups: List[Dict[str, Any]] = []

    i = 0
    while i < len(region):
        line = region[i]
        opinion_type = classify_header(line)
        if not opinion_type:
            i += 1
            continue

        header = line
        j = i + 1
        paras = ""
        while j < len(region) and PARA_LINE_RE.match(region[j]):
            paras = region[j]
            j += 1

        if j >= len(region):
            warnings.append("header_without_attribution:%s" % header)
            i += 1
            continue

        attribution, last_attr_idx = attribution_with_continuation(region, j)

        authors = extract_judge_names(attribution.split("(", 1)[0], sitting)
        all_names = extract_judge_names(attribution, sitting)

        if "The Court" in attribution:
            members = list(sitting)
            authors = []
        elif opinion_type == "unanimous":
            # Unanimous Judgment Read By names the reader/writer, but the group is the whole panel.
            members = list(sitting)
        else:
            members = list(all_names)

        if not members and authors:
            members = list(authors)

        groups.append(make_group(
            "G%d" % (len(groups) + 1),
            opinion_type,
            header,
            paras,
            attribution,
            authors,
            members,
            case_outcome,
        ))
        i = last_attr_idx + 1

    groups, deduped = clean_candidate_groups(groups)
    if deduped:
        warnings.append("deduped_exact_header_groups")
    return groups, warnings


def choose_best_fallback_candidate(
    current_groups: List[Dict[str, Any]],
    candidate_groups: List[Dict[str, Any]],
    sitting: List[str],
) -> bool:
    """Return True when a lower-priority candidate should replace current groups.

    This is intentionally conservative. A lower-priority parser can replace the
    current candidate only if it is usable and improves panel coverage. It cannot
    be *combined* with the current groups.
    """
    candidate_status = group_assignment_status(candidate_groups, sitting)
    current_status = group_assignment_status(current_groups, sitting)
    return bool(
        candidate_status["usable"]
        and candidate_status["assigned_count"] > current_status["assigned_count"]
    )


def parse_opinion_groups(text: str, sitting: List[str], case_outcome: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Parse opinion groups from SCC text using strict source priority.

    Priority order:
      1. Formal SCC structured opinion header after Coram.
      2. Headnote ``Per ...`` groups.
      3. Held-line fallback.
      4. Appeal-sentence fallback.
      5. Oral/unanimous fallback.

    Crucial rule: sources are alternatives, not additives. If the formal header
    covers every sitting judge exactly once, accept it and stop. This prevents
    cases like R. v. C.P. from being polluted by later headnote blocks that repeat
    or reframe the same opinion groups.
    """
    warnings: List[str] = []
    lines = [normalize_text(l.strip()) for l in text.splitlines() if l.strip()]

    # 1) Gold source: formal SCC opinion table/header.
    header_groups, header_warnings = parse_structured_header_groups(lines, sitting, case_outcome)
    warnings.extend(header_warnings)
    header_status = group_assignment_status(header_groups, sitting)

    if header_status["complete"]:
        # This is the exact scenario in R. v. C.P.: clean header, no overlap, no missing judges.
        return header_groups, warnings

    groups = header_groups if header_status["usable"] else []
    if header_groups and not header_status["usable"]:
        if header_status["duplicates"]:
            warnings.append("overlapping_header_groups_ignored:" + "|".join(header_status["duplicates"]))
        if header_status["extras"]:
            warnings.append("header_groups_extra_judges_ignored:" + "|".join(header_status["extras"]))

    # 2) Headnote Per-groups: alternative source, not additive.
    per_groups = parse_per_headnote_groups(lines, sitting, case_outcome)
    per_status = group_assignment_status(per_groups, sitting)
    if per_groups and not per_status["usable"]:
        warnings.append("ignored_overlapping_or_extra_headnote_per_groups")
    elif per_status["complete"]:
        warnings.append("used_headnote_per_groups")
        return per_groups, warnings
    elif choose_best_fallback_candidate(groups, per_groups, sitting):
        groups = per_groups
        warnings.append("used_headnote_per_groups")

    # 3) Held fallback: alternative source, not additive.
    held_groups = parse_held_fallback_groups(lines, sitting, case_outcome)
    held_status = group_assignment_status(held_groups, sitting)
    if held_status["complete"]:
        warnings.append("used_held_fallback_groups")
        return held_groups, warnings
    elif choose_best_fallback_candidate(groups, held_groups, sitting):
        groups = held_groups
        warnings.append("used_held_fallback_groups")

    # 4) Appeal sentence fallback: alternative source, not additive.
    appeal_fallback_groups = parse_appeal_dissent_fallback_groups(lines, sitting, case_outcome)
    appeal_status = group_assignment_status(appeal_fallback_groups, sitting)
    if appeal_status["complete"]:
        warnings.append("used_appeal_sentence_fallback_groups")
        return appeal_fallback_groups, warnings
    elif choose_best_fallback_candidate(groups, appeal_fallback_groups, sitting):
        groups = appeal_fallback_groups
        warnings.append("used_appeal_sentence_fallback_groups")

    # 5) Fallback: oral/unanimous short cases with "We are all of the view".
    if not groups:
        early = "\n".join(lines[:220])
        if re.search(r"\bwe are all of the view\b", early, re.I) or re.search(r"\bunanimous\b", early, re.I):
            groups = [make_group(
                "G1",
                "unanimous",
                "fallback_unanimous_oral",
                "",
                "fallback: all sitting judges",
                [],
                list(sitting),
                case_outcome,
            )]
            warnings.append("fallback_unanimous_oral")
            return groups, warnings

    # Repair: if one non-dissent group is present but only the author was captured,
    # assume all sitting judges joined. This is only for short unanimous/main cases.
    if len(groups) == 1 and groups[0]["opinion_type"] in ("main", "unanimous"):
        if len(groups[0]["members"]) == 1 and len(sitting) > 1:
            groups[0]["members"] = list(sitting)
            groups[0]["opinion_type"] = "unanimous"
            groups[0]["disposition_side"] = "unanimous"
            groups[0]["outcome_vote"] = case_outcome
            groups[0]["outcome_vote_method"] = "same_as_case_outcome_assumed_unanimous"
            warnings.append("single_main_author_assumed_unanimous")
            return groups, warnings

    groups, inferred_warning = add_inferred_complement_group(groups, sitting, case_outcome)
    if inferred_warning:
        warnings.append(inferred_warning)

    # Final safety: one judge should not appear in multiple opinion groups in this
    # judge-case table. If this happens, mark the case for manual review rather
    # than emitting duplicate judge rows or self-pairs downstream.
    final_status = group_assignment_status(groups, sitting)
    if groups and not final_status["usable"]:
        details = []
        if final_status["duplicates"]:
            details.append("duplicates=" + "|".join(final_status["duplicates"]))
        if final_status["extras"]:
            details.append("extras=" + "|".join(final_status["extras"]))
        warnings.append("overlapping_groups_unresolved_case_marked_unassigned" + (":" + ",".join(details) if details else ""))
        groups = [make_group(
            "UNASSIGNED",
            "unknown",
            "unassigned_overlap",
            "",
            "overlapping group assignments; needs manual review",
            [],
            list(sitting),
            "unknown",
        )]

    assigned: Set[str] = set()
    for g in groups:
        for judge in g["members"]:
            assigned.add(judge)

    missing = [j for j in sitting if j not in assigned]
    if missing:
        warnings.append("unassigned_judges:" + "|".join(missing))
        groups.append(make_group(
            "UNASSIGNED",
            "unknown",
            "unassigned",
            "",
            "not captured by regex/header/headnote/Held parser",
            [],
            missing,
            "unknown",
        ))

    return groups, warnings

def group_rows_for_case(case: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    text = case.get("unofficial_text_en") or ""
    citation = get_case_id(case)
    year = get_year(case)
    name = case.get("name_en") or ""
    date = case.get("document_date_en") or ""
    url = case.get("url_en") or ""
    case_number = extract_case_number(text)
    subjects = extract_subjects(text)
    appeal_from = extract_appeal_from(text)

    metadata_judges = parse_judges_field(text)
    coram_judges = parse_coram_field(text)
    raw_sitting = metadata_judges or coram_judges
    non_participating = detect_non_participating_judges(text, raw_sitting)
    sitting = [j for j in raw_sitting if j not in set(non_participating)]

    case_outcome, case_outcome_method = extract_case_outcome(text)

    warnings: List[str] = []
    if not metadata_judges and coram_judges:
        warnings.append("used_coram_as_judges_source")
    if non_participating:
        warnings.append("excluded_non_participating_judges:" + "|".join(non_participating))
    if not sitting:
        warnings.append("no_participating_judges_found")

    groups, group_warnings = parse_opinion_groups(text, sitting, case_outcome)
    warnings.extend(group_warnings)

    real_groups = [g for g in groups if g["opinion_group"] != "UNASSIGNED"]
    group_count = len(real_groups)
    assigned_count = sum(len(g["members"]) for g in groups)
    unassigned_count = sum(len(g["members"]) for g in groups if g["opinion_group"] == "UNASSIGNED")
    has_dissent = any(g["opinion_type"] in ("dissent", "dissent_in_part") for g in real_groups)
    has_concurrence = any(g["opinion_type"] == "concurrence" for g in real_groups)
    is_unanimous_reasons = (group_count == 1 and unassigned_count == 0)
    is_split_reasons = (group_count > 1)

    if not groups:
        confidence = "failed"
        warnings.append("no_opinion_groups_found")
    elif unassigned_count:
        confidence = "low"
    elif any(("fallback" in w or "assumed" in w or w.startswith("used_") or w.startswith("inferred_")) for w in warnings):
        confidence = "medium"
    else:
        confidence = "high"

    case_row = {
        "case_id": citation,
        "citation": citation,
        "case_name": name,
        "date": date,
        "year": year if year is not None else "",
        "url": url,
        "case_number": case_number,
        "appeal_from": appeal_from,
        "subjects": "|".join(subjects),
        "raw_coram_judges": "|".join(raw_sitting),
        "non_participating_judges": "|".join(non_participating),
        "participating_judges": "|".join(sitting),
        "panel_size": len(sitting),
        "sitting_judges": "|".join(sitting),
        "case_outcome": case_outcome,
        "case_outcome_method": case_outcome_method,
        "num_opinion_groups": group_count,
        "has_dissent": has_dissent,
        "has_concurrence": has_concurrence,
        "is_unanimous_reasons": is_unanimous_reasons,
        "is_split_reasons": is_split_reasons,
        "assigned_judge_rows": assigned_count,
        "unassigned_judge_count": unassigned_count,
        "extraction_confidence": confidence,
        "extraction_warnings": ";".join(warnings),
    }

    judge_rows: List[Dict[str, Any]] = []
    for g in groups:
        authors_set = set(g["authors"])
        for judge in g["members"]:
            if g["opinion_group"] == "UNASSIGNED":
                role = "unknown"
            elif judge in authors_set:
                if g["opinion_type"] == "main":
                    role = "main_author"
                elif g["opinion_type"] == "unanimous":
                    role = "unanimous_author_or_reader"
                elif g["opinion_type"] in ("concurrence", "concurrence_in_part"):
                    role = "concurrence_author"
                elif g["opinion_type"] == "dissent":
                    role = "dissent_author"
                elif g["opinion_type"] in ("dissent_in_part", "mixed_partial"):
                    role = "partial_dissent_author"
                elif g["opinion_type"] == "main_inferred":
                    role = "inferred_main_member"
                else:
                    role = "author"
            else:
                if g["opinion_type"] == "unanimous":
                    role = "joined_unanimous"
                elif g["opinion_type"] == "main":
                    role = "joined_main"
                elif g["opinion_type"] in ("concurrence", "concurrence_in_part"):
                    role = "joined_concurrence"
                elif g["opinion_type"] == "dissent":
                    role = "joined_dissent"
                elif g["opinion_type"] in ("dissent_in_part", "mixed_partial"):
                    role = "joined_partial_dissent"
                elif g["opinion_type"] == "main_inferred":
                    role = "inferred_main_member"
                else:
                    role = "joined"

            judge_rows.append({
                "case_id": citation,
                "citation": citation,
                "case_name": name,
                "date": date,
                "year": year if year is not None else "",
                "subjects": "|".join(subjects),
                "appeal_from": appeal_from,
                "judge": judge,
                "sat": True,
                "participated_in_final_disposition": True,
                "panel_size": len(sitting),
                "raw_coram_judges": "|".join(raw_sitting),
                "non_participating_judges": "|".join(non_participating),
                "case_outcome": case_outcome,
                "outcome_vote": g["outcome_vote"],
                "outcome_vote_method": g["outcome_vote_method"],
                "disposition_side": g["disposition_side"],
                "opinion_group": g["opinion_group"],
                "opinion_type": g["opinion_type"],
                "opinion_role": role,
                "opinion_author": "|".join(g["authors"]),
                "opinion_header": g["header"],
                "opinion_paras": g["paras"],
                "opinion_attribution": g["attribution"],
                "extraction_confidence": confidence,
                "needs_review": confidence in ("low", "failed"),
                "extraction_warnings": ";".join(warnings),
            })

    return case_row, judge_rows, warnings


# ---------------------------------------------------------------------------
# Derived pairwise tables
# ---------------------------------------------------------------------------


def str_bool(x: Any) -> str:
    if x is None:
        return ""
    return "true" if bool(x) else "false"


def build_pair_case_rows(judge_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_case: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in judge_rows:
        if row.get("sat"):
            by_case[row["case_id"]].append(row)

    pair_rows: List[Dict[str, Any]] = []
    for case_id, rows in by_case.items():
        # Defensive de-duplication: a malformed extraction must never create
        # judge-with-self pairs. Prefer assigned rows over UNASSIGNED rows and
        # higher-confidence rows over lower-confidence rows.
        def row_priority(r: Dict[str, Any]) -> Tuple[int, int]:
            conf_rank = {"high": 3, "medium": 2, "low": 1, "failed": 0}.get(r.get("extraction_confidence", ""), 0)
            assigned_rank = 0 if r.get("opinion_group") == "UNASSIGNED" else 1
            return (assigned_rank, conf_rank)

        best_by_judge: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            j = r["judge"]
            if j not in best_by_judge or row_priority(r) > row_priority(best_by_judge[j]):
                best_by_judge[j] = r

        rows = sorted(best_by_judge.values(), key=lambda r: r["judge"])
        for a, b in combinations(rows, 2):
            same_outcome = None
            unknown_outcomes = {"", "unknown", "partial_or_unknown"}
            if a["outcome_vote"] not in unknown_outcomes and b["outcome_vote"] not in unknown_outcomes:
                same_outcome = a["outcome_vote"] == b["outcome_vote"]

            same_disp = None
            if a["disposition_side"] and b["disposition_side"] and "unknown" not in (a["disposition_side"], b["disposition_side"]):
                same_disp = a["disposition_side"] == b["disposition_side"]

            same_group = None
            if a["opinion_group"] and b["opinion_group"] and "UNASSIGNED" not in (a["opinion_group"], b["opinion_group"]):
                same_group = a["opinion_group"] == b["opinion_group"]

            pair_rows.append({
                "case_id": case_id,
                "citation": a["citation"],
                "case_name": a["case_name"],
                "date": a["date"],
                "year": a["year"],
                "subjects": a["subjects"],
                "appeal_from": a["appeal_from"],
                "judge_a": a["judge"],
                "judge_b": b["judge"],
                "outcome_vote_a": a["outcome_vote"],
                "outcome_vote_b": b["outcome_vote"],
                "same_outcome_vote": str_bool(same_outcome),
                "disposition_side_a": a["disposition_side"],
                "disposition_side_b": b["disposition_side"],
                "same_disposition_side": str_bool(same_disp),
                "opinion_group_a": a["opinion_group"],
                "opinion_group_b": b["opinion_group"],
                "same_opinion_group": str_bool(same_group),
                "opinion_type_a": a["opinion_type"],
                "opinion_type_b": b["opinion_type"],
                "both_main_or_unanimous": str_bool(a["opinion_type"] in ("main", "main_inferred", "unanimous") and b["opinion_type"] in ("main", "main_inferred", "unanimous")),
                "both_dissent_or_partial": str_bool(a["opinion_type"] in ("dissent", "dissent_in_part", "mixed_partial") and b["opinion_type"] in ("dissent", "dissent_in_part", "mixed_partial")),
                "one_dissent_or_partial": str_bool((a["opinion_type"] in ("dissent", "dissent_in_part", "mixed_partial")) != (b["opinion_type"] in ("dissent", "dissent_in_part", "mixed_partial"))),
                "extraction_confidence": a["extraction_confidence"],
                "needs_review": str_bool(a["needs_review"] or b["needs_review"]),
            })
    return pair_rows


def bool_from_str(s: str) -> Optional[bool]:
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def build_pair_summary(pair_case_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(lambda: {
        "cases_together": 0,
        "outcome_known": 0,
        "outcome_same": 0,
        "disposition_known": 0,
        "disposition_same": 0,
        "opinion_group_known": 0,
        "opinion_group_same": 0,
        "review_rows": 0,
        "subjects": Counter(),
    })

    for row in pair_case_rows:
        key = (row["judge_a"], row["judge_b"])
        st = stats[key]
        st["cases_together"] += 1
        if row.get("needs_review") == "true":
            st["review_rows"] += 1
        for subj in row.get("subjects", "").split("|"):
            if subj:
                st["subjects"][subj] += 1

        for col, known_key, same_key in [
            ("same_outcome_vote", "outcome_known", "outcome_same"),
            ("same_disposition_side", "disposition_known", "disposition_same"),
            ("same_opinion_group", "opinion_group_known", "opinion_group_same"),
        ]:
            val = bool_from_str(row.get(col, ""))
            if val is not None:
                st[known_key] += 1
                if val:
                    st[same_key] += 1

    out: List[Dict[str, Any]] = []
    for (a, b), st in sorted(stats.items()):
        def rate(same_key: str, known_key: str) -> str:
            n = st[known_key]
            return "" if n == 0 else "%.6f" % (st[same_key] / n)

        out.append({
            "judge_a": a,
            "judge_b": b,
            "cases_together": st["cases_together"],
            "outcome_known_cases": st["outcome_known"],
            "outcome_agreement_rate": rate("outcome_same", "outcome_known"),
            "disposition_known_cases": st["disposition_known"],
            "disposition_agreement_rate": rate("disposition_same", "disposition_known"),
            "opinion_group_known_cases": st["opinion_group_known"],
            "opinion_group_agreement_rate": rate("opinion_group_same", "opinion_group_known"),
            "review_rows": st["review_rows"],
            "top_subjects": "|".join([s for s, _ in st["subjects"].most_common(5)]),
        })
    return out


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            safe = {k: row.get(k, "") for k in fields}
            w.writerow(safe)


CASE_FIELDS = [
    "case_id", "citation", "case_name", "date", "year", "url", "case_number",
    "appeal_from", "subjects", "raw_coram_judges", "non_participating_judges",
    "participating_judges", "panel_size", "sitting_judges", "case_outcome",
    "case_outcome_method", "num_opinion_groups", "has_dissent", "has_concurrence",
    "is_unanimous_reasons", "is_split_reasons", "assigned_judge_rows",
    "unassigned_judge_count", "extraction_confidence", "extraction_warnings",
]

JUDGE_CASE_FIELDS = [
    "case_id", "citation", "case_name", "date", "year", "subjects", "appeal_from",
    "judge", "sat", "participated_in_final_disposition", "panel_size",
    "raw_coram_judges", "non_participating_judges", "case_outcome", "outcome_vote", "outcome_vote_method",
    "disposition_side", "opinion_group", "opinion_type", "opinion_role",
    "opinion_author", "opinion_header", "opinion_paras", "opinion_attribution",
    "extraction_confidence", "needs_review", "extraction_warnings",
]

PAIR_CASE_FIELDS = [
    "case_id", "citation", "case_name", "date", "year", "subjects", "appeal_from",
    "judge_a", "judge_b", "outcome_vote_a", "outcome_vote_b", "same_outcome_vote",
    "disposition_side_a", "disposition_side_b", "same_disposition_side",
    "opinion_group_a", "opinion_group_b", "same_opinion_group",
    "opinion_type_a", "opinion_type_b", "both_main_or_unanimous",
    "both_dissent_or_partial", "one_dissent_or_partial", "extraction_confidence",
    "needs_review",
]

PAIR_SUMMARY_FIELDS = [
    "judge_a", "judge_b", "cases_together", "outcome_known_cases",
    "outcome_agreement_rate", "disposition_known_cases", "disposition_agreement_rate",
    "opinion_group_known_cases", "opinion_group_agreement_rate", "review_rows",
    "top_subjects",
]

WARNING_FIELDS = [
    "case_id", "citation", "case_name", "year", "extraction_confidence", "extraction_warnings",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SCC opinion groups and pairwise agreement tables without LLMs.")
    parser.add_argument("input", nargs="?", default="scc_cases_2016-2026.jsonl", help="Input SCC JSONL file")
    parser.add_argument("--outdir", default="opinion_group_outputs", help="Output directory")
    parser.add_argument("--start-year", type=int, default=None, help="Optional inclusive start year")
    parser.add_argument("--end-year", type=int, default=None, help="Optional inclusive end year")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional limit for quick tests")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    case_rows: List[Dict[str, Any]] = []
    judge_rows: List[Dict[str, Any]] = []
    warning_rows: List[Dict[str, Any]] = []
    method_counts = Counter()

    total = 0
    used = 0
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                case = json.loads(line)
            except json.JSONDecodeError as e:
                warning_rows.append({
                    "case_id": "",
                    "citation": "",
                    "case_name": "",
                    "year": "",
                    "extraction_confidence": "failed",
                    "extraction_warnings": "json_decode_error:%s" % str(e),
                })
                continue

            year = get_year(case)
            if args.start_year is not None and (year is None or year < args.start_year):
                continue
            if args.end_year is not None and (year is None or year > args.end_year):
                continue

            case_row, rows, warnings = group_rows_for_case(case)
            case_rows.append(case_row)
            judge_rows.extend(rows)
            method_counts[case_row["extraction_confidence"]] += 1
            if warnings or case_row["extraction_confidence"] in ("low", "failed"):
                warning_rows.append({
                    "case_id": case_row["case_id"],
                    "citation": case_row["citation"],
                    "case_name": case_row["case_name"],
                    "year": case_row["year"],
                    "extraction_confidence": case_row["extraction_confidence"],
                    "extraction_warnings": case_row["extraction_warnings"],
                })

            used += 1
            if args.max_cases is not None and used >= args.max_cases:
                break

    pair_case_rows = build_pair_case_rows(judge_rows)
    pair_summary_rows = build_pair_summary(pair_case_rows)

    write_csv(outdir / "case_level.csv", case_rows, CASE_FIELDS)
    write_csv(outdir / "judge_case_opinions.csv", judge_rows, JUDGE_CASE_FIELDS)
    write_csv(outdir / "judge_pair_case.csv", pair_case_rows, PAIR_CASE_FIELDS)
    write_csv(outdir / "judge_pair_summary.csv", pair_summary_rows, PAIR_SUMMARY_FIELDS)
    write_csv(outdir / "extraction_warnings.csv", warning_rows, WARNING_FIELDS)

    print("Read rows: %s" % total)
    print("Cases used: %s" % len(case_rows))
    print("Judge-case rows: %s" % len(judge_rows))
    print("Judge-pair-case rows: %s" % len(pair_case_rows))
    print("Judge-pair summary rows: %s" % len(pair_summary_rows))
    print("Output directory: %s" % outdir)
    print("Extraction confidence counts:")
    for k, v in method_counts.most_common():
        print("  %s: %s" % (k, v))


if __name__ == "__main__":
    main()

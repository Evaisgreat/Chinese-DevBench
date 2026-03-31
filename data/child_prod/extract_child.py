import re
from pathlib import Path
import pandas as pd

try:
    from opencc import OpenCC
    cc = OpenCC('t2s')  # Traditional -> Simplified
except ImportError:
    cc = None
    print("Warning: opencc not installed. Traditional characters will not be converted.")

# =========================
# 1. CONFIG
# =========================
INPUT_DIR = "/path/to/Mandarin"   # 改成CHILDES Mandarin 文件夹
OUTPUT_CSV = "child_utterances_clean.csv"

# 只保留哪些 child speaker，默认只保留 CHI
CHILD_SPEAKERS = {"CHI"}

# 是否删除没有汉字的行
DROP_NON_HANZI = False

# 是否删除完全是 xxx / yyy / www 的行
DROP_PLACEHOLDER_ONLY = True


# =========================
# 2. HELPERS
# =========================
def parse_age_to_months(age_raw: str):
    """
    Convert CHILDES age format like:
    0;08.21 -> 8.7
    2;03.10 -> 27.3
    Return None if cannot parse.
    """
    if not age_raw:
        return None

    m = re.match(r"^\s*(\d+);(\d{1,2})(?:\.(\d{1,2}))?\s*$", age_raw)
    if not m:
        return None

    years = int(m.group(1))
    months = int(m.group(2))
    days = int(m.group(3)) if m.group(3) else 0
    return round(years * 12 + months + days / 30.0, 1)


def contains_hanzi(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_chat_annotations(text: str) -> str:
    """
    Remove common CHAT annotations while trying to preserve spoken words.
    This is a practical cleaner, not a perfect CHAT parser.
    """
    if not text:
        return ""

    # Remove square-bracket annotations like:
    # [/] [//] [?] [=! laughs] [*] [% ...]
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Remove angle brackets but keep inside content:
    # <我 想> [/] -> 我 想
    text = text.replace("<", " ").replace(">", " ")

    # Remove &= actions/events, e.g. &=laughs
    text = re.sub(r"&=\S+", " ", text)

    # Remove standalone retracing markers etc.
    text = re.sub(r"\b[+][/.!?]?\b", " ", text)

    # Remove special CHAT punctuation markers often attached to tokens
    text = re.sub(r"[‡„‰↑↓⇗↗→←↘⌈⌉⌊⌋∙⌋⌈]", " ", text)

    # Remove common punctuation
    text = re.sub(r"[.?!,;:\"“”‘’(){}\[\]<>…~`•，。？！；：、】【（）《》〈〉]+", " ", text)

    # Remove stray symbols but keep Chinese, English letters, digits, underscore, apostrophe, hyphen, and spaces
    text = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_\-\s']", " ", text)

    text = normalize_whitespace(text)
    return text


def normalize_text(text: str) -> str:
    """
    Further normalize cleaned text:
    - Traditional -> Simplified (if OpenCC available)
    - normalize spaces
    """
    if not text:
        return ""

    if cc is not None:
        text = cc.convert(text)

    text = normalize_whitespace(text)
    return text


def is_placeholder_only(text: str) -> bool:
    """
    Check if text is only placeholders like xxx / yyy / www
    """
    if not text:
        return True

    tokens = text.split()
    if not tokens:
        return True

    allowed = {"xxx", "yyy", "www", "xx", "yy", "ww"}
    return all(tok.lower() in allowed for tok in tokens)


def extract_child_age_from_header(lines):
    """
    Try to extract child age from @ID lines or @Age.
    Prefer CHI age if present.
    """
    # First try @ID ... |CHI|0;08.21|...
    for line in lines:
        if line.startswith("@ID:"):
            parts = line.split("|")
            if len(parts) >= 5:
                role = parts[2].strip() if len(parts) > 2 else ""
                age = parts[3].strip() if len(parts) > 3 else ""
                if role == "CHI" and age:
                    return age

    # Fallback: any age-like pattern in header
    for line in lines:
        m = re.search(r"\b\d+;\d{1,2}(?:\.\d{1,2})?\b", line)
        if m:
            return m.group(0)

    return None


def parse_cha_file(file_path: Path, keep_speakers: set):
    """
    Parse one .cha file and return list of dict rows.
    Only keep utterances whose speaker is in keep_speakers.
    """
    rows = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="utf-16")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin1", errors="ignore")

    lines = content.splitlines()
    child_age_raw = extract_child_age_from_header(lines)
    child_age_months = parse_age_to_months(child_age_raw) if child_age_raw else None

    current_speaker = None
    current_utt = None

    for line in lines:
        line = line.rstrip()

        # New main speaker line
        if line.startswith("*"):
            # Save previous utterance
            if current_speaker is not None and current_utt is not None:
                if current_speaker in keep_speakers:
                    utterance_raw = current_utt.strip()
                    utterance_clean = remove_chat_annotations(utterance_raw)
                    utterance_norm = normalize_text(utterance_clean)
                    n_tokens = len(utterance_clean.split()) if utterance_clean else 0
                    has_hanzi = contains_hanzi(utterance_norm)

                    if DROP_PLACEHOLDER_ONLY and is_placeholder_only(utterance_norm):
                        pass
                    elif DROP_NON_HANZI and not has_hanzi:
                        pass
                    else:
                        rows.append({
                            "file_path": str(file_path),
                            "speaker": current_speaker,
                            "child_age_raw": child_age_raw,
                            "child_age_months": child_age_months,
                            "utterance_raw": utterance_raw,
                            "utterance_clean": utterance_clean,
                            "n_tokens": n_tokens,
                            "utterance_norm": utterance_norm,
                            "utterance_norm_has_hanzi": has_hanzi,
                        })

            # Start new utterance
            m = re.match(r"^\*([A-Z0-9]+):\s*(.*)$", line)
            if m:
                current_speaker = m.group(1)
                current_utt = m.group(2)
            else:
                current_speaker = None
                current_utt = None

        # Continuation line of the current main utterance
        elif line.startswith("\t") or line.startswith("  "):
            if current_utt is not None:
                current_utt += " " + line.strip()

        # Dependent tiers like %mor: %gra: ignore
        elif line.startswith("%"):
            continue

        else:
            continue

    # Save last utterance
    if current_speaker is not None and current_utt is not None:
        if current_speaker in keep_speakers:
            utterance_raw = current_utt.strip()
            utterance_clean = remove_chat_annotations(utterance_raw)
            utterance_norm = normalize_text(utterance_clean)
            n_tokens = len(utterance_clean.split()) if utterance_clean else 0
            has_hanzi = contains_hanzi(utterance_norm)

            if DROP_PLACEHOLDER_ONLY and is_placeholder_only(utterance_norm):
                pass
            elif DROP_NON_HANZI and not has_hanzi:
                pass
            else:
                rows.append({
                    "file_path": str(file_path),
                    "speaker": current_speaker,
                    "child_age_raw": child_age_raw,
                    "child_age_months": child_age_months,
                    "utterance_raw": utterance_raw,
                    "utterance_clean": utterance_clean,
                    "n_tokens": n_tokens,
                    "utterance_norm": utterance_norm,
                    "utterance_norm_has_hanzi": has_hanzi,
                })

    return rows


# =========================
# 3. MAIN
# =========================
def main():
    input_dir = Path(INPUT_DIR)
    cha_files = sorted(input_dir.rglob("*.cha"))

    if not cha_files:
        print(f"No .cha files found in: {input_dir}")
        return

    all_rows = []
    for fp in cha_files:
        rows = parse_cha_file(fp, keep_speakers=CHILD_SPEAKERS)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("No child utterances extracted.")
        return

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Done. Saved {len(df)} rows to: {OUTPUT_CSV}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
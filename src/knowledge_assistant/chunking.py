"""Turn a PDF into chunks that can be embedded, retrieved and cited.

1. Read every page and the PDF outline (the bookmarks of the document) with pypdf.
2. Clean each page: normalise ligatures, drop the running header and footer, figure captions
   and code examples, and re-join words that were split across lines.
3. Walk the pages in order and assign every line to the section it belongs to, using the outline.
   Front matter (pages numbered i, ii, ...), the table of contents and boilerplate sections such
   as "Notices" are skipped.
4. Split each section into chunks of whole sentences, with a few sentences of overlap between
   neighbouring chunks.

Chunks never cross a section boundary and never start or end in the middle of a sentence, so every
chunk reads as a complete passage and can be cited with its section and the page numbers printed
in the document (not the PDF's internal page index).
"""

from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from pypdf import PdfReader

SECTION_SEPARATOR = " > "

# Boilerplate sections that carry no knowledge, matched against outline titles at any level.
DEFAULT_SKIPPED_SECTIONS = frozenset({"Table of Contents", "Notices", "AWS Glossary"})

# A section with fewer words than this (typically a short intro right before its first
# sub-section) is merged into the sub-section that follows it instead of becoming its own chunk.
MIN_SECTION_WORDS = 25

_DOT_LEADER = re.compile(r"\.{5,}")
_ROMAN_NUMERAL = re.compile(r"[ivxlcdm]+", re.IGNORECASE)
_LINE_BREAK_HYPHEN = re.compile(r"(\w)-\n(?=\w)")
# Captions describe figures that are not part of the extracted text, e.g. "Figure 8: ...".
_FIGURE_CAPTION = re.compile(r"Figure \d+:")
# Code examples: indented lines, lines starting with a bracket, and JSON "key":value pairs.
_CODE_LINE = re.compile(r'^\s{2,}\S|^\s*[{}\[\]]|":')
# A sentence ends with . ! or ? (optionally followed by a closing quote or bracket).
_SENTENCE_END = re.compile(r"[.!?][\"\u201d\u2019)]*$")
_ABBREVIATIONS = frozenset({"e.g.", "i.e.", "etc.", "vs.", "inc.", "approx."})
# Well-Architected question IDs such as "REL 1", "SEC 3" or "PER 1" (as printed in the Lens).
_LENS_QUESTION = re.compile(r"\b(OPS|SEC|REL|PERF?|COST|SUS) (\d{1,2})\b")
# AWS service names: "AWS" or "Amazon" followed by up to three capitalised words (approximate,
# e.g. "Amazon API Gateway"). A following "AWS" or "Amazon" starts the next name.
_AWS_SERVICE = re.compile(r"\b(?:Amazon|AWS)(?: (?!AWS\b|Amazon\b)[A-Z0-9][\w-]*){1,3}")
_NOT_A_SERVICE = ("Amazon Web Services", "AWS Well-", "AWS Cloud", "AWS Region")


@dataclass(frozen=True)
class Page:
    number: int  # 1-based position in the PDF file
    label: str  # page number printed in the document, e.g. "iii" or "35"
    text: str  # raw text extracted from the page


@dataclass(frozen=True)
class Heading:
    title: str
    level: int  # 0 = top level of the outline
    page_number: int  # 1-based position in the PDF file


@dataclass(frozen=True)
class Chunk:
    key: str
    text: str
    # Outline titles from the top level down to the chunk's own section, e.g.
    # ("The pillars of the Well-Architected Framework", "Reliability pillar", "Throttling").
    headings: tuple[str, ...]
    page_start: str  # printed page numbers, used in citations
    page_end: str
    pdf_page: int  # 1-based PDF page where the chunk starts, for links such as "doc.pdf#page=55"
    pillar: str  # Well-Architected pillar the section belongs to, e.g. "Security", or ""
    lens_questions: tuple[str, ...]  # question IDs mentioned in the text, e.g. ("REL 1",)
    aws_services: tuple[str, ...]  # AWS services mentioned in the text, e.g. ("AWS Lambda",)

    @property
    def section(self) -> str:
        """The heading hierarchy as one readable path, e.g. "Pillars > Reliability pillar"."""
        return SECTION_SEPARATOR.join(self.headings)

    @property
    def pages(self) -> str:
        """Printed page range for citations, e.g. "35" or "35-36"."""
        if self.page_start == self.page_end:
            return self.page_start
        return f"{self.page_start}-{self.page_end}"

    def embedding_text(self) -> str:
        """Text sent to the embedding model. The section path gives the chunk its context."""
        return f"{self.section}\n\n{self.text}" if self.section else self.text

    def metadata(self) -> dict[str, str | int | list[str]]:
        """Everything stored next to the vector. Empty lists are left out."""
        metadata: dict[str, str | int | list[str]] = {
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pdf_page": self.pdf_page,
            "pillar": self.pillar,
        }
        if self.headings:
            metadata["headings"] = list(self.headings)
        if self.lens_questions:
            metadata["lens_questions"] = list(self.lens_questions)
        if self.aws_services:
            metadata["aws_services"] = list(self.aws_services)
        return metadata


def chunk_pdf(
    data: bytes,
    *,
    document_id: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    skipped_sections: frozenset[str] = DEFAULT_SKIPPED_SECTIONS,
) -> list[Chunk]:
    """Read a PDF and split it into chunks. See the module docstring for the steps."""
    pages, headings = read_pdf(data)
    chunks = chunk_pages(
        pages,
        headings,
        document_id=document_id,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
        skipped_sections=skipped_sections,
    )
    if not chunks:
        raise ValueError("No text could be extracted from the PDF (is it a scanned document?)")
    return chunks


def read_pdf(data: bytes) -> tuple[list[Page], list[Heading]]:
    """Extract the pages (with their printed labels) and the outline headings of a PDF."""
    reader = PdfReader(io.BytesIO(data))
    labels = list(reader.page_labels)
    pages = [
        Page(number=i + 1, label=labels[i] if i < len(labels) else str(i + 1), text=text)
        for i, text in enumerate(page.extract_text() or "" for page in reader.pages)
    ]
    headings = list(_walk_outline(reader, reader.outline, level=0))
    return pages, headings


def chunk_pages(
    pages: Sequence[Page],
    headings: Sequence[Heading],
    *,
    document_id: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
    skipped_sections: frozenset[str] = DEFAULT_SKIPPED_SECTIONS,
) -> list[Chunk]:
    """Split already-extracted pages into chunks that never cross a section boundary."""
    if not 0 <= chunk_overlap_words < chunk_size_words:
        raise ValueError("chunk_overlap_words must be >= 0 and smaller than chunk_size_words")

    chunks: list[Chunk] = []
    for section in extract_sections(pages, headings, skipped_sections):
        for start, end in sentence_windows(section.words, chunk_size_words, chunk_overlap_words):
            chunks.append(make_chunk(f"{document_id}:{len(chunks):04d}", section, start, end))
    return chunks


def make_chunk(key: str, section: Section, start: int, end: int) -> Chunk:
    """Build a chunk (with its metadata) from words `start` to `end` of a section."""
    text = " ".join(section.words[start:end])
    return Chunk(
        key=key,
        text=text,
        headings=section.path,
        page_start=section.page_labels[start],
        page_end=section.page_labels[end - 1],
        pdf_page=section.page_numbers[start],
        pillar=_pillar(section.path),
        lens_questions=_lens_questions(text),
        aws_services=_aws_services(text),
    )


def _pillar(path: Sequence[str]) -> str:
    for title in path:
        if title.lower().endswith(" pillar"):
            return title[: -len(" pillar")]
    return ""


def _lens_questions(text: str) -> tuple[str, ...]:
    return tuple(sorted({f"{name} {number}" for name, number in _LENS_QUESTION.findall(text)}))


def _aws_services(text: str) -> tuple[str, ...]:
    names = {match.rstrip(".,;:") for match in _AWS_SERVICE.findall(text)}
    return tuple(sorted(n for n in names if not n.startswith(_NOT_A_SERVICE)))


# --- Reading the PDF -------------------------------------------------------------------------


def _walk_outline(reader: PdfReader, items: Iterable, level: int) -> Iterable[Heading]:
    # pypdf returns the outline as a list where a nested list holds the children of the item
    # just before it.
    for item in items:
        if isinstance(item, list):
            yield from _walk_outline(reader, item, level + 1)
            continue
        page_index = reader.get_destination_page_number(item)
        if page_index is not None and page_index >= 0:
            yield Heading(title=_normalise(item.title), level=level, page_number=page_index + 1)


# --- Cleaning pages --------------------------------------------------------------------------


def _normalise(text: str) -> str:
    # NFKC turns typographic ligatures ("ﬁ", "ﬀ") into plain letters so that words match.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def _lines(page: Page) -> list[str]:
    """Non-empty, normalised lines of a page, without code (detected before normalising, because
    normalising removes the indentation)."""
    lines = (_normalise(raw) for raw in page.text.splitlines() if not _CODE_LINE.search(raw))
    return [line for line in lines if line]


def _is_footer(line: str, page: Page) -> bool:
    # A footer ends with the printed page number, e.g. "Operate 35".
    return line == page.label or line.endswith(f" {page.label}")


def _clean_lines(page: Page, header: str | None, has_footers: bool) -> list[str]:
    lines = _lines(page)
    if header and lines and lines[0] == header:
        lines = lines[1:]
    if has_footers and lines and _is_footer(lines[-1], page):
        lines = lines[:-1]
    return _drop_figure_captions(lines)


def _drop_figure_captions(lines: Sequence[str]) -> list[str]:
    kept: list[str] = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if _FIGURE_CAPTION.match(line):
            # A long caption wraps onto a short second line without a full stop.
            following = lines[i + 1] if i + 1 < len(lines) else ""
            skip_next = len(line) >= 85 and len(following) < 60 and not following.endswith(".")
            continue
        kept.append(line)
    return kept


def _running_header(pages: Sequence[Page]) -> str | None:
    """The first line repeated on at least half of the pages, if there is one."""
    first_lines = Counter(
        _normalise(page.text.strip().splitlines()[0]) for page in pages if page.text.strip()
    )
    if not first_lines:
        return None
    line, count = first_lines.most_common(1)[0]
    return line if count >= max(2, len(pages) // 2) else None


def _has_footers(pages: Sequence[Page]) -> bool:
    """Whether at least half of the pages end with a footer, so the last line is safe to drop."""
    count = sum(1 for page in pages if (lines := _lines(page)) and _is_footer(lines[-1], page))
    return count >= max(2, len(pages) // 2)


def _is_table_of_contents(lines: Sequence[str]) -> bool:
    return sum(1 for line in lines if _DOT_LEADER.search(line)) >= 3


def _is_front_matter(page: Page) -> bool:
    # Front matter (cover, legal notice, table of contents) is numbered i, ii, iii, ...
    return _ROMAN_NUMERAL.fullmatch(page.label) is not None


def _to_words(lines: Sequence[str]) -> list[str]:
    # Re-join words split at a line break ("purpose-\nbuilt" -> "purpose-built").
    return _LINE_BREAK_HYPHEN.sub(r"\1-", "\n".join(lines)).split()


# --- Sections --------------------------------------------------------------------------------


@dataclass
class Section:
    """The cleaned words of one outline section, with the page each word comes from."""

    path: tuple[str, ...]  # outline titles from the top level down to this section
    words: list[str] = field(default_factory=list)
    page_labels: list[str] = field(default_factory=list)  # printed page label of each word
    page_numbers: list[int] = field(default_factory=list)  # PDF page number of each word

    def add(self, words: Sequence[str], page: Page) -> None:
        self.words.extend(words)
        self.page_labels.extend([page.label] * len(words))
        self.page_numbers.extend([page.number] * len(words))


def extract_sections(
    pages: Sequence[Page],
    headings: Sequence[Heading],
    skipped_sections: frozenset[str] = DEFAULT_SKIPPED_SECTIONS,
) -> list[Section]:
    """Clean the pages and split their text into outline sections (steps 2 and 3 above)."""
    return _merge_short_sections(_split_into_sections(pages, headings, skipped_sections))


def _find_heading(lines: Sequence[str], title: str, start: int) -> tuple[int, int] | None:
    """Locate a heading printed on its own line (or wrapped over two lines).

    Returns (first line index, number of lines the heading uses), or None if it is not found.
    """
    for i in range(start, len(lines)):
        if lines[i] == title:
            return i, 1
        if i + 1 < len(lines) and f"{lines[i]} {lines[i + 1]}" == title:
            return i, 2
    return None


def _split_into_sections(
    pages: Sequence[Page], headings: Sequence[Heading], skipped_sections: frozenset[str]
) -> list[Section]:
    header = _running_header(pages)
    has_footers = _has_footers(pages)
    headings_by_page: dict[int, list[Heading]] = defaultdict(list)
    for heading in headings:
        headings_by_page[heading.page_number].append(heading)

    # Text before the first heading (or the whole document, if it has no outline) goes here.
    current = Section(path=())
    sections: list[Section] = [current]

    for page in pages:
        lines = _clean_lines(page, header, has_footers)
        skip_page = _is_front_matter(page) or _is_table_of_contents(lines)
        cursor = 0
        for heading in headings_by_page.get(page.number, []):
            # Headings that cannot be found as a line are assumed to start at the cursor.
            index, size = _find_heading(lines, heading.title, cursor) or (cursor, 0)
            if not skip_page:
                _add_lines(current, lines[cursor:index], page, skipped_sections)
            current = Section(path=(*current.path[: heading.level], heading.title))
            sections.append(current)
            cursor = index + size
        if not skip_page:
            _add_lines(current, lines[cursor:], page, skipped_sections)

    return [section for section in sections if section.words]


def _add_lines(
    section: Section, lines: Sequence[str], page: Page, skipped_sections: frozenset[str]
) -> None:
    if not any(title in skipped_sections for title in section.path):
        section.add(_to_words(lines), page)


def _merge_short_sections(sections: Sequence[Section]) -> list[Section]:
    """Merge a short intro into the sub-section that directly follows it."""
    merged: list[Section] = []
    carry: Section | None = None
    for section in sections:
        if carry is not None and section.path[: len(carry.path)] == carry.path:
            section.words[:0] = carry.words
            section.page_labels[:0] = carry.page_labels
            section.page_numbers[:0] = carry.page_numbers
        elif carry is not None:
            merged.append(carry)  # no sub-section follows, keep it as its own (short) chunk
        carry = None
        if len(section.words) < MIN_SECTION_WORDS:
            carry = section
        else:
            merged.append(section)
    if carry is not None:
        merged.append(carry)
    return merged


# --- Windows of sentences --------------------------------------------------------------------


def sentence_starts(words: Sequence[str]) -> list[int]:
    """Indexes of the words that start a sentence or a bullet point."""
    starts = [0] if words else []
    for i in range(1, len(words)):
        previous, word = words[i - 1], words[i]
        is_bullet = word == "\u2022"
        ends_sentence = (
            _SENTENCE_END.search(previous)
            and previous.lower() not in _ABBREVIATIONS
            and not previous[:-1].isdigit()  # "1." in a numbered list
            and (word[0].isupper() or word[0].isdigit() or word[0] in '"\u201c(')
        )
        if is_bullet or ends_sentence:
            starts.append(i)
    return starts


def sentence_windows(words: Sequence[str], size: int, overlap: int) -> list[tuple[int, int]]:
    """Start and end word indexes of chunks made of whole sentences.

    Each chunk holds as many whole sentences as fit in `size` words. The next chunk repeats the
    last sentences of the previous one, up to `overlap` words. A sentence longer than `size` words
    is the only thing that gets split mid-sentence.
    """
    bounds = [*sentence_starts(words), len(words)]
    sentences = [
        (start, min(start + size, end))
        for begin, end in pairwise(bounds)
        for start in range(begin, end, size)
    ]
    if not sentences:
        return []

    def length(index: int) -> int:
        return sentences[index][1] - sentences[index][0]

    windows: list[tuple[int, int]] = []
    first = 0
    while True:
        last, total = first, length(first)
        while last + 1 < len(sentences) and total + length(last + 1) <= size:
            last += 1
            total += length(last)
        windows.append((sentences[first][0], sentences[last][1]))
        if last == len(sentences) - 1:
            break
        # Start the next chunk with the trailing sentences of this one, up to `overlap` words.
        following, repeated = last + 1, 0
        while following - 1 > first and repeated + length(following - 1) <= overlap:
            following -= 1
            repeated += length(following)
        first = following

    # A last chunk that adds only a few new words is folded into the one before it.
    if len(windows) > 1 and windows[-1][1] - windows[-2][1] < size // 4:
        windows[-2:] = [(windows[-2][0], windows[-1][1])]
    return windows

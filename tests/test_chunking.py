import re
from pathlib import Path

import pytest
from fpdf import FPDF

from knowledge_assistant.chunking import (
    Heading,
    Page,
    chunk_pages,
    chunk_pdf,
    read_pdf,
    sentence_starts,
)
from knowledge_assistant.config import DEFAULT_CHUNK_OVERLAP_WORDS, DEFAULT_CHUNK_SIZE_WORDS

HEADER = "Example Whitepaper AWS Well-Architected Framework"
LENS_HEADER = "Serverless Applications Lens AWS Well-Architected Framework"
SOURCE_PDF = (
    Path(__file__).parents[1] / ".local" / "wellarchitected-serverless-applications-lens.pdf"
)


def words(count: int, prefix: str) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def sentences(count: int, length: int = 10, prefix: str = "S") -> str:
    """`count` sentences of `length` words each: "S0 word ... end. S1 word ... end."."""
    filler = " ".join(["word"] * (length - 2))
    return " ".join(f"{prefix}{i} {filler} end." for i in range(count))


def page(number: int, label: str, *lines: str) -> Page:
    """A page as pypdf extracts it: running header, content lines, footer with the page label."""
    return Page(number=number, label=label, text="\n".join([HEADER, *lines, f"Footer {label}"]))


def chunk(pages, headings, *, size=250, overlap=50):
    return chunk_pages(
        pages, headings, document_id="doc", chunk_size_words=size, chunk_overlap_words=overlap
    )


# --- Cleaning --------------------------------------------------------------------------------


def test_header_footer_ligatures_and_line_break_hyphens_are_cleaned():
    pages = [
        page(1, "1", "Body", "Use a serverless ﬁle store that is purpose-", "built for the job."),
        page(2, "2", "More text " + words(30, "m")),
    ]

    [first, _] = chunk(pages, [Heading("Body", 0, 1), Heading("Next", 0, 2)])

    assert first.text == "Use a serverless file store that is purpose-built for the job."
    assert HEADER not in first.text
    assert "Footer" not in first.text


def test_figure_captions_are_dropped_including_a_wrapped_second_line():
    long_caption = (
        "Figure 9: AWS X-Ray service map visualizing a workload using AWS Lambda, DynamoDB and"
    )
    pages = [
        page(1, "1", "Body", "Tracing helps.", long_caption, "Amazon EventBridge", "Use X-Ray."),
        page(2, "2", "Next", "Figure 10: Short caption", "Captions are gone. " + words(30, "n")),
    ]

    first, second = chunk(pages, [Heading("Body", 0, 1), Heading("Next", 0, 2)])

    assert first.text == "Tracing helps. Use X-Ray."
    assert second.text.startswith("Captions are gone.")


def test_code_examples_are_dropped():
    pages = [
        page(
            1,
            "1",
            "Body",
            "The following is an example:",
            "{",
            '    "level":"INFO",',
            '    "service":"booking"',
            "}",
            "Structured logs are easy to query.",
        ),
        page(2, "2", "Next", words(30, "n")),
    ]

    [first, _] = chunk(pages, [Heading("Body", 0, 1), Heading("Next", 0, 2)])

    assert first.text == "The following is an example: Structured logs are easy to query."


# --- Sections --------------------------------------------------------------------------------


def test_chunks_follow_the_outline_and_record_printed_page_labels():
    pages = [
        page(1, "1", "Intro", words(30, "a"), "Details", words(30, "b")),
        page(2, "2", words(30, "c"), "Next", words(30, "d")),
    ]
    headings = [Heading("Intro", 0, 1), Heading("Details", 1, 1), Heading("Next", 0, 2)]

    chunks = chunk(pages, headings)

    assert [(c.key, c.section, c.pages) for c in chunks] == [
        ("doc:0000", "Intro", "1"),
        ("doc:0001", "Intro > Details", "1-2"),
        ("doc:0002", "Next", "2"),
    ]
    assert chunks[1].headings == ("Intro", "Details")
    assert chunks[1].text == f"{words(30, 'b')} {words(30, 'c')}"
    assert chunks[1].embedding_text() == f"Intro > Details\n\n{chunks[1].text}"


def test_a_heading_wrapped_over_two_lines_is_found():
    pages = [
        page(1, "1", words(30, "a"), "A long heading that", "wraps", words(30, "b")),
        page(2, "2", "End", words(30, "e")),
    ]
    headings = [
        Heading("Start", 0, 1),
        Heading("A long heading that wraps", 0, 1),
        Heading("End", 0, 2),
    ]

    first, second, _ = chunk(pages, headings)

    assert (first.section, first.text) == ("Start", words(30, "a"))
    assert (second.section, second.text) == ("A long heading that wraps", words(30, "b"))


def test_a_heading_missing_from_the_page_text_starts_at_the_top_of_its_page():
    pages = [page(1, "1", "Start", words(30, "a")), page(2, "2", words(30, "b"))]
    headings = [Heading("Start", 0, 1), Heading("Not printed", 0, 2)]

    _, second = chunk(pages, headings)

    assert (second.section, second.text) == ("Not printed", words(30, "b"))


def test_front_matter_table_of_contents_and_boilerplate_are_skipped():
    pages = [
        page(1, "i", "Title page"),
        page(2, "ii", "Table of Contents", "Body ........ 1", "Notices ...... 2", "More ..... 3"),
        page(3, "1", "Body", words(30, "a")),
        page(4, "2", "Notices", "Customers are responsible for their own assessment."),
    ]
    headings = [
        Heading("Table of Contents", 0, 2),
        Heading("Body", 0, 3),
        Heading("Notices", 0, 4),
    ]

    chunks = chunk(pages, headings)

    assert [(c.section, c.text) for c in chunks] == [("Body", words(30, "a"))]


def test_a_table_of_contents_is_skipped_even_without_roman_page_numbers():
    pages = [
        page(1, "1", "Contents", "Body ......... 2", "Retries ...... 3", "Limits ....... 4"),
        page(2, "2", "Body", words(30, "a")),
    ]

    chunks = chunk(pages, [Heading("Body", 0, 2)])

    assert [(c.section, c.text) for c in chunks] == [("Body", words(30, "a"))]


def test_boilerplate_sections_are_skipped_at_any_outline_level():
    pages = [page(1, "1", "Body", words(30, "a")), page(2, "2", "Notices", words(30, "n"))]
    headings = [Heading("Whitepaper", 0, 1), Heading("Body", 1, 1), Heading("Notices", 1, 2)]

    chunks = chunk(pages, headings)

    assert [c.section for c in chunks] == ["Whitepaper > Body"]


def test_appendix_pages_with_letter_labels_are_kept():
    pages = [page(1, "1", "Body", words(30, "a")), page(2, "A-1", "Appendix", words(30, "x"))]

    chunks = chunk(pages, [Heading("Body", 0, 1), Heading("Appendix", 0, 2)])

    assert [(c.section, c.pages) for c in chunks] == [("Body", "1"), ("Appendix", "A-1")]


def test_a_last_line_ending_in_the_page_number_is_kept_when_pages_have_no_footer():
    pages = [
        Page(1, "1", f"{HEADER}\nBody\n{words(30, 'a')}"),
        Page(2, "2", f"{HEADER}\nRetry failed calls, but retry at most 2"),
    ]

    chunks = chunk(pages, [Heading("Body", 0, 1)])

    assert chunks[0].text.endswith("retry at most 2")


def test_a_short_intro_is_merged_into_the_sub_section_after_it():
    pages = [
        page(1, "1", "Pillar", words(10, "intro"), "Child", words(30, "c")),
        page(2, "2", "Short", words(5, "s"), "Sibling", words(30, "x")),
    ]
    headings = [
        Heading("Pillar", 0, 1),
        Heading("Child", 1, 1),
        Heading("Short", 0, 2),
        Heading("Sibling", 0, 2),
    ]

    chunks = chunk(pages, headings)

    assert [(c.section, c.text) for c in chunks] == [
        ("Pillar > Child", f"{words(10, 'intro')} {words(30, 'c')}"),
        ("Short", words(5, "s")),  # no sub-section follows, so it stays on its own
        ("Sibling", words(30, "x")),
    ]


# --- Metadata --------------------------------------------------------------------------------


def test_chunks_carry_their_heading_hierarchy_and_metadata():
    text = (
        "SEC 2 asks how you manage access. Use AWS Lambda with Amazon API Gateway. "
        "The AWS Well-Architected Framework and REL 1 are related."
    )
    pages = [page(5, "1", "Identity", text), page(6, "2", "Other", words(30, "o"))]
    headings = [
        Heading("The pillars", 0, 5),
        Heading("Security pillar", 1, 5),
        Heading("Identity", 2, 5),
        Heading("Other", 0, 6),
    ]

    first = chunk(pages, headings)[0]

    assert first.headings == ("The pillars", "Security pillar", "Identity")
    assert first.section == "The pillars > Security pillar > Identity"
    assert first.pillar == "Security"
    assert first.pdf_page == 5
    assert first.lens_questions == ("REL 1", "SEC 2")
    assert first.aws_services == ("AWS Lambda", "Amazon API Gateway")
    assert first.metadata() == {
        "text": text,
        "page_start": "1",
        "page_end": "1",
        "pdf_page": 5,
        "pillar": "Security",
        "headings": ["The pillars", "Security pillar", "Identity"],
        "lens_questions": ["REL 1", "SEC 2"],
        "aws_services": ["AWS Lambda", "Amazon API Gateway"],
    }


def test_chunks_outside_the_pillars_have_no_pillar_and_omit_empty_lists():
    pages = [page(1, "1", "Intro", words(30, "a")), page(2, "2", "Next", words(30, "n"))]

    first = chunk(pages, [Heading("Intro", 0, 1), Heading("Next", 0, 2)])[0]

    assert first.pillar == ""
    assert "lens_questions" not in first.metadata()
    assert "aws_services" not in first.metadata()


# --- Sentences and windows -------------------------------------------------------------------


def test_sentence_starts_respect_abbreviations_numbered_lists_and_bullets():
    def starts(text: str) -> list[int]:
        return sentence_starts(text.split())

    assert starts("Use a queue, e.g. Amazon SQS, here. Then retry.") == [0, 7]
    assert starts("Intro text. 1. Customers call the API. 2. The API answers.") == [0, 2, 7]
    assert starts("Topics \u2022 Compute layer \u2022 Data layer") == [0, 1, 4]


def test_long_sections_are_split_into_overlapping_chunks_of_whole_sentences():
    pages = [page(1, "1", "Long", sentences(60)), page(2, "2", "Other", words(30, "o"))]
    headings = [Heading("Long", 0, 1), Heading("Other", 0, 2)]

    chunks = [c for c in chunk(pages, headings) if c.section == "Long"]

    assert [len(c.text.split()) for c in chunks] == [250, 250, 200]
    assert all(c.text.startswith("S") and c.text.endswith("end.") for c in chunks)
    assert chunks[0].text.split()[-50:] == chunks[1].text.split()[:50]
    assert chunks[-1].text.startswith("S40 ")


def test_a_chunk_that_adds_only_a_few_words_is_folded_into_the_previous_one():
    pages = [page(1, "1", "Long", sentences(46)), page(2, "2", "Other", words(30, "o"))]
    headings = [Heading("Long", 0, 1), Heading("Other", 0, 2)]

    chunks = [c for c in chunk(pages, headings) if c.section == "Long"]

    assert [len(c.text.split()) for c in chunks] == [250, 260]
    assert chunks[-1].text.endswith("S45 word word word word word word word word end.")


def test_a_sentence_longer_than_the_chunk_size_is_split_by_words():
    pages = [page(1, "1", "Long", words(400, "w") + "."), page(2, "2", "Other", words(30, "o"))]
    headings = [Heading("Long", 0, 1), Heading("Other", 0, 2)]

    chunks = [c for c in chunk(pages, headings) if c.section == "Long"]

    assert [len(c.text.split()) for c in chunks] == [250, 150]


def test_overlap_must_be_smaller_than_the_chunk_size():
    with pytest.raises(ValueError, match="smaller than chunk_size_words"):
        chunk([page(1, "1", "text")], [], size=50, overlap=50)


# --- Reading real PDFs -----------------------------------------------------------------------


def build_pdf() -> bytes:
    """A small PDF with an outline, a running header, footers and roman-numbered front matter."""
    pdf = FPDF()
    pdf.set_font("helvetica", size=11)

    def write(text: str) -> None:
        pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")

    def new_page(label_style: str | None = None) -> None:
        pdf.add_page()
        if label_style:
            pdf.set_page_label(label_style=label_style, label_start=1)
        write(HEADER)

    new_page(label_style="r")
    write("Cover page")
    new_page(label_style="D")
    pdf.start_section("Throttling", level=0)
    write("Throttling")
    write("Throttle requests at the API layer to protect downstream systems. " * 4)
    new_page()
    pdf.start_section("Retries", level=0)
    write("Retries")
    write("Retry failed calls with exponential backoff and jitter. " * 4)
    return bytes(pdf.output())


def test_read_pdf_returns_printed_labels_and_outline():
    pages, headings = read_pdf(build_pdf())

    assert [p.label for p in pages] == ["i", "1", "2"]
    assert headings == [Heading("Throttling", 0, 2), Heading("Retries", 0, 3)]


def test_chunk_pdf_end_to_end():
    chunks = chunk_pdf(build_pdf(), document_id="doc", chunk_size_words=250, chunk_overlap_words=50)

    assert [(c.section, c.pages) for c in chunks] == [("Throttling", "1"), ("Retries", "2")]
    assert chunks[0].text.startswith("Throttle requests at the API layer")
    assert "Cover page" not in " ".join(c.text for c in chunks)


@pytest.mark.skipif(not SOURCE_PDF.exists(), reason=f"source PDF not downloaded to {SOURCE_PDF}")
def test_serverless_applications_lens():
    chunks = chunk_pdf(
        SOURCE_PDF.read_bytes(),
        document_id="lens",
        chunk_size_words=DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap_words=DEFAULT_CHUNK_OVERLAP_WORDS,
    )

    assert 100 <= len(chunks) <= 200
    assert chunks[0].section == "Serverless Applications Lens - AWS Well-Architected Framework"
    assert all(c.page_start.isdigit() and c.page_end.isdigit() for c in chunks)
    assert not any("....." in c.text or LENS_HEADER in c.text for c in chunks)
    assert not any(re.search(r"Figure \d+:", c.text) or '":' in c.text for c in chunks)
    throttling = [c for c in chunks if c.section.endswith("Foundations > Throttling")]
    assert throttling and throttling[0].page_start == "50"

# Retrieval evaluation

This evaluation was run during the PoC to choose how the whitepaper is chunked and how chunks are
retrieved. It compared six chunking strategies and six retrieval methods against a set of test
questions. Only low-cost methods were tested: none of them calls an LLM at ingest or at query
time.

- **Script:** `scripts/eval_retrieval.py`
- **Questions:** `eval/questions.json`
- **Results:** `eval/results/retrieval_eval_local.xlsx`

## Summary

- **Best combination:** section-and-sentence chunks with weighted hybrid search (alpha 0.5).
  - The evidence was ranked first for 80% of the questions and was in the top 5 for 93%.
  - Mean reciprocal rank (MRR) was 0.84.
- **Hybrid search is clearly better than either search alone.** On the same chunks:
  - keyword search: MRR 0.70;
  - vector search: MRR 0.77;
  - weighted hybrid: MRR 0.84;
  - RRF hybrid: MRR 0.83.
- **Splitting along the document's sections matters more than the exact chunk size.** Fixed
  windows that ignore sections rank the evidence first only 60% of the time.
- **Embedding model caveat:** the run used a local open-source embedding model, because Bedrock was
  not yet enabled on the PoC account. The results must be confirmed with Amazon Titan Text
  Embeddings V2, which the deployed pipeline uses.

## Setup

### Document

- *AWS Well-Architected Framework: Serverless Applications Lens*, English, July 14, 2022 edition.
  This is the same PDF the ingest function loads.
- All strategies start from the same cleaned text produced by `knowledge_assistant.chunking`:
  - Headers, footers, figure captions, code examples and the table of contents are removed.
  - The front matter is dropped, along with the "Notices" and "AWS Glossary" sections.
  - Text is assigned to sections using the PDF outline: 124 headings, giving 114 sections after
    very short intros are merged into the section that follows.

### Questions

- **Set:** 30 questions about the whitepaper, written for the PoC.
- **Question types:**
  - **Keyword (13):** use the document's own terms, for example "How does RDS Proxy help Lambda
    functions that talk to a relational database?"
  - **Paraphrase (17):** describe the problem in other words, for example "How do I avoid latency
    spikes from cold starts when traffic goes up?" The document's answer is "use provisioned
    concurrency".
- **Evidence:** each question has one or more quotes copied from the document. Every quote was
  checked against the cleaned text.
- **Hit rule:** a retrieved text answers a question when it contains one of the question's quotes.
  Case and curly quotes are ignored.

### Embedding model

| | Used in this run | Used by the deployed pipeline |
|---|---|---|
| Model | `BAAI/bge-small-en-v1.5` | Amazon Titan Text Embeddings V2 |
| Where it runs | Locally on CPU (fastembed, ONNX), free | Amazon Bedrock, pay per token |
| Dimensions | 384 | 1024 |
| Similarity | Cosine (normalised vectors) | Cosine (normalised vectors) |

- **Why a stand-in model:** Bedrock quotas on the new PoC account were 0 until AWS finished
  verifying the account.
- **Why bge-small:** it is a small, widely used English retrieval model (MIT licence) that runs on
  a laptop in seconds.
- **Query prefix:** questions are prefixed with the model's recommended instruction ("Represent
  this sentence for searching relevant passages: "). Chunks are embedded as they are.
- **Re-running with Titan:** the script supports `--embedder titan`. Expect different absolute
  numbers. The comparison between strategies and methods is what needs confirming.

## Chunking strategies

In every strategy except A and D, the searched text is the section's heading path followed by the
chunk text, for example `Reliability pillar > Foundations > Throttling`. This is the same text the
pipeline embeds (`Chunk.embedding_text()`).

| Strategy | How chunks are made | Size / overlap (words) | Chunks | Avg words |
|---|---|---|---|---|
| A. Fixed size | 250-word windows over the whole text. Ignores sections and sentences. No headings. | 250 / 50 | 100 | 250 |
| B. Sections + word windows | Split by section, then 250-word windows. Can cut sentences. (First version of the chunker.) | 250 / 50 | 153 | 144 |
| C. Sections + sentences | Split by section, then whole sentences up to 250 words. (Current chunker.) | 250 / 50 | 144 | 147 |
| D. C without headings | Same chunks as C, but the heading path is not added to the searched text. | 250 / 50 | 144 | 147 |
| E. Small sentence chunks | Same as C with smaller chunks. | 120 / 25 | 236 | 92 |
| F. Parent-child | Search the small chunks of E. Send the whole section ("parent") they belong to. | 120 / 25 | 236 searched, 114 sections sent | 92 searched |

## Retrieval methods

| Method | How it works |
|---|---|
| Keyword (BM25) | Okapi BM25 with k1 = 1.5 and b = 0.75. Lowercase words and numbers, English stopwords removed, no stemming. |
| Vector | Cosine similarity between the question's embedding and each chunk's embedding. |
| Weighted hybrid (alpha 0.25, 0.5, 0.75) | Take the top 30 from each search and rescale each list's scores to 0..1 (min-max). Score = alpha × vector + (1 − alpha) × keyword. A chunk missing from a list gets 0 from that list. |
| RRF hybrid | Reciprocal rank fusion: take the top 30 from each search and sum 1 / (60 + rank) over both lists, with equal weights. Scores are ignored, only ranks count. |

The hybrid functions are the ones the query function will use
(`knowledge_assistant.hybrid_search`).

## Metrics

- **Recall@k (k = 1, 3, 5):** the share of questions whose evidence is in the top k retrieved
  texts. Recall@5 matters most, because the top 5 are sent to the model.
- **MRR (mean reciprocal rank):** the average of 1 / rank of the first text with the evidence.
  - Rank 1 scores 1.0, rank 2 scores 0.5, and so on.
  - A question not answered in the top 10 scores 0.
- **Recall@5 by question type:** shows separately how each method handles keyword and paraphrased
  questions.
- **Words sent to the model (top 5):** the size of the context the model would receive. It is a
  proxy for generation cost.
- **Parent-child ranks:** for F, ranks count distinct parent sections.

## Results

### Evidence ranked first (Recall@1)

| Chunking | Keyword | Vector | Weighted 0.25 | Weighted 0.5 | Weighted 0.75 | RRF |
|---|---|---|---|---|---|---|
| A. Fixed size | 60% | 50% | 60% | 60% | 57% | 60% |
| B. Sections + word windows | 60% | 60% | 63% | 77% | 67% | 73% |
| C. Sections + sentences | 60% | 70% | 70% | **80%** | 70% | **80%** |
| D. C without headings | 60% | 67% | 70% | 77% | 77% | **80%** |
| E. Small sentence chunks | 40% | 53% | 57% | 57% | 60% | 57% |
| F. Parent-child | 57% | 70% | 70% | 67% | 73% | 70% |

### Evidence in the top 5 (Recall@5)

| Chunking | Keyword | Vector | Weighted 0.25 | Weighted 0.5 | Weighted 0.75 | RRF |
|---|---|---|---|---|---|---|
| A. Fixed size | 80% | 93% | 83% | 93% | 93% | 90% |
| B. Sections + word windows | 83% | 90% | 83% | 93% | 93% | 87% |
| C. Sections + sentences | 83% | 87% | 87% | 93% | 90% | 90% |
| D. C without headings | 83% | 87% | 87% | 93% | 90% | 90% |
| E. Small sentence chunks | 77% | 83% | 87% | **97%** | 87% | 90% |
| F. Parent-child | 83% | 90% | 90% | **97%** | 90% | 90% |

### Mean reciprocal rank (MRR)

| Chunking | Keyword | Vector | Weighted 0.25 | Weighted 0.5 | Weighted 0.75 | RRF |
|---|---|---|---|---|---|---|
| A. Fixed size | 0.68 | 0.65 | 0.71 | 0.75 | 0.69 | 0.73 |
| B. Sections + word windows | 0.71 | 0.72 | 0.74 | 0.83 | 0.76 | 0.80 |
| C. Sections + sentences | 0.70 | 0.77 | 0.78 | **0.84** | 0.79 | 0.83 |
| D. C without headings | 0.70 | 0.76 | 0.78 | 0.83 | 0.82 | **0.84** |
| E. Small sentence chunks | 0.57 | 0.65 | 0.68 | 0.71 | 0.72 | 0.71 |
| F. Parent-child | 0.69 | 0.79 | 0.77 | 0.78 | 0.82 | 0.78 |

At three decimals, C + weighted 0.5 scores 0.844 and D + RRF scores 0.839.

### Context size

| Chunking | Words sent to the model (top 5) |
|---|---|
| A. Fixed size | 1,250 |
| B, C, D | about 950 to 1,010 |
| E. Small sentence chunks | about 500 |
| F. Parent-child | about 1,780 to 2,040 |

## What we learned

### Keyword and vector search fail on different questions

Recall@5 on strategy C:

| Search | Keyword questions | Paraphrased questions |
|---|---|---|
| Keyword | 13 of 13 | 12 of 17 |
| Vector | 11 of 13 | 15 of 17 |
| Weighted hybrid (alpha 0.5) | 13 of 13 | 15 of 17 |

- Keyword search finds exact terms such as "REL 1" and "PutMetricData", but misses questions
  phrased in other words.
- Vector search is the other way round. For example, it ranked "What does question REL 1 cover?"
  9th.
- Weighted hybrid at alpha 0.5 keeps the best of both.

### Weighted alpha 0.5 and RRF are effectively tied

- They differ by one question at rank 1 and two questions in the top 5.
- With 30 questions, one question is 3.3 percentage points, so this is within noise.
- **Where RRF is weaker:** RRF only looks at ranks, so a chunk found by one search alone loses to
  chunks that both searches rank moderately. For "How should I manage feature flags…?", vector
  search ranked the answer 2nd and keyword search did not have it in its top 10. Weighted fusion
  kept it at 8th; RRF pushed it out of the top 10.
- **Where weighted fusion is weaker:** it depends on alpha. On C, alpha 0.25 and 0.75 were both
  worse than 0.5: MRR 0.78 and 0.79, against 0.84.

### Respecting sections and sentences helps

- A, B and C give similar top-5 recall.
- But C ranks the evidence first more often than A: 80% vs 60% with hybrid search.
- With vector search alone, C beats B on rank 1 (70% vs 60%), because B's chunks can start and end
  mid-sentence.

### Smaller chunks rank the right passage first less often

- E's 90-word chunks reach the best top-5 recall (97%), but only 57% at rank 1.
- A small chunk often holds only part of the answer, so neighbouring chunks compete with it.

### Parent-child doubles the context for no ranking gain

- F's top-5 recall is as good as C's or better: 90 to 97%.
- It ranks the evidence first less often: 67 to 73% with hybrid search, against 80% for C.
- It sends about twice as many words to the model, which roughly doubles the generation input
  cost.

### The heading prefix made no measurable difference

- C and D are within one or two questions of each other, and neither is ahead consistently.
- The heading hierarchy is still stored as chunk metadata, for citations and the source list.

### Questions that still fail

With C and weighted hybrid at alpha 0.5, the evidence was not ranked first for six questions.

| Question | Type | Rank |
|---|---|---|
| "Why is it a bad idea to have functions call each other in code to run a multi-step process?" | Paraphrase | Not in the top 10 |
| Feature flags question | Paraphrase | 8th |
| Saga pattern question | Keyword | 5th |
| Three other paraphrased questions | Paraphrase | 3rd |

**The question that missed:**

- Every method on C missed it. Only parent-child found it, at 9th.
- It shares almost no words with the passage that answers it ("Chaining Lambda executions within
  the code…").

## Decision for the PoC

**Chunking:** strategy C.

- Split by section, then whole sentences up to 250 words with 50 words of overlap. Short intro
  sections are merged into the next section.
- Each chunk keeps its heading hierarchy, its printed page numbers, the PDF page, its pillar, the
  Well-Architected question IDs it mentions and the AWS services it names.

**Retrieval:** weighted hybrid search with alpha 0.5.

- 30 candidates are taken from vector search (S3 Vectors) and from keyword search (BM25).
- The two lists are fused, and the top 5 chunks are sent to the model.
- Weighted alpha 0.5 had the best measured result. It is also less likely than RRF to bury a
  chunk that only one search finds.
- RRF is a one-line switch if the Titan re-run favours it.

## Limitations

- **Stand-in embedding model:** Titan V2 may rank differently. Re-run with
  `--embedder titan` before treating these numbers as final.
- **Small question set:** 30 questions only detect large differences. Alpha was only tried at three
  values and should not be tuned further on this set.
- **Who wrote the questions:** they were written during the PoC with the document open, by the
  people who built the chunker. Real users phrase questions differently. Questions written by the
  client's engineers would be a fairer test.
- **Hit rule:** it requires an exact quote.
  - It under-counts chunks that answer the question in other words.
  - It counts a miss when a quote is split across two chunks. This affects small chunks more.
- **Retrieval only:** the quality of the generated answers (faithfulness, correct page citations)
  was not measured.
- **Exact vs approximate search:** the evaluation searches vectors exactly. S3 Vectors uses
  approximate search, which should make no practical difference for about 150 vectors.
- **One document:** the results may not carry over to other documents.

## Methods not tested

These were left out because they add an LLM or model call per chunk or per question, which goes
against the PoC's cost constraint. Bedrock was also not available during the evaluation.

- **Semantic chunking:** splits where the embedding similarity between neighbouring sentences
  drops. It needs an embedding per sentence at ingest, and the document's outline already marks
  topic changes reliably.
- **LLM-based (agentic) chunking:** an LLM decides the chunk boundaries. It needs one LLM call per
  section at ingest.
- **Contextual retrieval:** an LLM writes a short context for each chunk before embedding. It needs
  one LLM call per chunk at ingest.
- **Re-ranking:** a re-ranking model re-orders the top candidates. It adds a model call to every
  question.
- **Query rewriting or HyDE:** an LLM rewrites the question, or drafts a hypothetical answer to
  search with. It adds an LLM call to every question.

See `docs/recommendations.md` for which of these to try after the PoC.

## Reproducing

From the repository root, after downloading the PDF to `.local/` (see the README):

```bash
uv sync --group eval
uv run --group eval python scripts/eval_retrieval.py                     # local model (downloaded from Hugging Face on first run)
uv run --group eval python scripts/eval_retrieval.py --embedder titan    # Titan V2 through Bedrock (needs AWS credentials)
uv run --group eval python scripts/eval_retrieval.py --embedder none     # keyword search only
```

`--embedder titan` calls Bedrock in eu-central-1 by default (`--region` to change). It needs
`bedrock:InvokeModel` on `amazon.titan-embed-text-v2:0`.

The script prints a summary and writes `eval/results/retrieval_eval_<embedder>.xlsx`:

| Sheet | Contents |
|---|---|
| Summary | Every strategy × method with all metrics |
| Matrix | Recall@5 and MRR as strategy × method grids |
| Per question | The rank of every question for every combination (the raw data) |
| Strategies | The chunking settings |
| Questions | The questions and their evidence |
| Settings | The run's parameters |

Every metric in the workbook is an Excel formula over the per-question ranks.

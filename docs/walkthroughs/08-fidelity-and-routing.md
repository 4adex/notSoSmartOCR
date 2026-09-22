# Fidelity repairs and routing results

Update, 22 September 2026: the native reader now retains decoder-failure evidence and retries sustained generation loops. Refinement preserves successful sibling crops while exposing rejected table and form-control readings as pending alternatives. Form proposals retain their source pixels, and pages without readable canonical evidence report failure. Optional native feature capture supports further routing experiments; no learned clinical router or reliable physical-source linker is enabled by these changes. The measurements below describe the earlier 13 September run.

Measured on 13 September 2026. The selected change preserves text surrounding a generated table and isolates malformed TableFormer responses. It does not solve overlapping recognition or establish an improvement in unseen-document OCR accuracy. The broader layout and automatic-escalation comparison is negative; complete-region Falcon recognition remains the baseline.

## Runtime and ownership

The demo was initially unavailable. An isolated baseline and candidate were started on the existing A10G host, preserving the other checkouts and unfinished work. The verified configuration is orientation → Heron layout inside the native Falcon server → complete-region Falcon recognition → evidence-risk/evidence-layout → canonical records → rendering. DocTR proposes orientation; Tesseract OSD checks it and also reads side margins. The additional app-level Heron proposal pass, Nemotron fusion, TATR, TableFormer, handwriting specialists, and external Qwen refinement are inactive. Required TATR command arguments do not prove execution. Conversely, the outer `--no-heron-proposals` option does not disable Heron inside the native wrapper: `DocumentLayoutEngine.run_layout_detection` delegates to `HeronTableDetector`. The composition label alone obscures this distinction; the README's Heron-first architecture was substantially correct.

The native Falcon installation predates the upstream output-materialization merge. Its local serving wrapper already batches generation-output transfers. The environment uses PyTorch 2.11.0+cu128, Transformers 5.16.1, python-doctr 1.1.0, bfloat16, native compilation, maximum batch size 32, and layout threshold 0.5. The recognition model is [`tiiuae/Falcon-OCR`](https://huggingface.co/tiiuae/Falcon-OCR), published by TII with Apache-2.0 licensing. The detector is [`ds4sd/docling-layout-heron-101`](https://huggingface.co/docling-project/docling-layout-heron-101), published by IBM/Docling with Apache-2.0 licensing and RT-DETR architecture lineage; it should not be represented as proof of an exclusively non-Chinese backbone. The user's all-origin research authorization applies, and these existing components were retained explicitly. The current public Falcon model card describes v1.5, but no checkpoint update was downloaded or silently adopted. Alibaba Qwen was an explicitly authorized, public-data-only comparison, not a deployment switch. Its hosted API does not establish a downloadable checkpoint or an open-weight license.

Separate app processes share the unchanged native Falcon backend. The candidate imports its own copied source tree and uses a separate feedback directory. Source edits were followed by an app restart and composition/readback checks. Restored sessions retain the outputs originally produced; they are not reruns.

## Root causes established

| Boundary | Evidence | Decision |
| --- | --- | --- |
| Distinct detector queries and recognition crops | GSoC regions 11 and 13 cover the same navigation text; region 14 reads part of the adjacent mark. Paper author containers overlap individual names. Financial table regions overlap a common source table. | Keep complete recognition context. Neither unique request IDs nor downstream deduplication assigns generated tokens to source ink. |
| Legitimate repetition | The private signature page physically has two sets of form fields and two footer strings. Its output contains extra copies of some labels/footer text. | Preserve genuine repeated fields. Earlier one-footer assumptions were incorrect. |
| Table serialization | The parser accumulated only cell data, dropping a caption or note in the same generated response. | Preserve leading and trailing text once, attached to the existing parent region. |
| Unrepresentable table topology | Multiple/nested grids or prose between rows cannot be represented faithfully as one cell grid. | Reject that structure, retain raw model evidence and review state, expose a collapsed review disclosure. Unresolved markup stays out of document Markdown. |
| Refinement exception boundary | A malformed numeric coordinate or row index escaped TableFormer's error wrapper and aborted the refinement request. | Parse within the existing protected call so the bad table is rejected while independent successful siblings survive. |

No source boxes were shrunk, expanded, suppressed, or made disjoint. No token coordinates, confidence scores, calibrated probabilities, or replacement routing were added. Caption/note geometry is explicitly the parent region's geometry, not invented text localization. Existing cell ranges and raw generation evidence remain available.

## Research decisions

All external links in this walkthrough were read through Jina on 13 September 2026. Source claims are not local benchmark results.

| Question | Primary evidence and applicability | Experiment and decision |
| --- | --- | --- |
| Did upstream Falcon fixes solve overlap? | [PR 32](https://github.com/tiiuae/Falcon-Perception/pull/32) remained open; [PR 34](https://github.com/tiiuae/Falcon-Perception/pull/34) was merged. [Issue 33](https://github.com/tiiuae/Falcon-Perception/issues/33) remained open and [issue 35](https://github.com/tiiuae/Falcon-Perception/issues/35) was closed. Materialization timing concerns output transfer, not complete OCR. | Inspected installed source and wrapper. No general overlap fix is established. |
| Can official layout postprocessing safely suppress repeated regions? | [Docling layout report](https://arxiv.org/html/2509.11720v1) and [official postprocessor](https://raw.githubusercontent.com/docling-project/docling/main/docling/utils/layout_postprocessor.py) use PDF text cells for transfer and recovery. [PaddleX processors](https://raw.githubusercontent.com/PaddlePaddle/PaddleX/develop/paddlex/inference/models/layout_analysis/processors.py) retain filtering even when NMS is disabled; a raw Transformers decode is not equivalent. | Existing local comparisons of Heron/Egret Docling and PP-DocLayout variants already lost important notes/form content. Those are reused negative development results, not new measured gains. No detector switch. |
| Does padding fix boundary tables? | [Docling issue 2965](https://github.com/docling-project/docling/issues/2965) is a report of edge-table misses and padding attempts, not a controlled universal result. | No fixed padding adopted. It cannot establish ownership of generated content. |
| Should Qwen replace selected Falcon crops? | [Qwen3.7-Flash API](https://openrouter.ai/qwen/qwen3.7-flash), verified identifier `qwen/qwen3.7-flash`, canonical slug `qwen/qwen3.7-flash-20260727`. | Eleven identical public crops, paired with existing Falcon outputs, show both corrections and regressions. Automatic escalation rejected. |
| Is a calibrated router justified? | Existing decoder evidence contains selected-token scores, not complete vocabulary distributions or token coordinates. | Current grouping misses known financial/paper mistakes. Eleven selected crops cannot train or validate expected-benefit calibration on separate documents. No weighted-score router or claimed local/remote quota. |

Prior page-level Falcon and fragmented-crop experiments were also inspected: the former omitted substantial notes/GSoC content; the latter damaged digits and complete expressions. Repeating those approaches without a supported ownership mechanism was not justified. No training, token pruning, or parallel decoding modification was introduced; their published gains would not establish drop-in compatibility here.

## Measured results

### Integration fidelity

Controlled backend responses pass through the actual HTTP reader, upload API, canonical export, Markdown download, and browser. These establish the repair's effect independently of recognition:

- A caption, 2×2 table, and trailing note previously lost the two surrounding text blocks. They now appear once in canonical text, rendered output, and Markdown, with the table cells unchanged.
- Adjacent headings, blockquotes, inline markup, Unicode, and literal HTML-like content preserve boundaries and remain safely rendered.
- Three unsupported structure forms retain original evidence and explicit review state instead of silently combining grids. The workbench offers a collapsed literal-evidence disclosure. JSON preserves the unresolved content; Markdown preserves the existing resolved-content contract.
- Malformed TableFormer responses no longer abort good sibling refinements. TableFormer remains inactive in the main demo; this robustness improvement applies to the experimental refinement path.

These are controlled response tests, not evidence that Falcon newly recognized a caption. None of the six public model examples exercised the surrounding-text repair. Their region text and source boxes remained identical in the paired runs.

### Public model examples

| Example | Baseline request and exports, seconds | Candidate request and exports, seconds | Result |
| --- | ---: | ---: | --- |
| Attention Is All You Need | 42.78 | 8.34 | Same 41 regions; overlapping author content remains. |
| GSoC Final Evaluation | 5.27 | 8.16 | Same 25 regions; repeated navigation remains. |
| Handwritten physics | 10.21 | 27.06 | Same 22 regions; complete expressions retained, symbol errors remain. |
| Financial report | 7.35 | 9.31 | Same 109 regions; duplicate table coverage and missing values remain. |
| Code | 2.81 | 4.45 | Same 20 regions; malformed table-like generation remains review evidence. |
| Scanned form | 2.75 | 4.47 | Same 43 regions; no new recognition-quality claim. |

These are single observations, not a speed comparison: baseline measurements ran on the serving host, candidate measurements included the SSH/network path, and the first baseline request included lazy initialization. Backend loading and compilation were not timed precisely enough to report cold-start latency. No speedup is claimed.

The authorized private 44-page PDF completed through the upload API in 201.15 seconds with zero API failures. That is execution success, not clinical or legal extraction accuracy. The signature page retains genuine repeated fields but also extra label/footer transcriptions. Phone-number fragments arise in overlapping source regions. Private pixels, values, feedback, and outputs remain local/private; none were sent to OpenRouter or Jina. Downstream payer/provider identity and absent-field extraction were not evaluated.

### Generalization checks and concurrency

A separately generated two-column form family was not used to choose the parser repair: upright, 90/180/270-degree rotations, 1.5× scaling, added margins, and moderate blur, plus a blank page. Together with the six public examples and a printed-calculus scan, all 15 paired cases retained identical region text, boxes, reading order, resolution, structure, and page text. Derived variants are one related document family, not independent held-out samples.

This is preservation evidence, not success on all cases. The blank image returns `orientation_views_failed` / `falcon_layout_empty` with no regions in both versions. The base synthetic form and all right-angle rotations contain 10,314 characters in page text from excessive generation; baseline/candidate requests take roughly 46-50 seconds. Since upright input fails too, this cannot be attributed solely to rotation. Scaling and blur produce 170-character outputs, but one of the two “By” labels is missing; added margins produce 160 characters and miss both. The printed-calculus scan retains the same 29 regions and 1,152 page-text characters; timing was 4.64/4.63 seconds, without an accuracy or formula-equivalence claim. These failures remain in the denominator.

Two simultaneous candidate requests (code and scanned form) completed in 4.74 and 8.29 seconds with correct separate outputs. The shared native decoder serializes work; a browser GSoC request queued behind the long synthetic case and took 46.38 seconds backend time. This demonstrates queueing and preserved response assignment, not scalable throughput. No latency optimization or VRAM minimum is established.

### Qwen comparison

Eleven public crops from the paper, GSoC, handwriting, financial table, and code examples were submitted without changing crop boundaries. Original Falcon alternatives remained untouched. The initial eleven requests with required zero-data-retention routing all failed with HTTP 404 because no matching endpoint existed. A separate public-only attempt without that requirement completed 11/11; data-collection denial remained configured. This is not an authorization or retention claim for private data.

The successful calls cost $0.00040744 in returned usage, took 51.87 seconds summed, and had median call time 4.48 seconds. Feature extraction, local OCR, networking, and possible review cost must be added in an actual routed system. No crop-versus-page token saving was measured.

Pixel-checked outcomes were mixed: Qwen recovered a missing fourth financial column and some handwritten differential symbols, but changed `$1.0`/`$1.9` into `$10`/`$19` in another financial crop, changed an author's email spelling, and worsened some handwritten prose. All four table-labelled requests returned non-HTML text despite the table-format request; one was an author container rather than a true grid. Successful API/schema responses therefore did not establish safe canonical replacement. These selected development crops have no independent held-out routing estimate.

## Verification and limitations

The final app was restarted at 18:15:27 UTC. A fresh financial-table request completed in 9.71 seconds and its downloaded JSON matched the API result. A separately uploaded private signature-page image retained identical 74-region content and geometry in the paired baseline/candidate runs (7.11/11.00 seconds); those counts are specific to that raster upload, not the earlier PDF session. The candidate demo remains on local forwarded port 18093 and the unchanged baseline on 18092.

The browser was exercised through file selection, Parse, Text output, Visual, Markdown, and source overlays. Controlled table captions/notes and the unresolved-evidence disclosure were visually inspected. Public GSoC feedback was submitted through the UI and read back with both source image and result persisted. Downloaded JSON matched canonical results; Markdown was checked separately. No internal decoder token display was introduced.

All project tests in `tests/` passed: 949 tests, five dependency deprecation warnings, 49.66 seconds. Ruff checks and formatting passed for changed Python files and relevant tests. An unscoped discovery attempt collected unrelated vendored research suites and failed for their missing dependencies; the project test directory was then run explicitly. Existing ignored tests were accidentally overwritten by one agent, reconstructed from historical source snapshots, and rerun without weakening their assertions. Known pre-edit source sections and line positions matched; byte-for-byte identity was not established. New tests remain local and unpublished.

A 180-sample, approximately 1 Hz observation saw the native Falcon process reach 6,418 MiB GPU memory. This is a sampled maximum, not an exact peak or minimum VRAM requirement. The A10G exposes 23,028 MiB. Orientation runs in separate child processes and must be counted separately; resident snapshots showed 438 MiB for the baseline child and 352 MiB for the candidate child while both apps were alive. Host process high-water readings included approximately 4.52 GiB native Falcon, 2.12 GiB baseline app, and 1.64 GiB candidate app. Child processes and unobserved transient allocations prevent treating these as a complete peak budget.

The optional cross-page table merger is inactive and does not yet preserve surrounding-text metadata from later continuation parts; this repair is not a cross-page export claim.

No comprehensive independent annotations were produced for character accuracy, physical-content coverage, order, table cells, or control association. Accordingly there is no CER, layout-IoU, calibration, or unseen clinical benchmark improvement claim. Blank inputs and orientation/decoder failures remain visible in the evaluation denominator. The primary remaining obstacle is learned source ownership and robust recognition, not duplicate bookkeeping. The selected repair removes demonstrated integration loss while retaining the stronger recognition baseline.

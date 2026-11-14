# Neurocomputing (Elsevier) — submission checklist

Journal: **Neurocomputing**, ISSN 0925-2312
Portal: Editorial Manager (link on the journal's *Guide for Authors* page)
Article type: **Regular article** (Type 1)
Reference style: **Elsevier numbered** (`elsarticle-num`)

Sources for every item below are listed in `README.md` under
"Requirements and where they come from".

---

## 1. Files in this folder

| Path | What it is | Upload item type |
|---|---|---|
| `main.tex` | manuscript (elsarticle, `5p`) | Manuscript |
| `sections/*.tex` | body sections, incl. `declarations.tex` | Manuscript (merged) |
| `generated/*.tex` | 12 tables, derived from the canonical tables in `results/tables/`; 8 are printed in the manuscript and 4 in the supplement | Table |
| `figures/*.pdf` | 11 vector figures; 5 in the manuscript and 6 in the supplement | Figure |
| `references.bib` | bibliography source | Manuscript |
| `main.bbl` | compiled bibliography (after build) | Manuscript |
| `main.pdf` | compiled manuscript (proof) | (proof only) |
| `submission/highlights.txt` / `.docx` | 5 highlights, ≤85 chars each | Highlights |
| `submission/abstract.docx` / `.pdf` | standalone abstract (239 words) for the new service's required "Abstract" slot | Abstract |
| `submission/cover_letter.tex` / `.pdf` | cover letter | Cover Letter |
| `supplementary.tex` / `submission/SCORE-Net_supplementary.pdf` | standalone supplement | Supplementary Material |
| `submission/SCORE-Net_submission.zip` | the 33 LaTeX source files above, flattened and pre-ordered for EM (no build artifacts, `main.tex` first) | uploaded as one archive at Attach Files |
| *(declarations tool)* | Declaration of Interest is produced as a `.docx` by <https://declarations.elsevier.com/> (see §2); it is not stored in this repo | Declaration of Interest |
| `submission/data_availability.md` | text of the data availability statement | — |
| `submission/reviewers.md` | guidance for the optional reviewer field | — |
| `results/cv_baselines_<key>.csv` | baseline CV on the shared folds (Table 3) | — |
| `submission/CRediT.md` | author contribution statement | — |

---

## 2. Before you start: decisions to make

- [x] **Affiliation and corresponding e-mail.** Both authors use Department of
      Software Engineering, Faculty of Computer and Information Sciences,
      Sakarya University, Esentepe Campus, 54187 Serdivan, Sakarya, Turkey.
      Corresponding author: Ahmet Alkan, ahmetalkan@sakarya.edu.tr.
- [x] **ORCID iDs.** Both are printed on the title page as footnotes:
      Ahmet Alkan `0009-0006-4919-8080`, Gozde Yolcu Oztel `0000-0002-7841-2131`.
      Enter the same iDs in Editorial Manager for each author record.
- [x] **Repository public.** The URL is in `sections/declarations.tex`
      (`https://github.com/aalkann/SCORE-Net`) and the repository is **public**
      (verified, HTTP 200), so reviewers can open it.
- [x] **Zenodo DOI.** The data availability statement cites the **concept DOI**
      `https://doi.org/10.5281/zenodo.24217263`, which always resolves to the
      newest archived release. The version DOI of `v1.0.0` is
      `10.5281/zenodo.24217264`; `v1.0.1` is the release archived from the
      submission commit.
- [ ] **Competing interests.** The Guide for Authors says the
      [declarations tool](https://declarations.elsevier.com/) "should always be
      completed": answer the yes/no questions, pick "I have nothing to declare"
      if that is accurate, download the resulting `.docx`, and upload it at the
      Attach Files step with item type **Declaration of Interest**. Author
      signatures are not required. The same statement is already printed in
      `sections/declarations.tex`; the tool's sheet is the copy the journal
      expects as a file.
- [ ] **Suggested reviewers.** Optional at Neurocomputing; the field can be left
      empty and the editor will choose. If you fill it, use 3–5 researchers you
      do **not** know personally and who are not from Sakarya University, recent
      co-authors, or collaborators. Safe candidates are authors you already cite
      in the reference list. See `submission/reviewers.md`.
- [ ] **Funding.** If any funding applies, replace the statement in
      `sections/declarations.tex`.

---

## 3. Build the manuscript

```bash
cd paper/neurocomputing
./build.sh          # or: latexmk -C && latexmk -pdf main.tex
```

`build.sh` also regenerates the derived table copies and then verifies that the
build is clean (0 errors, 0 overfull boxes, 0 undefined references).

---

## 4. Formatting requirements — status

| Requirement | Status |
|---|---|
| Elsevier `elsarticle` class | done |
| Numbered reference style (`elsarticle-num`) | done |
| DOIs printed as `doi:` | done (24 entries) |
| Journal names abbreviated (LTWA) | done (17 journal and 4 conference-series names) |
| Abstract ≤ 250 words | done (239) |
| Keywords 1–7, no "and"/"of" | done (6, single terms) |
| Title without unexplained abbreviations | done |
| Full postal affiliation + corresponding e-mail | done (Sakarya University) |
| Tables as editable text, no vertical rules | done (booktabs) |
| All tables/figures cited in order, captions present | done |
| Figures vector/high-resolution | done (PDF) |
| Highlights: 3–5 bullets, ≤85 chars, separate file | done (5, max 83) |
| Figure type size ≥6 pt at printed size | done (all 11 pass `audit_figures.py`) |
| CRediT author contribution statement | done |
| Declaration of competing interest | text done; `.docx` from the declarations tool uploaded at submission |
| Data availability statement | text + repository URL + concept DOI `10.5281/zenodo.24217263` |
| Declaration of generative AI use | done |
| No subfolders in the submission archive | see §5 |

---

## 5. Build the flattened submission archive

Editorial Manager **cannot process subfolders**; every file must sit at the same
level. Create the archive with:

```bash
cd paper/neurocomputing
./make_submission.sh     # writes submission/SCORE-Net_submission.zip
```

The script flattens `sections/`, `generated/`, and `figures/` into one level,
rewrites the `\input`/`\includegraphics` paths, and compiles the flattened copy
to confirm it still builds before zipping.

---

## 6. Upload in the new submission service

Neurocomputing now uses Elsevier's **new submission service**
(`submit.elsevier.com/NEUCOM`), not the classic Editorial Manager upload screen.
The submission is managed in EM only after it has been sent.

The upload step lists three **required** items:

| Required item | What to upload |
|---|---|
| Abstract | `submission/abstract.docx` (fallback: `submission/abstract.pdf`) |
| Manuscript | `submission/main.pdf` — the compiled 20-page proof |
| Declaration of competing interests | tick "I confirm that no authors … have any competing … interests"; the badge turns **Confirmed**. Only upload the declarations-tool `.docx` if the journal asks for a file |

Then set **"Tell us the format of your manuscript" → LaTeX**. The service says
"Upload a .zip and a compiled .pdf", so a second slot appears:

| Item | What to upload |
|---|---|
| LaTeX source files | `submission/SCORE-Net_submission.zip` (33 files, flat, `main.tex` first) |

Optional / supplementary slots (upload if the dropdown offers them):

| File | Item type |
|---|---|
| `submission/cover_letter.pdf` | Cover letter |
| `submission/highlights.docx` | Highlights |
| `submission/SCORE-Net_supplementary.pdf` | Supplementary material |

Do **not** upload the individual figures, tables or `.tex` sections — they are
already inside `main.pdf` and inside the source zip. Do not upload build
artifacts (`main.aux`, `.log`, `.spl`, `.blg`).

The service extracts title, abstract, keywords and authors from `main.pdf`;
check every extracted field on the following screens against the manuscript.
File-name rules enforced by the service: at most 64 characters, no
`% < > : " / \ | ? *`, no duplicate names, no `docm/dotm/exe/bat/pptm/xlsm/ps1`.

Finally, **proofread the generated PDF**: 20 pages, figures present, no `??`
citations, data availability line with `zenodo.24217263`.

---

## 6b. Repository health check

Before packaging, confirm the repository is internally consistent:

```bash
python scripts/check_all.py            # all gates: notebooks, artefacts, numbers, figures
python scripts/check_notebooks.py      # 18 notebooks, all well formed
python scripts/verify_artifacts.py     # 13 invariants: encodings, CSVs, tables, no NaN, test-claim guard, ...
```

`verify_artifacts.py` asserts that the frame loaders used for cross-validation
encode each dataset identically to the splitting loaders used for the single
split, that every cached CSV exists, that the LaTeX tables in `results/tables/`
match `paper/neurocomputing/generated/`, and that no numeric cell is NaN.

## 7. Known items to verify by eye before submitting

- [x] Page 1: affiliation, e-mail, corresponding-author footnote.

- [ ] All main-text floats (8 tables, 5 figures) appear inside the body, **before** the
      reference list, and each one is cited before or on its own page. This
      depends on `dblfloatfix` and `\dbltopnumber=3` in `main.tex`; see
      `README.md`, section "Float placement".
- [ ] Every table fits the text width and is legible at 100 % zoom.
- [ ] Every figure is cited before it appears.
- [ ] Reference list: DOIs present, journal names abbreviated (LTWA).
- [ ] Declarations section appears before the references.
- [ ] No trace of the earlier IEEE layout (`IEEEtran`, `\IEEEPARstart`,
      `IEEEkeywords`, `\markboth`).

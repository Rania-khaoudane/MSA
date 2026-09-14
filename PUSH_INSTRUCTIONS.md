# Pushing this to GitHub

These files update `github.com/Rania-khaoudane/MSA`. The old 80-passage corpus
and 300 pilot items are **preserved** under `data/pilot/` rather than deleted.

## 1. Unzip and enter the folder

```bash
unzip msa-repo-update.zip
cd gh
```

## 2. Clone your existing repo alongside it, then copy these files in

```bash
cd ..
git clone https://github.com/Rania-khaoudane/MSA.git
cp -r gh/* MSA/
cp gh/.gitignore MSA/
cd MSA
```

## 3. Remove the superseded files

The old flat layout is replaced by the `data/` structure.

```bash
git rm -r --cached data/corpus.json data/qa_pairs.json data/qa_pairs_merged.json data/stats.json 2>/dev/null
rm -f data/qa_pairs.json data/qa_pairs_merged.json data/stats.json
rm -f LICENSE          # replaced by LICENSE.md (dual licensing)
rm -f DATASHEET.md     # superseded by docs/EXPERIMENTAL_REPORT.md
```

## 4. Check what will be committed

```bash
git status
git add -A
git status
```

Confirm the list looks right before committing. In particular confirm no
`*_checkpoint.json` or model weights appear — `.gitignore` should exclude them.

## 5. Commit and push

```bash
git commit -m "Add Wikipedia corpus, 500-item benchmark, and full experimental pipeline

- Corpus expanded from 80 to 3,054 passages (2,974 Arabic Wikipedia)
- Benchmark expanded to 500 parallel MSA/Darija QA items
- Ten notebooks: corpus building, validation, retrieval, encoder comparison,
  contrastive fine-tuning, robustness, and end-to-end generation
- Findings summary and full experimental report
- Dual licensing: CC BY-SA 4.0 for Wikipedia passages, Apache 2.0 for original work"

git push origin main
```

## If the push is rejected

```bash
git pull origin main --no-rebase
# resolve any conflicts, keeping the new versions
git add -A && git commit -m "Merge" && git push origin main
```

## After pushing

1. Upload your result CSVs to `results/` — see `results/README.md` for which
   file came from which notebook. Without them the repository has the pipeline
   but no record of the numbers reported in the paper.
2. On the GitHub repo page, add a description and topics
   (`arabic-nlp`, `rag`, `information-retrieval`, `dialectal-arabic`).

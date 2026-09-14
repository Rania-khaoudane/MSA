# Licensing

This repository contains material under two different licences. Which one
applies depends on the file.

---

## 1. Wikipedia-derived passages — CC BY-SA 4.0

**Applies to:** the passages in `data/corpus.json` whose `source` field is
`wikipedia_ar` (2,974 of 3,054 passages).

These are extracted and chunked from Arabic Wikipedia, which is published under
the **Creative Commons Attribution-ShareAlike 4.0 International Licence**
(https://creativecommons.org/licenses/by-sa/4.0/).

Under that licence, if you redistribute these passages or works derived from
them you must:

- **Attribute** Arabic Wikipedia and its contributors. Each passage records its
  originating article in the `article_title` field; the corresponding source is
  `https://ar.wikipedia.org/wiki/<article_title>`.
- **Indicate changes.** The text here has been modified: section headings,
  footnote markers and templates were stripped, and articles were split into
  300–900 character passages and deduplicated.
- **Share alike.** Redistribution must be under CC BY-SA 4.0 or a compatible
  licence.

---

## 2. Everything else — Apache License 2.0

**Applies to:** all original work in this repository, namely

- the MSA/Darija question–answer pairs in `data/qa_pairs_wiki.json`,
  `data/qa_pairs_all.json` and `data/pilot/qa_pairs_pilot.json`
- the 80 author-written passages in `data/pilot/corpus_pilot.json`
  (also present in `data/corpus.json` with `source: pilot_synthetic`)
- all notebooks in `notebooks/`
- all documentation in `docs/`
- results in `results/`

Copyright 2026 Rania Khaoudane

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
these files except in compliance with the License. You may obtain a copy of the
License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

---

## Note on `data/corpus.json`

This file mixes both licences: passages tagged `wikipedia_ar` are CC BY-SA 4.0,
passages tagged `pilot_synthetic` are Apache 2.0. The `source` field on every
passage identifies which applies. Anyone redistributing the file as a whole
should treat it as CC BY-SA 4.0, since that is the more restrictive of the two.

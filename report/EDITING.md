# Editing the report

`IDO_AUM_Lite_Report.docx` is the primary editable copy. It uses Word paragraphs,
heading styles, actual tables and native editable OMML equations. The four
measured figures are embedded PNGs; their reusable source is `src/plots.py`.

The cover follows the supplied Group 2 reference: the six names are a group
roster, not an allocation of implementation or writing work. No individual
contribution claims have been added. Add a contribution statement only after
roles and subsequent changes are confirmed. Codex assistance is described at
project level.

Open the DOCX in Word to adjust the cover, headings, spacing, page limits,
citations or content for the final submission rules. There is no editing
protection. Save a new version before making major changes.

`report.md` is a parallel editable Markdown source. Changes made manually in Word
are not automatically reflected in Markdown, or vice versa.

`build_report.py` regenerates both files from the verified `results/` export and
its content definitions. Run it with an authoring Python runtime containing
`python-docx`; this version was built with the Codex bundled document runtime,
without changing the project's scientific `.venv` dependencies. Regeneration
replaces manual edits, so preserve the edited copy first. Numerical tables come
from the saved metrics and should be checked whenever text is changed.

The seven-page document was rendered through Microsoft Word on Windows and
inspected page by page. Temporary PDF/page images are kept only under ignored
`runs/report_work/`; they are not the editable deliverable. Final page counts can
change with fonts, Word settings or later submission formatting.

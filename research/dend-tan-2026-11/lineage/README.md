# Compact historical archive

This directory contains sanitized, allowlisted root-level records from all indexed local experiments. New explanatory text is English; historical protocols/reports preserve their original language and are clearly labeled.

## Inclusion and omission

The exporter uses explicit filename allowlists for protocols, reports, result summaries, independent audits, stopped/failure records, source manifests and relevant first-party research Python modules. It never traverses experiment subdirectories. Array archives, trained weights, private environments, images, PDFs, logs, copied external repositories, vendor files and `original_tan.py` are excluded. The two principal confirmation packages are linked to the separate evidence directory rather than duplicated.

Large RESULTS records are explicitly projected to primary contrasts, inference definitions and selected compact summaries; each projection lists every omitted top-level field. The export makes no claim to preserve all raw results. Original manifest/index records are redacted provenance evidence, not manifests of this public tree.

## Redaction and provenance

Personal absolute local paths, personal machine hostnames and private external checkout locations are replaced by neutral markers. Path/hash changes are recorded per file. Links to copied files are rewritten; links and embeds for omitted assets become plain filenames. This can remove local reproduction commands. Historical source hashes remain hashes of the original source bytes; the public hashes describe exported bytes. Sanitization does not alter stored numerical result values except string metadata, and no scientific statistics are recomputed.

The included Python files are for inspection and provenance. They have been syntax-parsed only. Omitted modules, source roots and raw arrays mean the compact archive is not a standalone runnable reproduction. Recreating experiments requires the documented dependencies, inputs and an explicitly prepared execution environment. No historical experiment code was imported or run during export. Only this archive curation utility performed copying, sanitization and static checks.

## Verification

The exporter checks coverage of all input study directories, JSON readability, Python syntax, local Markdown link targets, known private-path/host markers, obvious credential-value patterns and the total size limit. This is a bounded export review, not a mathematical guarantee that arbitrary text can contain no sensitive material. Consult [ARCHIVE_INDEX.json](ARCHIVE_INDEX.json) for source/public digests and transformations, [EXPORT_CHECKS.json](EXPORT_CHECKS.json) for the final inspection, and [the English study map](../LINEAGE.md).

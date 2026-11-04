# Completed TAN research snapshot — 2026-11-03

This publication preserves all 680 files from the screened scientific archive, byte for byte. It adds evidence alongside the existing repository; it does not replace the original source or represent a new scientific rerun.

- [Completed scientific audit v11](audit-v11/README.md) — 440 files.
- [Native eight-stage research](native-eight-stage/README.md) — 110 files.
- [Exploratory studies](exploratory/) — 100 files.
- [Local clean training v1](local-clean-training-v1/README.md) — 30 files.
- [Publication and reproduction limitations](audit-v11/PUBLICATION_NOTES.md).
- [Complete file inventory and checksums](SNAPSHOT_MANIFEST.json).

Earlier and exploratory results retain their historical status. Computational checks are not independent human peer review. The screened copy omits operational/private material and preserves the documented public-copy transformations; it is not an unmodified backup of the entire original machine.

## Complete original ZIP

The exact screened ZIP is stored in three ordered parts under [archive](archive/) so every Git object remains below 100 MiB. From that directory, reconstruct and verify it with:

```sh
sha256sum -c SHA256SUMS
cat TAN-completed-research-2026-11-03.zip.part01 TAN-completed-research-2026-11-03.zip.part02 TAN-completed-research-2026-11-03.zip.part03 > TAN-completed-research-2026-11-03.zip
printf '%s  %s\n' ca8a2ae51b4e8f18ab341fa483ca9970a7a3077c5d4e4013e5df13a9811b6167 TAN-completed-research-2026-11-03.zip | sha256sum -c -
```

Expected ZIP size: 189,164,458 bytes. It contains 680 files totaling 383,445,961 uncompressed bytes. The extracted directories above contain exactly those same bytes. The manifest, this guide, Git attributes and archive parts are publication packaging added outside the 680-file count.

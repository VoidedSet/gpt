#!/bin/bash
set -e

echo "[*] Downloading Complete Works of Shakespeare (Project Gutenberg)..."
curl -fSL https://www.gutenberg.org/cache/epub/100/pg100.txt -o dataset/complete_shakespeare.txt

# Remove Project Gutenberg header/footer metadata if present, or keep it simple
# The raw file is around 5.5MB. Let's truncate it or use it as is.
# Let's check size
SIZE=$(wc -c < "dataset/complete_shakespeare.txt")
echo "[+] Downloaded $(numfmt --to=iec-bytes $SIZE) of text."

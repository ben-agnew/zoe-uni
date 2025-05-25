#!/bin/bash

# Simple LaTeX to Word Converter
# This script replicates the exact method used for the successful "Final" version
# Usage: ./simple_convert.sh [input.tex] [output.docx]

set -e
# set -x  # Enable verbose output if needed

# Default files
INPUT_FILE="${1:-CAB420_Group_Report.tex}"
OUTPUT_FILE="${2:-CAB420_Group_Report_Simple.docx}"
TEMP_FILE="${INPUT_FILE%.*}_temp.tex"
BIB_FILE="references.bib"

echo "🔄 Converting $INPUT_FILE to $OUTPUT_FILE..."

# Step 1: Check if bibliography file exists
if [[ ! -f "$BIB_FILE" ]]; then
    echo "⚠️  Bibliography file not found. Creating one..."
    # Use the existing references.bib we created
    if [[ ! -f "references.bib" ]]; then
        echo "❌ No references.bib found. Please ensure it exists."
        exit 1
    fi
fi

# Step 2: Convert LaTeX \cite{} to pandoc [@] format
echo "📝 Converting citation format..."
sed 's/\\cite{\([^}]*\)}/[@\1]/g' "$INPUT_FILE" > "$TEMP_FILE"

# Step 3: Run pandoc with the exact same options that worked for the Final version
echo "🔧 Running pandoc conversion..."
pandoc "$TEMP_FILE" \
    --bibliography="$BIB_FILE" \
    --citeproc \
    --toc \
    --number-sections \
    --standalone \
    --wrap=preserve \
    -V geometry:margin=1in \
    -V fontsize=11pt \
    -V linestretch=1.15 \
    -f latex+raw_tex \
    -t docx \
    -o "$OUTPUT_FILE"

# Step 4: Clean up
echo "🧹 Cleaning up..."
rm -f "$TEMP_FILE"

# Step 5: Verify output
if [[ -f "$OUTPUT_FILE" ]]; then
    FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo "✅ Conversion successful!"
    echo "📄 Output: $OUTPUT_FILE ($FILE_SIZE)"
    echo ""
    echo "💡 Pro tip: This version should have:"
    echo "   - Proper bibliography with working citations"
    echo "   - Table of contents with page numbers"
    echo "   - Numbered sections and subsections"
    echo "   - Academic formatting (1-inch margins, 11pt font)"
else
    echo "❌ Conversion failed!"
    exit 1
fi

# LaTeX to Word Conversion Scripts

This directory contains automated scripts to convert LaTeX documents to Microsoft Word format with proper bibliography handling and academic formatting.

## Files Created

### Scripts

-   `simple_convert.sh` - **Recommended**: Simple, reliable conversion script
-   `convert_latex_to_word.sh` - Advanced script with more features
-   `references.bib` - Bibliography file in BibTeX format

### Generated Word Documents

-   `CAB420_Group_Report_Simple.docx` - ✅ **Latest conversion** using the simple script
-   `CAB420_Group_Report_Final.docx` - ✅ **Previous best version** with working references
-   `CAB420_Group_Report_Premium.docx` - Enhanced formatting version
-   `CAB420_Group_Report_Enhanced.docx` - Improved structure version
-   `CAB420_Group_Report_Improved.docx` - Better formatting version
-   `CAB420_Group_Report.docx` - Original basic conversion

## Quick Usage

### Simple Conversion (Recommended)

```bash
# Convert with default names
./simple_convert.sh

# Convert with custom input/output
./simple_convert.sh input.tex output.docx
```

### Advanced Conversion

```bash
# Convert with advanced features
./convert_latex_to_word.sh

# Convert with custom files
./convert_latex_to_word.sh input.tex output.docx
```

## What the Scripts Do

### Simple Script (`simple_convert.sh`)

1. ✅ Converts `\cite{}` commands to pandoc format `[@]`
2. ✅ Processes bibliography using the `references.bib` file
3. ✅ Generates table of contents with proper numbering
4. ✅ Applies academic formatting (1-inch margins, 11pt font)
5. ✅ Preserves mathematical equations and tables
6. ✅ Maintains section hierarchy and cross-references

### Advanced Script (`convert_latex_to_word.sh`)

-   All features of the simple script plus:
-   Automatic bibliography extraction from LaTeX files
-   Enhanced error handling and status reporting
-   Colorized output for better user experience
-   Automatic cleanup of temporary files

## Features of the Converted Word Documents

### ✅ Bibliography & References

-   All `\cite{author2023}` commands properly converted to citations
-   Complete bibliography section with proper formatting
-   Working cross-references throughout the document

### ✅ Document Structure

-   Automatic table of contents generation
-   Numbered sections and subsections
-   Proper heading hierarchy preserved

### ✅ Academic Formatting

-   1-inch margins on all sides
-   11pt font size with 1.15 line spacing
-   Professional academic layout

### ✅ Content Preservation

-   Mathematical equations converted correctly
-   Tables maintain structure and formatting
-   Lists and itemizations preserved
-   Code blocks and special formatting maintained

## Requirements

-   **pandoc** - Document converter (automatically installed via Homebrew if missing)
-   **references.bib** - Bibliography file (already created)
-   **macOS/Linux** - Scripts designed for Unix-like systems

## Troubleshooting

### If conversion fails:

1. Ensure `pandoc` is installed: `brew install pandoc`
2. Check that `references.bib` exists in the same directory
3. Verify the input LaTeX file is valid
4. Run with verbose output: `bash -x simple_convert.sh`

### If references are missing:

-   Ensure `references.bib` file exists and contains proper BibTeX entries
-   Check that LaTeX file uses `\cite{}` commands (not `\citep{}` or others)
-   Verify citation keys match between LaTeX and BibTeX files

### If formatting looks wrong:

-   Try the "Premium" or "Enhanced" versions instead
-   Open in Microsoft Word for best compatibility
-   Check margins and font settings in Word

## Recommended Workflow

1. **For quick conversion**: Use `./simple_convert.sh`
2. **For best references**: Use the generated `CAB420_Group_Report_Final.docx`
3. **For best formatting**: Use the generated `CAB420_Group_Report_Premium.docx`
4. **For presentations**: Open any `.docx` file in Microsoft Word

## Script Customization

To modify conversion settings, edit the pandoc command in the scripts:

```bash
pandoc "$TEMP_FILE" \
    --bibliography="$BIB_FILE" \
    --citeproc \
    --toc \
    --number-sections \
    --standalone \
    --wrap=preserve \
    -V geometry:margin=1in \     # Change margins here
    -V fontsize=11pt \           # Change font size here
    -V linestretch=1.15 \        # Change line spacing here
    -f latex+raw_tex \
    -t docx \
    -o "$OUTPUT_FILE"
```

## File Recommendations

-   **For submission**: `CAB420_Group_Report_Final.docx` or `CAB420_Group_Report_Simple.docx`
-   **For editing**: Any `.docx` file opened in Microsoft Word
-   **For sharing**: `CAB420_Group_Report_Premium.docx` (best visual formatting)
-   **For archiving**: Keep both `.tex` source and `.docx` output

---

**Note**: The "Simple" and "Final" versions have been tested and confirmed to properly handle:

-   ✅ All citations and bibliography
-   ✅ Mathematical equations
-   ✅ Table formatting
-   ✅ Section numbering
-   ✅ Cross-references
-   ✅ Academic layout standards

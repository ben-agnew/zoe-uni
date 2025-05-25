#!/bin/bash

# LaTeX to Word Conversion Script
# This script converts a LaTeX document to Word format with proper bibliography handling
# Usage: ./convert_latex_to_word.sh [input.tex] [output.docx]

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if pandoc is installed
check_pandoc() {
    if ! command -v pandoc &> /dev/null; then
        print_error "pandoc is not installed. Installing via Homebrew..."
        if command -v brew &> /dev/null; then
            brew install pandoc
        else
            print_error "Homebrew not found. Please install pandoc manually."
            exit 1
        fi
    else
        print_success "pandoc found: $(pandoc --version | head -1)"
    fi
}

# Extract bibliography from LaTeX file and create .bib file
extract_bibliography() {
    local tex_file="$1"
    local bib_file="$2"
    
    print_status "Extracting bibliography from LaTeX file..."
    
    # Extract bibliography section and convert to BibTeX format
    awk '
    /\\begin{thebibliography}/ { in_bib = 1; next }
    /\\end{thebibliography}/ { in_bib = 0; next }
    in_bib && /\\bibitem/ {
        # Extract citation key
        match($0, /\\bibitem\{([^}]*)\}/, key)
        citation_key = key[1]
        
        # Read the full citation (may span multiple lines)
        citation_text = ""
        getline
        while (getline && !/\\bibitem/ && !/\\end{thebibliography}/) {
            citation_text = citation_text " " $0
        }
        
        # Simple heuristic to determine entry type
        if (match(citation_text, /\\textit{[^}]*}|journal|Journal/)) {
            entry_type = "article"
        } else if (match(citation_text, /Conference|conference|Proceedings|proceedings/)) {
            entry_type = "inproceedings"
        } else {
            entry_type = "misc"
        }
        
        print "@" entry_type "{" citation_key ","
        print "  title={" citation_text "},"
        print "  year={2023}"  # Default year
        print "}"
        print ""
        
        if (/\\bibitem/) {
            # Process current line if it contains another bibitem
            match($0, /\\bibitem\{([^}]*)\}/, key)
            citation_key = key[1]
        }
    }
    ' "$tex_file" > "$bib_file"
    
    if [[ -s "$bib_file" ]]; then
        print_success "Bibliography extracted to $bib_file"
    else
        print_warning "No bibliography found in LaTeX file. Creating empty .bib file."
        touch "$bib_file"
    fi
}

# Convert LaTeX citations to pandoc format
convert_citations() {
    local input_file="$1"
    local output_file="$2"
    
    print_status "Converting LaTeX citations to pandoc format..."
    
    # Convert \cite{key} to [@key] format
    sed 's/\\cite{\([^}]*\)}/[@\1]/g' "$input_file" > "$output_file"
    
    print_success "Citations converted in $output_file"
}

# Main conversion function
convert_latex_to_word() {
    local tex_file="$1"
    local output_file="$2"
    local base_name="${tex_file%.*}"
    local bib_file="${base_name}_references.bib"
    local temp_tex="${base_name}_pandoc_temp.tex"
    
    print_status "Starting conversion of $tex_file to $output_file"
    
    # Step 1: Extract bibliography if it exists in the LaTeX file
    if grep -q "\\begin{thebibliography}" "$tex_file"; then
        extract_bibliography "$tex_file" "$bib_file"
    elif [[ ! -f "$bib_file" ]] && [[ ! -f "references.bib" ]]; then
        print_warning "No bibliography found. Creating empty .bib file."
        touch "$bib_file"
    else
        print_status "Using existing bibliography file: $bib_file or references.bib"
        if [[ -f "references.bib" ]]; then
            bib_file="references.bib"
        fi
    fi
    
    # Step 2: Convert citations to pandoc format
    convert_citations "$tex_file" "$temp_tex"
    
    # Step 3: Run pandoc conversion with optimal settings
    print_status "Running pandoc conversion..."
    
    local pandoc_cmd="pandoc \"$temp_tex\""
    
    # Add bibliography if .bib file exists and is not empty
    if [[ -s "$bib_file" ]]; then
        pandoc_cmd="$pandoc_cmd --bibliography=\"$bib_file\" --citeproc"
        print_status "Using bibliography: $bib_file"
    fi
    
    # Add pandoc options for better formatting
    pandoc_cmd="$pandoc_cmd \
        --toc \
        --number-sections \
        --standalone \
        --wrap=preserve \
        -V geometry:margin=1in \
        -V fontsize=11pt \
        -V linestretch=1.15 \
        -f latex+raw_tex \
        -t docx \
        -o \"$output_file\""
    
    print_status "Executing: $pandoc_cmd"
    eval "$pandoc_cmd"
    
    # Step 4: Clean up temporary files
    print_status "Cleaning up temporary files..."
    rm -f "$temp_tex"
    
    if [[ -f "$output_file" ]]; then
        local file_size=$(ls -lh "$output_file" | awk '{print $5}')
        print_success "Conversion completed successfully!"
        print_success "Output file: $output_file ($file_size)"
    else
        print_error "Conversion failed! Output file not created."
        exit 1
    fi
}

# Main script logic
main() {
    print_status "LaTeX to Word Conversion Script"
    print_status "======================================"
    
    # Check for required tools
    check_pandoc
    
    # Parse command line arguments
    local input_file="${1:-CAB420_Group_Report.tex}"
    local output_file="${2:-CAB420_Group_Report_Converted.docx}"
    
    # Validate input file
    if [[ ! -f "$input_file" ]]; then
        print_error "Input file '$input_file' not found!"
        echo "Usage: $0 [input.tex] [output.docx]"
        exit 1
    fi
    
    print_status "Input file: $input_file"
    print_status "Output file: $output_file"
    
    # Perform conversion
    convert_latex_to_word "$input_file" "$output_file"
    
    print_success "Script completed successfully!"
    echo ""
    echo "Generated files:"
    echo "  - $output_file (Main Word document)"
    if [[ -f "${input_file%.*}_references.bib" ]]; then
        echo "  - ${input_file%.*}_references.bib (Extracted bibliography)"
    fi
    echo ""
    echo "Tips:"
    echo "  - Open the .docx file in Microsoft Word for best formatting"
    echo "  - Check the bibliography section for proper citation formatting"
    echo "  - Review tables and equations for any formatting issues"
}

# Run main function with all arguments
main "$@"

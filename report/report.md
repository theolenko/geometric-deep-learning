# Report — Build Instructions

## Dependencies

| Tool | Version |
|------|---------|
| TeX Live | 2026 (Arch Linux) |
| pdfTeX | 3.141592653-2.6-1.40.29 |
| BibTeX | 0.99e |
| texlive-fontsrecommended | required for Times font |

Install on Arch Linux:
```bash
sudo pacman -S texlive texlive-fontsrecommended
```

## File Structure

```
report/
├── main.tex                  # main document
├── neurips_2026.sty          # NeurIPS 2026 style file
├── references.bib            # bibliography
├── sections/
│   ├── 01_introduction.tex   # Lara
│   ├── 02_data.tex           # Theodor
│   └── 03_methodology.tex    # Marla
└── report.md                 # this file
```

## Compile

**First time or after adding new references:**
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**After normal text changes:**
```bash
pdflatex main.tex
```

Output: `main.pdf`

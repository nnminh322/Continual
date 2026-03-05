#!/usr/bin/env python3
"""
Enhanced PDF extraction for paper introductions
Uses pdfplumber with proper import handling
"""

import os
import sys
import re
from pathlib import Path

# Force fresh import of pdfplumber
import importlib
import subprocess

def ensure_pdfplumber():
    """Ensure pdfplumber is available."""
    try:
        import pdfplumber
        return pdfplumber
    except ImportError:
        print("Installing pdfplumber...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
        import pdfplumber
        return pdfplumber

def extract_section_from_pdf(pdf_path, section_name="introduction", max_chars=1500):
    """Extract a specific section from PDF using pdfplumber."""
    try:
        pdfplumber = ensure_pdfplumber()
        
        with pdfplumber.open(pdf_path) as pdf:
            # Search through first 10 pages
            text = ""
            for i in range(min(10, len(pdf.pages))):
                page_text = pdf.pages[i].extract_text() or ""
                text += page_text + "\n"
            
            # Search for section header
            pattern = rf'(?:^|\n)(?:{section_name}|1\.|1\.1[^0-9])[^\n]*\n'
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            if match:
                start = match.end()
                # Get next section boundary
                next_section = re.search(
                    r'\n(?:related|background|method|approach|2\.|2\.1|discussion|conclusion)',
                    text[start:],
                    re.IGNORECASE
                )
                if next_section:
                    end = start + next_section.start()
                else:
                    end = start + max_chars
                
                return text[start:end].strip()
            else:
                # Fallback: return first 1500 chars after skipping title/abstract
                abstract_end = text.find("1. Introduction")
                if abstract_end == -1:
                    abstract_end = 0
                return text[abstract_end:abstract_end + max_chars].strip()
    
    except Exception as e:
        return f"[Error: {str(e)[:100]}]"

def extract_arxiv_id(url):
    """Extract arxiv ID from URL."""
    match = re.search(r'arxiv\.org/(abs/)?(\d+\.\d+)', url)
    if match:
        return match.group(2)
    return None

def generate_enhanced_summary(papers_dir, links_file, output_file):
    """Generate summary document with PDF extracts and code info."""
    
    # Read paper links
    papers = {}
    with open(links_file, 'r') as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) >= 6:
                pid = parts[0].strip()
                papers[pid] = {
                    'year': parts[1].strip(),
                    'venue': parts[2].strip(),
                    'title': parts[4].strip(),
                    'url': parts[5].strip()
                }
    
    print(f"Generating enhanced summaries for {len(papers)} papers...")
    
    # Generate markdown with extracts
    with open(output_file, 'a') as f:
        f.write("\n\n---\n\n# ENHANCED ANALYSIS WITH PDF INTRODUCTIONS\n\n")
        
        for pid in sorted(papers.keys(), key=lambda x: int(x))[:10]:  # First 10 for demo
            meta = papers[pid]
            pdf_files = list(Path(papers_dir).glob(f"{int(pid):02d}_*.pdf"))
            
            if not pdf_files:
                continue
            
            pdf_path = str(pdf_files[0])
            arxiv_id = extract_arxiv_id(meta['url'])
            
            print(f"Paper {pid}: Extracting...", end=" ")
            intro = extract_section_from_pdf(pdf_path)
            print(f"✓ ({len(intro)} chars)")
            
            f.write(f"## Paper {int(pid):02d}: {meta['title']}\n\n")
            f.write(f"**Venue:** {meta['venue']} | **Year:** {meta['year']}\n")
            f.write(f"**arxiv:** {arxiv_id or 'N/A'}\n\n")
            f.write(f"### Introduction (from PDF)\n")
            f.write(f"```\n{intro}\n```\n\n")

if __name__ == "__main__":
    papers_dir = "/Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/papers"
    links_file = "/Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/paper_links.txt"
    eval_file = "/Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/summaries/all_papers_evaluation.md"
    extract_file = "/Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/summaries/pdf_introductions_sample.md"
    
    generate_enhanced_summary(papers_dir, links_file, extract_file)
    print(f"\n✅ Sample PDF extractions saved to {extract_file}")

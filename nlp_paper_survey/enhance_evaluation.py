#!/usr/bin/env python3
"""
Enhanced evaluation script:
1. Extract introduction from each PDF (first 2-3 pages after abstract)
2. Check GitHub/Code availability from arxiv pages
3. Classify domain (NLP vs CV)
4. Update evaluation with additional context
"""

import os
import json
import re
from pathlib import Path
from urllib.parse import urlparse
import urllib.request
import urllib.error

# Try to import PDF processing libraries
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    print("⚠️ pdfplumber not installed. Will attempt pypdf or fallback.")

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# Try requests for web scraping
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("⚠️ requests not installed. Will use urllib only.")


class PDFProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text = None
    
    def extract_text(self, pages=None):
        """Extract text from PDF. pages=None means all pages."""
        if HAS_PDFPLUMBER:
            return self._extract_pdfplumber(pages)
        elif HAS_PYPDF:
            return self._extract_pypdf(pages)
        else:
            print(f"❌ No PDF library available for {self.pdf_path}")
            return ""
    
    def _extract_pdfplumber(self, pages=None):
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                if pages is None:
                    pages = list(range(len(pdf.pages)))
                text = ""
                for i in pages:
                    if i < len(pdf.pages):
                        text += pdf.pages[i].extract_text() or ""
                return text
        except Exception as e:
            print(f"❌ pdfplumber error for {self.pdf_path}: {e}")
            return ""
    
    def _extract_pypdf(self, pages=None):
        try:
            reader = PdfReader(self.pdf_path)
            if pages is None:
                pages = list(range(len(reader.pages)))
            text = ""
            for i in pages:
                if i < len(reader.pages):
                    text += reader.pages[i].extract_text() or ""
            return text
        except Exception as e:
            print(f"❌ pypdf error for {self.pdf_path}: {e}")
            return ""
    
    def extract_introduction(self):
        """Extract introduction section (typically pages 1-3 after abstract)."""
        # Try first 5 pages to capture introduction
        text = self.extract_text(pages=list(range(min(5, 10))))
        
        # Find "introduction" section
        intro_match = re.search(
            r'(?:introduction|1\s+introduction|1\.1\s+|introduction\s*\n)',
            text,
            re.IGNORECASE
        )
        
        if intro_match:
            intro_start = intro_match.start()
            # Get ~1000 chars after "Introduction" header
            intro_text = text[intro_start:intro_start+2000]
            # Try to find next section header to cut off
            next_section = re.search(
                r'\n(?:related|background|method|approach|2\.|2\.1)',
                intro_text,
                re.IGNORECASE
            )
            if next_section:
                intro_text = intro_text[:next_section.start()]
            return intro_text.strip()
        else:
            # Fallback: return first 1500 chars
            return text[:1500].strip()


class CodeChecker:
    """Check if paper has public code (GitHub, GitLab, etc.)"""
    
    DOMAIN_KEYWORDS_CV = {
        'image', 'vision', 'visual', 'object detection', 'segmentation',
        'classification', 'video', 'scene', 'spatial', 'rendering', 'point cloud',
        'depth', 'reconstruction', 'pose', 'tracking', 'optical flow',
        'disparity', 'vit', 'cnn', 'convolution', 'inception', 'resnet',
        'efficientnet', 'yolo', 'mask r-cnn', 'faster r-cnn'
    }
    
    DOMAIN_KEYWORDS_NLP = {
        'nlp', 'language', 'text', 'bert', 'gpt', 'transformer', 'token',
        'semtext', 'semantic', 'parsing', 'sentiment', 'translation',
        'question answering', 'qa', 'nlu', 'nlg', 'summarization',
        'dialogue', 'conversation', 'linguistic', 'embedding', 'llm',
        'prompt', 'instruction', 'generation'
    }
    
    @staticmethod
    def get_arxiv_abstract_page(arxiv_id):
        """Get HTML of arxiv abstract page."""
        url = f"https://arxiv.org/abs/{arxiv_id}"
        try:
            if HAS_REQUESTS:
                resp = requests.get(url, timeout=5)
                return resp.text
            else:
                with urllib.request.urlopen(url, timeout=5) as response:
                    return response.read().decode('utf-8')
        except Exception as e:
            return None
    
    @staticmethod
    def find_github_in_html(html_content):
        """Find GitHub/code repository links in HTML."""
        if not html_content:
            return None
        
        # Search for common code hosting patterns
        patterns = [
            r'https?://github\.com/[\w\-]+/[\w\-]+',
            r'https?://gitlab\.com/[\w\-]+/[\w\-]+',
            r'https?://huggingface\.co/[\w\-]+/[\w\-]+',
            r'https?://([a-z0-9\-]+)?\.github\.io/[\w\-]+',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html_content)
            if match:
                return match.group(0)
        
        return None
    
    @staticmethod
    def classify_domain(title, abstract, intro=""):
        """Classify paper as NLP, CV, or ML."""
        combined_text = f"{title} {abstract} {intro}".lower()
        
        cv_score = sum(1 for kw in CodeChecker.DOMAIN_KEYWORDS_CV if kw in combined_text)
        nlp_score = sum(1 for kw in CodeChecker.DOMAIN_KEYWORDS_NLP if kw in combined_text)
        
        if cv_score > nlp_score:
            return "CV"
        elif nlp_score > cv_score:
            return "NLP"
        else:
            return "ML/Multi"  # General ML if ambiguous


def process_all_papers(papers_dir, links_file):
    """Process all papers: extract intro, check code, classify domain."""
    
    # Read paper links
    papers_data = {}
    with open(links_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('|')
            if len(parts) >= 6:
                paper_id = parts[0].strip()
                year = parts[1].strip()
                venue = parts[2].strip()
                title = parts[4].strip()
                url = parts[5].strip()
                papers_data[paper_id] = {
                    'year': year,
                    'venue': venue,
                    'title': title,
                    'url': url,
                    'introduction': None,
                    'code_link': None,
                    'domain': 'Unknown'
                }
    
    print(f"📝 Processing {len(papers_data)} papers...")
    
    for paper_id, meta in papers_data.items():
        pdf_path = os.path.join(papers_dir, f"{paper_id.zfill(2)}_*.pdf")
        # Find actual PDF file
        matching_files = list(Path(papers_dir).glob(f"{paper_id.zfill(2)}_*.pdf"))
        
        if not matching_files:
            print(f"⏭️  Paper {paper_id}: PDF not found")
            continue
        
        pdf_path = str(matching_files[0])
        paper_code = paper_id.zfill(2)
        
        # 1. Extract introduction
        print(f"📄 Paper {paper_code}: Extracting intro...", end=" ")
        processor = PDFProcessor(pdf_path)
        intro = processor.extract_introduction()
        meta['introduction'] = intro[:500] + "..." if len(intro) > 500 else intro
        print(f"✓ ({len(intro)} chars)")
        
        # 2. Check for code
        print(f"  Checking code...", end=" ")
        url = meta['url']
        
        # Extract arxiv ID if present
        arxiv_match = re.search(r'arxiv[.org]*/(?:abs/)?(\d+\.\d+)', url)
        if arxiv_match:
            arxiv_id = arxiv_match.group(1)
            html = CodeChecker.get_arxiv_abstract_page(arxiv_id)
            code_link = CodeChecker.find_github_in_html(html)
            if code_link:
                meta['code_link'] = code_link
                print(f"✓ Found: {code_link}")
            else:
                print("✗ No code link")
        else:
            print("⏭️  Not arxiv")
        
        # 3. Classify domain
        print(f"  Classifying domain...", end=" ")
        domain = CodeChecker.classify_domain(
            meta['title'],
            meta['introduction'],
            meta.get('introduction', '')
        )
        meta['domain'] = domain
        print(f"✓ {domain}")
    
    return papers_data


def update_evaluation_file(output_file, papers_data):
    """Update all_papers_evaluation.md with new information."""
    
    print(f"\n📊 Generating enhanced evaluation file...")
    
    with open(output_file, 'a') as f:
        f.write("\n\n## ENHANCED EVALUATION DATA\n\n")
        f.write("### Domain Classification\n")
        f.write("| Paper ID | Domain | Title |\n")
        f.write("|----------|--------|-------|\n")
        
        for paper_id in sorted(papers_data.keys(), key=lambda x: int(x)):
            meta = papers_data[paper_id]
            f.write(f"| {paper_id.zfill(2)} | {meta['domain']:<10} | {meta['title'][:60]}... |\n")
        
        f.write("\n### Code Availability Checklist\n")
        f.write("| Paper ID | Public Code | Link |\n")
        f.write("|----------|-------------|------|\n")
        
        for paper_id in sorted(papers_data.keys(), key=lambda x: int(x)):
            meta = papers_data[paper_id]
            code_status = "✅" if meta['code_link'] else "❌"
            code_link = meta['code_link'] or "N/A"
            if len(code_link) > 50:
                code_link = code_link[:47] + "..."
            f.write(f"| {paper_id.zfill(2)} | {code_status} | {code_link} |\n")
    
    print(f"✅ Enhanced evaluation saved to {output_file}")
    
    # Return summary
    cv_count = sum(1 for m in papers_data.values() if m['domain'] == 'CV')
    nlp_count = sum(1 for m in papers_data.values() if m['domain'] == 'NLP')
    ml_count = sum(1 for m in papers_data.values() if m['domain'] == 'ML/Multi')
    code_count = sum(1 for m in papers_data.values() if m['code_link'])
    
    return {
        'cv': cv_count,
        'nlp': nlp_count,
        'ml': ml_count,
        'code': code_count,
        'total': len(papers_data)
    }


if __name__ == "__main__":
    papers_dir = "/Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/papers"
    links_file = "/Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/paper_links.txt"
    eval_file = "/Users/nnminh322/Desktop/personal/Continual/nlp_paper_survey/summaries/all_papers_evaluation.md"
    
    # Check libraries
    print("📦 Checking dependencies...")
    if HAS_PDFPLUMBER:
        print("✅ pdfplumber available")
    elif HAS_PYPDF:
        print("✅ pypdf available")
    else:
        print("⚠️  No PDF library - will try installation")
    
    if HAS_REQUESTS:
        print("✅ requests available")
    else:
        print("⚠️  requests not available - using urllib")
    
    # Process papers
    papers_data = process_all_papers(papers_dir, links_file)
    
    # Update evaluation
    stats = update_evaluation_file(eval_file, papers_data)
    
    print(f"\n📊 Summary:")
    print(f"   CV papers: {stats['cv']}")
    print(f"   NLP papers: {stats['nlp']}")
    print(f"   ML/Multi papers: {stats['ml']}")
    print(f"   Papers with public code: {stats['code']}/{stats['total']}")

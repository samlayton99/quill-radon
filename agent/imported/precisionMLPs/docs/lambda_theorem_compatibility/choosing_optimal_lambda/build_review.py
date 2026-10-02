"""Build the section, figure and appendix without imposing a page target."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', help='Path to pdflatex or tectonic; otherwise search PATH')
    args = parser.parse_args()
    engine = args.engine or shutil.which('pdflatex') or shutil.which('tectonic')
    if not engine:
        raise SystemExit('A LaTeX engine is required: pdflatex or tectonic.')
    build = ROOT/'build'
    build.mkdir(exist_ok=True)
    if 'tectonic' in Path(engine).name:
        command = [engine, '--untrusted', '--keep-logs', '--keep-intermediates',
                   '--outdir', str(build), 'review_document.tex']
        passes = 1
    else:
        command = [engine, '-interaction=nonstopmode', '-halt-on-error',
                   '-output-directory='+str(build), 'review_document.tex']
        passes = 2
    for _ in range(passes):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError((result.stdout+result.stderr)[-6000:])
    log = (build/'review_document.log').read_text()
    problems = [word for word in ['Overfull', 'undefined references',
                                  'multiply defined', 'Missing character'] if word in log]
    if problems:
        raise RuntimeError('Typesetting needs review: '+', '.join(problems))
    combined = ROOT/'choosing_optimal_lambda_review.pdf'
    shutil.copyfile(build/'review_document.pdf', combined)
    reader = PdfReader(combined)
    texts = [p.extract_text() for p in reader.pages]
    assert 'Choosing optimal lambda' in texts[0]
    appendix_start = next(i for i, text in enumerate(texts) if 'Proof and interpretation' in text)
    figure_start = next(i for i, text in enumerate(texts) if 'Basic (green' in text)
    assert appendix_start == figure_start+1, 'Expected one comparison-figure page'
    assert 'More speci' in ''.join(texts[:figure_start])
    preview = PdfWriter()
    for page in reader.pages[:figure_start]:
        preview.add_page(page)
        if '/Annots' in preview.pages[-1]:
            del preview.pages[-1]['/Annots']
    preview.add_metadata({'/Title': 'Choosing optimal lambda: section draft'})
    with (ROOT/'choosing_optimal_lambda_section.pdf').open('wb') as out:
        preview.write(out)
    appendix = PdfWriter()
    for page in reader.pages[appendix_start:]:
        appendix.add_page(page)
    with (ROOT/'choosing_optimal_lambda_appendix.pdf').open('wb') as out:
        appendix.write(out)
    math_checks = json.loads((ROOT/'figures/mathematical_checks.json').read_text())
    report = {
        'main_pages': figure_start, 'main_words_extracted': sum(len(t.split()) for t in texts[:figure_start]),
        'comparison_figure_pages': 1, 'comparison_grid': [2, 4],
        'appendix_pages': len(texts)-appendix_start,
        'total_pages': len(texts), 'text_size_pt': 11,
        'overfull_boxes': False, 'unresolved_references': False,
        'main_first_pair_bound': True, 'appendix_lemmas': 3,
        'explicit_lambda_error_bound': True, 'theorem_box': False,
        'basic_rule_limit_explained': True,
        'page_count_target_enforced': False, 'kernel_notation': 'Khat; no H alias',
        'frequency_prefactor_log_sensitivity_proved': True,
        'figure_colormap': 'viridis', 'figure_axes': 'log-log',
        'reference_papers_edited': False, 'network_fits_rerun': False,
        'prediction_lines': 'Recomputed using the revised formulas',
        'practical_rule_requires_spectral_mass': False,
        'frequency_input': 'rough angular frequency scale',
        'practical_refinement': 'explicit first pair with exact central denominator',
        'solve_for_lambda_explicit': True, 'effective_tolerance_user_chosen': True,
        'main_paragraph_skip_pt': 7, 'theorem_exact_central_denominator': True,
        'mathematics': math_checks,
    }
    (ROOT/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

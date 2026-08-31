use Cwd qw(abs_path);

# Keep this root-level handoff buildable without copying the canonical paper.
my $canonical_source = abs_path('../papers/route_certificates');
die "Cannot find canonical route-certificate manuscript\n"
    unless defined $canonical_source;

$ENV{'TEXINPUTS'} = $canonical_source . ':' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BIBINPUTS'} = $canonical_source . ':' . ($ENV{'BIBINPUTS'} // '');

$pdf_mode = 1;
$out_dir = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';

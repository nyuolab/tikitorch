# Official, unmodified venue styles are installed into ignored build/style.
$ENV{'TEXINPUTS'} = 'build/style//:build/texmf/tex//:' . ($ENV{'TEXINPUTS'} // '');
$ENV{'BSTINPUTS'} = 'build/style//:' . ($ENV{'BSTINPUTS'} // '');
$pdf_mode = 1;
$out_dir = 'build';
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';

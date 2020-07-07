LATEX       = pdflatex
CHECK_RERUN = grep "Rerun to get" $*.log

all: cont-estimator.pdf

%.pdf: %.tex
	${LATEX} $*
	# bibtex $*
	# ${LATEX} $*
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $* ) || echo "Done."

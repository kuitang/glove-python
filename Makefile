all:
	python setup.py build_ext
	python setup.py install

get-wiki:
	wget http://dumps.wikimedia.org/enwiki/20151201/enwiki-20151201-pages-meta-current1.xml-p000000010p000010000.bz2

process-wiki:
	python -- examples/example.py -w -c enwiki-20151201-pages-meta-current1.xml-p000000010p000010000.bz2

train-wiki:
	python -i -- examples/example.py -t 30 -p 2

all-wiki: get-wiki process-wiki train-wiki

.PHONY: all get-wiki process-wiki train-wiki

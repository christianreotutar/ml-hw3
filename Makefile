FILES=README
ZIPNAME = hw3b-fconsta1-creotut1.zip
ZIP=zip

default: noop

noop: ;

zip:
	rm $(ZIPNAME)
	$(ZIP) $(ZIPNAME) $(FILES)

run: 
	./collapsed-sampler input-train.txt input-test.txt output.txt 5 0.5 0.1 0.01 1100 1000

dumb:
	./collapsed-sampler input-train.txt input-test.txt output.txt 10 0.5 0.1 0.01 1100 1000



test: 
	python -m cProfile driver.py input-train.txt input-test.txt output.txt 10 0.5 0.1 0.01 300 200

pypy:
	pypy driver.py input-train.txt input-test.txt output.txt 10 0.5 0.1 0.01 300 200


topwords:
	python topwords.py output.txt-phi

clean:
	rm *.pyc

small:
	python driver.py small_train.txt small_test.txt output.txt 10 0.5 0.1 0.01 300 200

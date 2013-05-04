compile:
	mkdir -p classes
	javac -cp fig.jar:util.jar:Jama-1.0.2.jar:iText-2.1.0.jar -d classes `find java_src -name "*.java"`
	jar cf UnsupLexCCA.jar -C classes .
	rm -r classes

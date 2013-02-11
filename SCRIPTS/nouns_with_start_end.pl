use strict;

while (my $line = <>) {
	chomp $line;
	$line = "_${line}_";
	$line =~ tr/-/_/;
	print $line. "\n";
}

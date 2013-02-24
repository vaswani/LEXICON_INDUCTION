use strict;

while (<>) {
	s/^/_/;
	s/$/_/;
	s/,/_,_/g;
	s/-/_/g;
	print lc($_);
}

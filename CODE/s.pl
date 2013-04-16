use strict;

my $S = <>;
my @A = split(/\) , \(/, $S);
$" = "\n";
for (my $i=0; $i < scalar @A; $i++) {
  $A[$i] = lc($A[$i]);
}
print "@A";

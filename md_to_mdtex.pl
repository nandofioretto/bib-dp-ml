#!/usr/bin/perl 

use strict;
use warnings;

#$^I = '.bak'; # create a backup copy 

my ($in, $out) = @ARGV;
if (not defined $in  or not defined $out) {
  die "Syntax error: ./script.pl <input> <output>\n";
}
print("$in $out\n");

my $ldel = '<img align="center" src="https://latex.codecogs.com/gif.latex?';
my $rdel = '"/>';
my $lldel = '
<center>
<img align="center" src="https://latex.codecogs.com/gif.latex?';
my $rrdel = '"/>
</center>
';

open(FILE, "$in") || die "File not found";
my @lines = <FILE>;
close(FILE);

my @newlines;
foreach(@lines) {
	$_ =~ s/\\\(/$ldel/g; # do the replacement
	$_ =~ s/\\\)/$rdel/g; # do the replacement
	$_ =~ s/\\\[/$lldel/g; # do the replacement
	$_ =~ s/\\\]/$rrdel/g; # do the replacement
	push(@newlines,$_);
}

open(FILE, ">$out") || die "File not found";
print FILE @newlines;
close(FILE);
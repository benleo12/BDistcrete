#!/usr/bin/env python3
"""Extend EERAD3's seed index from 0..99 to 0..699 in blocks of 100.

Stock readinit clamps inum to 0..99 (anything else silently becomes 0: same seeds, same
output filename E00.*), and names output files 'E' + two digits. Here inum = 100*iblk + jnum:
seeds are the table entry for jnum offset by 3000*iblk (RANMAR needs ij < 31329 and
kl < 30082; the table maximum is 9980, so seven blocks fit), and the output prefix letter is
'E' + iblk. The VEGAS grid filename is built from 'E'//fname(4:13) with a literal 'E', so all
blocks keep reading the same grid files."""
import re, sys
p = sys.argv[1]
s = open(p).read()

def sub1(pat, rep, flags=re.M):
    global s
    n = len(re.findall(pat, s, flags))
    assert n == 1, f'{n} matches for {pat!r}'
    s = re.sub(pat, rep, s, flags=flags)

sub1(r'^      if \(inum\.lt\.0\.or\.inum\.gt\.99\) inum = 0[ \t]*$',
     '      if (inum.lt.0.or.inum.gt.699) inum = 0\n'
     'c --- seed blocks of 100: jnum indexes the table, iblk offsets it\n'
     '      iblk = inum/100\n'
     '      jnum = mod(inum,100)')
sub1(r'^      i1 = iseeds\(1,inum\)[ \t]*$', '      i1 = iseeds(1,jnum) + 3000*iblk')
sub1(r'^      i2 = iseeds\(2,inum\)[ \t]*$', '      i2 = iseeds(2,jnum) + 3000*iblk')
sub1(r'^      if \(inum\.lt\.10\) then[ \t]*$', '      if (jnum.lt.10) then')
sub1(r"^         write\(fname\(3:3\),'\(I1\)'\) inum[ \t]*$",
     "         write(fname(3:3),'(I1)') jnum")
sub1(r"^         write\(fname\(2:3\),'\(I2\)'\) inum[ \t]*$",
     "         write(fname(2:3),'(I2)') jnum\n"
     "      endif\n"
     "      fname(1:1) = char(ichar('E')+iblk)\n"
     "      write(*,*) '* seed block',iblk,' seeds',i1,i2,' ',fname(1:3)\n"
     "      if (.false.) then")
open(p, 'w').write(s)
for ln in s.splitlines():
    if len(ln) > 72 and not ln.lstrip().startswith(('c', 'C', '*', '!')):
        raise SystemExit(f'line >72 cols: {ln!r}')
print('patched', p)

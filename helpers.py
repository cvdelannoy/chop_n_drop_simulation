import shutil
import os, re
from glob import glob
from datetime import datetime
import pathlib


def parse_output_dir(out_dir, clean=False):
    out_dir = os.path.abspath(out_dir) + '/'
    if clean:
        shutil.rmtree(out_dir, ignore_errors=True)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_input_dir(in_dir, pattern=None):
    if type(in_dir) != list: in_dir = [in_dir]
    out_list = []
    for ind in in_dir:
        if not os.path.exists(ind):
            raise ValueError(f'{ind} does not exist')
        if os.path.isdir(ind):
            ind = os.path.abspath(ind)
            if pattern is None: out_list.extend(glob(f'{ind}/**/*', recursive=True))
            else: out_list.extend(glob(f'{ind}/**/{pattern}', recursive=True))
        else:
            if pattern is None: out_list.append(ind)
            elif pattern.strip('*') in ind: out_list.append(ind)
    return out_list

def print_timestamp():
    return datetime.now().strftime('%y-%m-%d_%H:%M:%S')



def create_digest_re(enzyme="", digest_rules=None):
    """ Create regular expression for digestion rules
    iputs (note that digest_rules takes precedence over enzyme):
    enzyme - Digesting enzyme.  From enzyme option help:
      <no-enzyme|trypsin|trypsin/p|chymotrypsin|elastase|clostripain|cyanogen-
      bromide|iodosobenzoate|proline-endopeptidase|staph-protease|asp-n|lys-c|
      lys-n|arg-c|glu-c|pepsin-a|elastase-trypsin-chymotrypsin|
      custom-enzyme> - Specify the enzyme used to digest the proteins in silico.
      Available enzymes (with the corresponding digestion rules indicated in
      parentheses) include no-enzyme ([X]|[X]), trypsin ([RK]|{P}), trypsin/p
      ([RK]|[]), chymotrypsin ([FWYL]|{P}), elastase ([ALIV]|{P}),
      clostripain ([R]|[]), cyanogen-bromide ([M]|[]), iodosobenzoate
      ([W]|[]), proline-endopeptidase ([P]|[]), staph-protease ([E]|[]),
      asp-n ([]|[D]), lys-c ([K]|{P}), lys-n ([]|[K]), arg-c ([R]|{P}),
      glu-c ([DE]|{P}), pepsin-a ([FL]|{P}), elastase-trypsin-chymotrypsin
      ([ALIVKRWFY]|{P}). Specifying --enzyme no-enzyme yields a non-enzymatic digest.
      Warning: the resulting peptide database may be quite large. Default = trypsin.
    digest_rules - Rules detailing digest.  From custom-enzyme option help:
      <string> - Specify rules for in silico digestion of protein sequences.
      Overrides the enzyme option. Two lists of residues are given enclosed
      in square brackets or curly braces and separated by a |. The first
      list contains residues required/prohibited before the cleavage site
      and the second list is residues after the cleavage site. If the
      residues are required for digestion, they are in square brackets,
      '[' and ']'. If the residues prevent digestion, then they are
      enclosed in curly braces, '{' and '}'. Use X to indicate all residues.
      For example, trypsin cuts after R or K but not before P which is
      represented as [RK]|{P}. AspN cuts after any residue but only before
      D which is represented as [X]|[D]. Default = <empty>.
    outpus:
    digest_re - python regular expression to digest proteins
    Source: Javier Alfaro (via Natalie Worp)
    """
    enzyme = enzyme.lower()

    if not digest_rules:
        if enzyme == 'no-enzyme':
            digest_rules = '[X]|[X]'
        elif enzyme == 'trypsin':
            digest_rules = '[RK]|{P}'
        elif enzyme == 'trypsin/p':
            digest_rules = '[RK]|[]'
        elif enzyme == 'chymotrypsin':
            digest_rules = '[FWYL]|{P}'
        elif enzyme == 'elastase':
            digest_rules = '[ALIV]|{P}'
        elif enzyme == 'clostripain':
            digest_rules = '[R]|[]'
        elif enzyme == 'cyanogen-bromide':
            digest_rules = '[M]|[]'
        elif enzyme == 'iodosobenzoate':
            digest_rules = '[W]|[]'
        elif enzyme == 'proline-endopeptidase':
            digest_rules = '[P]|[]'
        elif enzyme == 'staph-protease':
            digest_rules = '[E]|[]'
        elif enzyme == 'asp-n':
            digest_rules = '[X]|[D]'
        elif enzyme == 'lys-c':
            digest_rules = '[K]|{P}'
        elif enzyme == 'lys-n':
            digest_rules = '[]|[K]'
        elif enzyme == 'arg-c':
            digest_rules = '[R]|{P}'
        elif enzyme == 'glu-c':
            digest_rules = '[DE]|{P}'
        elif enzyme == 'pepsin-a':
            digest_rules = '[FL]|{P}'
        elif enzyme == 'elastase-trypsin-chymotrypsin':
            digest_rules = '[ALIVKRWFY]|{P}'
        else:
            raise ValueError(f'Invalid digestion enzyme name: {enzyme}')

    # parse digestion rule
    h = digest_rules.split('|')
    if len(h) != 2:
        raise ValueError("Invalid custom enzyme rules")

    nterm = h[0]
    if nterm[0] == '[':
        if nterm[-1] != ']':
            print("Error reading digesting enzyme, missing closing bracket ]")
            exit(3)

        include_nterm = True
        if nterm[1:-1]:
            nterm_aa = nterm[1:-1]
        else:
            nterm_aa = ''
    elif nterm[0] == '{':
        if nterm[-1] != '}':
            print("Error reading digesting enzyme, missing closing bracket ]")
            exit(4)

        include_nterm = False
        if nterm[1:-1]:
            nterm_aa = nterm[1:-1]
        else:
            nterm_aa = ''

    cterm = h[1]
    if cterm[0] == '[':
        if cterm[-1] != ']':
            print("Error reading digesting enzyme, missing closing bracket ]")
            exit(5)

        include_cterm = True
        if cterm[1:-1]:
            cterm_aa = cterm[1:-1]
        else:
            cterm_aa = ''
    elif cterm[0] == '{':
        if cterm[-1] != '}':
            print("Error reading digesting enzyme, missing closing bracket ]")
            exit(6)

        include_cterm = False
        if cterm[1:-1]:
            cterm_aa = cterm[1:-1]
        else:
            cterm_aa = ''

    if nterm_aa == 'X':
        nterm_aa = ''
    if cterm_aa == 'X':
        cterm_aa = ''
    # form digestion regular expression
    # tryptic digest with lysine, arginine secondary digestion suppression ".(?:(?<![KR](?!P|K|R)).)*"
    if include_nterm:
        if not include_cterm:
            digest_re = ".(?:(?<!"
            if nterm_aa:
                digest_re += "[" + nterm_aa + "]"
            if cterm_aa:
                digest_re += "(?!"
                for aa in cterm_aa[:-1]:
                    digest_re += aa
                    digest_re += "|"
                digest_re += cterm_aa[-1]
                digest_re += ")"
            digest_re += ").)*"
        else:
            digest_re = ".(?:(?<!"
            if nterm_aa:
                digest_re += "[" + nterm_aa + "]"
            if cterm_aa:
                digest_re += "["
                digest_re += cterm_aa
                digest_re += "]"
            digest_re += ").)*"
    else:
        if not include_cterm:
            digest_re = ".(?:(?<!"
            if nterm_aa:
                digest_re += "[^" + nterm_aa + "]"
            if cterm_aa:
                digest_re += "(?!"
                for aa in cterm_aa[:-1]:
                    digest_re += aa
                    digest_re += "|"
                digest_re += cterm_aa[-1]
                digest_re += ")"
            digest_re += ").)*"
        else:
            digest_re = ".(?:(?<!"
            if nterm_aa:
                digest_re += "[^" + nterm_aa + "]"
            if cterm_aa:
                digest_re += "["
                digest_re += cterm_aa
                digest_re += "]"
            digest_re += ").)*"

    return re.compile(digest_re)
# enzyme cleavage rules from '.(?:(?<![R][RKGV]).)*'
# http://www.sbcs.qmul.ac.uk/iubmb/enzyme/EC3/4/23/49.html


def digest(sequence):
 peptides = re.compile('.(?:(?<![KR](?=[RKGV])).)*').findall(sequence)
 return peptides

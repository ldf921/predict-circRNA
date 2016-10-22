import numpy as np
import pickle as pkl
from timeit import default_timer
from operator import attrgetter, itemgetter
from itertools import groupby, filterfalse
import math
import bisect
from tqdm import tqdm

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement[base] for base in seq]
    return complseq

def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))
    
def get_seq_for_circularRNA(circbase, whole_seq):
    fp = open(circbase, 'r')
    for line in fp:
        values = line.split()
        chr_n = values[0]
        start = int(values[1])
        end = int(values[2])
        circurna_name = values[3]
        strand = values[5]
        seq = whole_seq[chr_n]
        extract_seq = seq[start: end]
        extract_seq = extract_seq.upper()
        if strand == '-':
            extract_seq =  reverse_complement(extract_seq)
    fp.close()

def get_hg19_whole_seq():
    from Bio import SeqIO
    with open('hg19.fa') as f:
        whole_seq = SeqIO.to_dict(SeqIO.parse(f, 'fasta'))
        for k in whole_seq:
            whole_seq[k] = whole_seq[k].seq
    return whole_seq

def feature_base(l):
    index = {'A' : 0, 'G' : 1, 'C' : 2, 'T' : 3}
    seq_tag = [index[c] for c in l.seq]
    feature = np.zeros([l.length, 4], dtype=np.float32)
    feature[np.arange(0, l.length), seq_tag] = 1
    return feature

class Loci:
    ''' A pair of positions in DNA sequence'''
    @property
    def length(self):
        return self.end - self.start

    @property
    def feature_dim(self):
        if not hasattr(self, 'features'):
            self.get_feature()
        return self.features.shape[1]

    @property
    def chrom_strand(self):
        return self.chr + self.strand


    whole_seq = None

    def init_seq(self):
        if Loci.whole_seq is None:
            Loci.whole_seq = get_hg19_whole_seq()

        seq = self.whole_seq[self.chr]
        extract_seq = seq[self.start : self.end]
        extract_seq = extract_seq.upper()
        if extract_seq.find('N') != -1:
            self.seq = None
        else:
            if self.strand == '-':
                extract_seq =  reverse_complement(extract_seq)
            self.seq = extract_seq

    def get_feature(self):
        extract_func = [feature_base]
        features = [ func(self) for func in extract_func]
        self.features = np.concatenate(features, axis = 1)

def read_sequence(file_name, label):    
    with open(file_name, 'r') as f:
        locis = []
        for line in f.readlines():
            tokens = line.split()
            if len(tokens) > 3:
                l = Loci()
                l.chr = tokens[0] 
                l.start = int(tokens[1])
                l.end = int(tokens[2])
                l.strand = tokens[5 if len(tokens) > 6 else 3]
                l.label = label
                locis.append(l)
        return locis

class ChromPositions:
    @staticmethod
    def group_locis(locis):
        return groupby(sorted(map(lambda l : (l.chrom_strand, l), locis), key=itemgetter(0)), key=itemgetter(0))

    def __init__(self, locis):
        self.chrom_positions = dict()

        for chrom, locis in self.group_locis(locis):
            assert chrom not in self.chrom_positions
            start_positions = [] 
            end_positions = [] 
            start, end = 0, -math.inf
            for l in sorted(map(itemgetter(1), locis), key=attrgetter('start')):
                if end < l.start:
                    if end > 0:
                        start_positions.append(start)
                        end_positions.append(end)
                    start, end = l.start, l.end
                else:
                    end = max(end, l.end)
            if end > 0:
                start_positions.append(start)
                end_positions.append(end)

            self.chrom_positions[chrom] = (start_positions, end_positions)

    def get_intersection(self, loci):
        start_positions, end_positions = self.chrom_positions.get(loci.chrom_strand, ([], []))
        lp = bisect.bisect_right(end_positions, loci.start)
        rp = bisect.bisect_left(start_positions, loci.end)
        return list(zip(start_positions[lp:rp], end_positions[lp:rp]))

    def is_intersect(self, loci):
        start_positions, end_positions = self.chrom_positions.get(loci.chrom_strand, ([], []))
        lp = bisect.bisect_right(end_positions, loci.start)
        rp = bisect.bisect_left(start_positions, loci.end)
        return lp < rp


def filter_negative(locis_pos, locis_neg):
    positions = ChromPositions(locis_pos)
    return filterfalse(lambda l : positions.is_intersect(l), locis_neg)

def read_data():
    T0 = default_timer()
    try:
        with open('filt_locis.bin', 'rb') as f:
            locis = pkl.load(f)
    except IOError:
        locis = read_sequence('hsa_hg19_Rybak2015.bed', 1)
        locis_neg = read_sequence('all_exons.bed', 0)
        lp = len(locis)
        locis.extend(filter_negative(locis, locis_neg))
        ln = len(locis) - lp
        print(lp, ln)

        for l in tqdm(locis):
            l.init_seq()

        with open('filt_locis.bin', 'wb') as f:
            pkl.dump(locis, f)
    print('read data used %.3fs' % (default_timer() - T0))
    return locis


def process_feature():
    '''Deprecated'''
    T0 = default_timer()
    
    locis = read_data()
    for l in locis:
        l.get_feature()

    print('process_feature used %.3fs' % (default_timer() - T0))

    return locis

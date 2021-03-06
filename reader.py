import numpy as np
import pickle as pkl
from timeit import default_timer
from operator import attrgetter, itemgetter
from itertools import groupby, filterfalse
import math
import bisect
from tqdm import tqdm
from sklearn.externals import joblib
import os
import re
import random
from hmmlearn.hmm import MultinomialHMM

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    complseq = [complement[base] for base in seq]
    return complseq

def reverse_complement(seq):
    # seq = list(seq)
    # seq.reverse()
    # return ''.join(complement(seq))
    return seq.reverse_complement()
    
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
    # index = {'A' : 0, 'G' : 1, 'C' : 2, 'T' : 3}
    # seq_tag = [index[c] for c in l.seq]
    feature = np.zeros([l.length, 4], dtype=np.float32)
    feature[np.arange(0,l.length,dtype=np.int32), l.seq] = 1
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

    def extend(self, window = 100):
        self.start = max(self.start - window, 0)
        self.end = self.end + window

    def decode_seq(self):
        if self.seq is not None:
            self.end = self.start + len(self.seq)
            seq = np.array(list(self.seq))
            self.seq = np.array((seq == 'G') + (seq == 'C') * 2 + (seq == 'T') * 3)

    extract_func = [feature_base]
    def get_feature(self):
        # print(self.seq)
        self.decode_seq()
        features = [ func(self) for func in self.extract_func]
        self.features = np.concatenate(features, axis = 1)

def get_token(regex, tokens):
    for s in tokens:
        if re.match(regex, s):
            return s
    assert False, 'Fail to match %s' % regex

def read_sequence(file_name, label, extract_exon = False):    
    with open(file_name, 'r') as f:
        locis = []
        for line in f.readlines():
            tokens = line.split()
            if len(tokens) > 3:
                l = Loci()
                l.chr = tokens[0] 
                l.start = int(tokens[1])
                l.end = int(tokens[2])
                l.strand = get_token(r'[\+-]', tokens)
                l.label = label
                if len(tokens) >= 12 and extract_exon:
                    for size, relative_start in zip(map(int, tokens[10].split(',')), map(int ,tokens[11].split(','))):
                        e = Loci()
                        e.chr = l.chr
                        e.start = l.start + relative_start
                        e.end = e.start + size
                        e.strand = l.strand
                        e.label = label
                        locis.append(e)
                else:
                    locis.append(l)
        return locis

class AluFeatureExtrator:
    def __init__(self, model_file = None, components = None):

        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
        else:
            alu_file = 'Alu_sequence.pkl'
            if os.path.exists(alu_file):
                locis = joblib.load(alu_file)
            else:
                locis = read_sequence('hg19_Alu.bed', 0)
                locis = random.sample(locis, 100000)
                for l in tqdm(locis):
                    l.init_seq()
                    l.decode_seq()
                locis = list(filter(lambda l : l.seq is not None, locis))
                joblib.dump(locis, alu_file)

            print('Alu Loaded')
            locis = locis[0:5000]
            model = MultinomialHMM(n_components=components, verbose=True, n_iter=50)
            x = np.concatenate(list(map(attrgetter('seq'), locis)))
            x = np.reshape(x, [x.shape[0], 1])
            length = list(map(attrgetter('length'), locis))
            model.fit(x, length)
            self.model = model
            joblib.dump(self.model, model_file)

    def __call__(self, l):
        prob = np.dot(self.model.predict_proba(np.reshape(l, [-1, 1])),
            self.model.emissionprob_)
        return prob[np.arange(0,l.length,dtype=np.int32), l.seq].reshape(-1, 1)

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

class AluIntersectionExtrator:
    def __init__(self):
        model_file = 'Alu_position.pkl'
        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
        else:
            locis = read_sequence('hg19_Alu.bed', 0)
            self.model = ChromPositions(locis)
            joblib.dump(self.model, model_file)

    def __call__(self, l):
        return None

def filter_negative(locis_pos, locis_neg):
    positions = ChromPositions(locis_pos)
    return filterfalse(lambda l : positions.is_intersect(l), locis_neg)

def read_data():
    T0 = default_timer()
    # Loci.extract_func.append(AluFeatureExtrator(components=20, model_file="hmm_Alu_big.pkl"))
    # Loci.extract_func.append(AluFeatureExtrator(components=20, model_file="hmm_Alu_big.pkl"))
    input_file = 'filt_locis.bin'
    # input_file = 'extend_locis.bin'
    try:
        with open(input_file, 'rb') as f:
            locis = pkl.load(f)
    except IOError:
        locis = read_sequence('hsa_hg19_Rybak2015.bed', 1, True)
        locis_neg = read_sequence('all_exons.bed', 0)
        lp = len(locis)
        locis.extend(filter_negative(locis, locis_neg))
        ln = len(locis) - lp
        print("Number of postives {}, Number of negatives {}".format(lp, ln))

        for l in tqdm(locis):
            l.init_seq()

        with open(input_file, 'wb') as f:
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

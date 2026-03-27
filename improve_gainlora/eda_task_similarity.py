#!/usr/bin/env python3
"""EDA: Compute task similarity metrics for Long Sequence Order 3 benchmark."""

import json, os, sys
import numpy as np

base = 'CL_Benchmark/Long_Sequence'
tasks = ['yelp','amazon','mnli','cb','copa','qqp','rte','imdb','sst2',
         'dbpedia','agnews','yahoo','multirc','boolq','wic']

# Load task texts
task_texts = {}
for t in tasks:
    train = json.load(open(os.path.join(base, t, 'train.json')))
    texts = [inst['input'] for inst in train['Instances'][:500]]
    task_texts[t] = ' '.join(texts)

# TF-IDF cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf = vectorizer.fit_transform([task_texts[t] for t in tasks])
sim_matrix = cosine_similarity(tfidf)

print("=" * 70)
print("TF-IDF Cosine Similarity — Top 15 most similar task pairs:")
print("=" * 70)
pairs = []
for i in range(len(tasks)):
    for j in range(i+1, len(tasks)):
        pairs.append((sim_matrix[i,j], tasks[i], tasks[j]))
pairs.sort(reverse=True)
for s, a, b in pairs[:15]:
    print(f'  {a:10s} - {b:10s}: {s:.4f}')

print("\n" + "=" * 70)
print("Average label length (proxy for task complexity):")
print("=" * 70)
for t in tasks:
    train = json.load(open(os.path.join(base, t, 'train.json')))
    labels = [inst['output'] for inst in train['Instances'][:500]]
    avg_len = np.mean([len(l) for l in labels])
    unique = len(set(labels))
    print(f'  {t:10s} | avg_label_len={avg_len:6.1f} | unique_labels={unique}')

# Intra-group similarity
print("\n" + "=" * 70)
print("Intra-cluster similarity (same-domain):")
print("=" * 70)
groups = {
    'sentiment': ['yelp','amazon','imdb','sst2'],
    'NLI': ['mnli','cb','rte'],
    'news': ['dbpedia','agnews','yahoo'],
    'RC': ['multirc','boolq'],
}
for name, group in groups.items():
    idxs = [tasks.index(t) for t in group]
    intra = np.mean([sim_matrix[i,j] for i in idxs for j in idxs if i != j])
    print(f'  {name:10s}: avg intra-sim = {intra:.4f}')

# Full similarity matrix (compact)
print("\n" + "=" * 70)
print("Full Similarity Matrix (abbreviated task names):")
print("=" * 70)
short = ['ylp','amz','mnl','cb','cop','qqp','rte','imd','ss2','dbp','agn','yah','mrc','blq','wic']
header = '     ' + ' '.join([f'{s:>5}' for s in short])
print(header)
for i, t in enumerate(short):
    row = f'{t:>5}'
    for j in range(len(short)):
        if j <= i:
            row += f' {sim_matrix[i,j]:5.2f}'
        else:
            row += '     '
    print(row)

# Distribution of input lengths
print("\n" + "=" * 70)
print("Input length distribution (tokens, approximate):")
print("=" * 70)
for t in tasks:
    train = json.load(open(os.path.join(base, t, 'train.json')))
    inputs = [inst['input'] for inst in train['Instances'][:500]]
    lens = [len(inp.split()) for inp in inputs]
    print(f'  {t:10s} | mean={np.mean(lens):6.1f} | median={np.median(lens):6.1f} | '
          f'p25={np.percentile(lens,25):5.0f} | p75={np.percentile(lens,75):5.0f} | max={max(lens)}')

import sys
import re
import string
import math
import matplotlib
matplotlib.use('Agg') # For SVG in the terminal
import matplotlib.pyplot as plt
import numpy as np
from typing import TextIO, Generator
from natasha import Segmenter, NewsEmbedding, NewsMorphTagger, MorphVocab, Doc
import nltk
from nltk.corpus import cmudict

# NLTK Initialize
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)

d_en = cmudict.dict()

# Natasha Initialize
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
tagger = NewsMorphTagger(emb)

STOP_POS = {'PREP', 'CONJ', 'PRCL', 'INTJ', 'PUNCT', 'ADP', 'CCONJ', 'SCONJ', 'DET'}
TENSION_POS = {'VERB', 'ADV', 'AUX', 'ADJ'}

# Stat Initialize
word_count = 0
word_stats = {}
all_files_data = {}
all_rhythm_lines = [] # For line output

# Rhythm Analyze
def count_syllables_ru(word: str) -> int:
    vowels = "аеёиоуыэюя"
    return len([char for char in word.lower() if char in vowels])

def count_syllables_en(word: str) -> int:
    if not word:
        return 0

    word = word.lower()
    if word in d_en:
        return max([len([y for y in x if y[-1].isdigit()]) for x in d_en[word]])
    
    count = 0

    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
        
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1

    if word.endswith("e"):
        count -= 1

    return max(1, count)

def export_vss_resonance(data_map):
    if len(data_map) < 2:
        return

    plt.figure(figsize=(16, 8), facecolor="#2D2D2D")
    ax = plt.gca(); ax.set_facecolor('#2D2D2D')
    colors = ['#329b47', '#b07b08', "#051D96", "#4c0854"]
    
    for i, (path, words_data) in enumerate(data_map.items()):
        syllables = [d['syllables'] for d in words_data]
        label = path.split('/')[-1]
        plt.plot(np.arange(len(syllables)), syllables, color=colors[i % len(colors)], label=f"VSS: {label}", alpha=0.6, linewidth=2)
        plt.fill_between(np.arange(len(syllables)), syllables, color=colors[i % len(colors)], alpha=0.1)

    plt.title("VSS RHYTHM RESONANCE SCAN", color='white', fontsize=14)
    plt.legend(facecolor='#2D2D2D', labelcolor='white')
    plt.axis('off')
    plt.savefig("vss_resonance.svg", format='svg', bbox_inches='tight', facecolor='#2D2D2D')
    plt.close()

    print(">>> VSS Resonance exported to: vss_resonance.svg")

def export_vss_cardio(words_data, path: str):
    if not words_data:
        return
    
    syllables = [d['syllables'] for d in words_data]
    tension = [d['is_tension'] for d in words_data]
    words = [d['word'] for d in words_data]
    
    plt.figure(figsize=(16, 6), facecolor='#2D2D2D')
    ax = plt.gca(); ax.set_facecolor('#2D2D2D')
    x = np.arange(len(syllables))
    plt.plot(x, syllables, color="#329b47", alpha=0.3, linestyle='-', linewidth=1)
    
    for i in range(len(x)):
        color = "#b07b08" if tension[i] else '#329b47'
        plt.vlines(x[i], 0, syllables[i], colors=color, linewidth=2, alpha=0.8)
        plt.scatter(x[i], syllables[i], color=color, s=25, zorder=3)
        plt.text(x[i], -0.3, words[i], rotation=45, color='white', fontsize=7, ha='right')

    clean_path = path.replace('/', '_').replace('\\', '_')
    output_name = f"{clean_path}_vss.svg"

    plt.axis('off')
    plt.savefig(output_name, format='svg', bbox_inches='tight', facecolor='#2D2D2D')
    plt.close()

    print(f">>> VSS Cardio exported to: {output_name}")

def process_nlp_multilang(text: str):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(tagger)
    
    words_data, meaningful_lemmas = [], []
    for token in doc.tokens:
        is_ru = bool(re.search('[а-яА-Я]', token.text))
        if is_ru:
            token.lemmatize(morph_vocab)
            lemma, syls = token.lemma, count_syllables_ru(token.text)
        else:
            lemma, syls = token.text.lower(), count_syllables_en(token.text)
        if token.pos != 'PUNCT' and syls > 0:
            words_data.append({'word': token.text, 'syllables': syls, 'is_tension': token.pos in TENSION_POS})
        if token.pos not in STOP_POS and len(token.text) > 1:
            meaningful_lemmas.append(lemma)
            
    tension = sum(1 for d in words_data if d['is_tension']) / len(words_data) if words_data else 0
    return meaningful_lemmas, tension, words_data

def analyze(path: str):
    global word_count, word_stats, all_rhythm_lines
    file_tension_accum = []
    file_all_words_data = []
    try:
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                clean_line = line.strip()
                if not clean_line: continue
                
                lemmas, t_score, w_data = process_nlp_multilang(clean_line)
                
                # Stat
                for lemma in lemmas:
                    word_count += 1
                    word_stats[lemma] = word_stats.get(lemma, 0) + 1
                
                if w_data:
                    file_all_words_data.extend(w_data)
                    file_tension_accum.append(t_score)
                    # Save rhytm of the line
                    all_rhythm_lines.append([d['syllables'] for d in w_data])
        
        if file_all_words_data:
            all_files_data[path] = file_all_words_data
            export_vss_cardio(file_all_words_data, path)
            
        return sum(file_tension_accum)/len(file_tension_accum) if file_tension_accum else 0
    except FileNotFoundError: 
        print(f"File not found: {path}", file=sys.stderr)
    except Exception as e: 
        print(f"Unable to read file: {path}: {e}")
    return 0

def main(args: list[str]):
    if not args:
        print("no input files", file=sys.stderr); return
    tension_scores = [analyze(p) for p in args]
    if word_count == 0: return
    
    export_vss_resonance(all_files_data)
    
    stats = dict(sorted(word_stats.items(), key=lambda item: item))
    entropy, prob_sum = 0, 0
    for token, count in stats.items():
        prob = count / word_count
        prob_sum += prob
        entropy -= prob * math.log2(prob)
        print(f"{token}: {count} ({prob})")
        
    print(f"\nRhytm Analyze")
    for i, scheme in enumerate(all_rhythm_lines[:15]): 
        print(f"Line {i+1}: {'-'.join(map(str, scheme))} \n// Total Syllables: {sum(scheme)}")
            
    unique_count = len(stats)
    ttr = unique_count / word_count
    ent_max = math.log2(unique_count) if unique_count > 1 else 1
    
    print(f"\nWord count: {word_count}")
    print(f"Unique count: {unique_count}")
    print(f"TTR: {ttr}")
    print(f"Probability sum: {prob_sum:.4f}")
    print(f"Entropy: {entropy}")
    print(f"Max entropy: {ent_max}")
    print(f"Normalized entropy: {entropy/ent_max*100:.4f}%")
    print(f"Bpse: {entropy * ttr}")
    print(f"Lexical tension (VSS Response): {sum(tension_scores)/len(tension_scores):.4f}")

if __name__ == "__main__":
    main(sys.argv[1:])

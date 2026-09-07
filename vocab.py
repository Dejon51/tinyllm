import re
from collections import Counter
import sys

def build_vocab(input_file, output_file, vocab_size=10000):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    # Remove punctuation and digits (keep apostrophes for contractions)
    text = re.sub(r"[^a-z'\s]", ' ', text)
    words = text.split()
    word_counts = Counter(words)
    most_common = word_counts.most_common(vocab_size)
    vocab = ['<UNK>'] + [word for word, _ in most_common]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))
    print(f"Vocabulary saved to {output_file} ({len(vocab)} tokens)")

if __name__ == '__main__':
    build_vocab('data.txt', 'vocab.txt', 10000)
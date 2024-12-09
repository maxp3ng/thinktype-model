import pandas as pd
from itertools import product

def generate_vowel_patterns():
    """Generate systematic vowel dropping patterns"""
    base_pairs = [
        ("the", "th"),
        ("quick", "qk", "qck", "quik"),
        ("brown", "brwn", "brn"),
        ("fox", "fx"),
        ("jumps", "jmps", "jumps"),
        ("over", "ovr", "ovr"),
        ("lazy", "lzy", "lazy"),
        ("dog", "dg", "dog"),
    ]
    
    # Generate full sentences from combinations
    full_sentence = "the quick brown fox jumps over the lazy dog"
    training_pairs = []
    
    # Add the classic example
    training_pairs.append(("th qk brwn fx jmps ovr th lzy dg", full_sentence))
    
    # Generate variations
    words = [p[1:] for p in base_pairs]  # Get all variations for each word
    for variation in product(*words):
        shorthand = " ".join(variation)
        training_pairs.append((shorthand, full_sentence))
    
    return training_pairs

def generate_common_patterns():
    """Generate common shorthand patterns"""
    patterns = [
        # Basic words with multiple shorthand versions
        ("what", ["wht", "wt", "wat"]),
        ("your", ["yr", "ur", "yor"]),
        ("please", ["pls", "plz", "plse"]),
        ("thanks", ["thx", "thks", "thnx"]),
        ("good", ["gd", "gud", "good"]),
        ("great", ["grt", "gr8", "great"]),
        
        # Common phrases
        ("how are you", ["hw r u", "how r u", "hw are u"]),
        ("be right back", ["brb", "b rght bk", "be rt bk"]),
        ("as soon as possible", ["asap", "as sn as psbl", "asap psbl"]),
        ("in my opinion", ["imo", "in my opn", "n my opn"]),
        
        # Technical terms
        ("function", ["func", "fn", "fnctn"]),
        ("parameter", ["param", "prm", "prmtr"]),
        ("variable", ["var", "vr", "vrbl"]),
        ("implementation", ["impl", "impln", "implmtn"]),
    ]
    
    pairs = []
    for full, shorts in patterns:
        for short in shorts:
            pairs.append((short, full))
    
    return pairs

# Combine all our training data
training_data = generate_vowel_patterns() + generate_common_patterns()

# Add some full context examples
context_examples = [
    ("i nd to fx ths asap", "i need to fix this as soon as possible"),
    ("cn u hlp w the impl", "can you help with the implementation"),
    ("yr func is nt wrkng", "your function is not working"),
    ("gd wrk on the prjct", "good work on the project")
]
training_data.extend(context_examples)

# Convert to DataFrame and save
df = pd.DataFrame(training_data, columns=['shorthand', 'fulltext'])
df = df.drop_duplicates()

# Save to CSV
df.to_csv('thinktype_training_data_v2.csv', index=False)

print(f"Created dataset with {len(df)} examples")
print("\nSample entries:")
print(df.head(10).to_string())
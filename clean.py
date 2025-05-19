input_file = 'CompCars/train_test_split/classification/train.txt'  # ou train_all.txt si tu as ce fichier
output_file = 'CompCars/train_test_split/classification_clean/train_all_clean.txt'

clean_paths = []
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            clean_paths.append(line)

with open(output_file, 'w') as f:
    for path in clean_paths:
        f.write(path + '\n')

print(f"Fichier nettoyé écrit dans {output_file} ({len(clean_paths)} lignes)")

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse

def load_results(out_root):
    results = []
    # Search for results.json recursively
    files = glob.glob(os.path.join(out_root, "**", "results.json"), recursive=True)
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                
            args = data.get('args', {})
            # Parse info from args or folder name
            # We rely on args stored in json
            
            entry = {
                'Model': data.get('model_name'),
                'Test Acc': data.get('test_acc'),
                'Best Val Acc': data.get('best_val_acc'),
                'Seq Len': args.get('data_dir', '').split('seq')[1].split('_')[0] if 'data_dir' in args else None,
                'Key Len': args.get('data_dir', '').split('key')[1].split('_')[0] if 'data_dir' in args else None,
                'Seed': args.get('seed'),
                'Path': f
            }
            
            # Refine Seq/Key parsing if needed, assuming data_dir naming convention from run script:
            # ".../seq{seq_len}_key{key_len}_seed{seed}"
            if entry['Seq Len'] is None:
                # Try parsing from exp_name
                exp_name = args.get('exp_name', '')
                if 'seq' in exp_name:
                    entry['Seq Len'] = int(exp_name.split('seq')[1].split('_')[0])
                if 'key' in exp_name:
                    entry['Key Len'] = int(exp_name.split('key')[1].split('_')[0])
            else:
                entry['Seq Len'] = int(entry['Seq Len'])
                entry['Key Len'] = int(entry['Key Len'])
                
            results.append(entry)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    return pd.DataFrame(results)

def plot_results(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Accuracy vs Seq Len (for Key Len = 1)
    subset = df[df['Key Len'] == 1]
    if not subset.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x='Seq Len', y='Test Acc', hue='Model', marker='o')
        plt.title('MQAR Accuracy vs Sequence Length (2-gram)')
        plt.ylabel('Test Accuracy')
        plt.xlabel('Sequence Length')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'acc_vs_seqlen_2gram.png'))
        plt.close()
        
    # 2. Accuracy vs Key Len (for Seq Len = 256 or most common)
    common_seq_len = df['Seq Len'].mode()[0] if not df.empty else 256
    subset = df[df['Seq Len'] == common_seq_len]
    if not subset.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=subset, x='Key Len', y='Test Acc', hue='Model', marker='o')
        plt.title(f'MQAR Accuracy vs Association Order (Seq Len {common_seq_len})')
        plt.ylabel('Test Accuracy')
        plt.xlabel('Key Length (1=2-gram, 2=3-gram...)')
        plt.xticks(sorted(df['Key Len'].unique()))
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'acc_vs_keylen_seq{common_seq_len}.png'))
        plt.close()

    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="./out/mqar")
    parser.add_argument("--save_dir", type=str, default="./analysis_results")
    args = parser.parse_args()
    
    df = load_results(args.out_root)
    if df.empty:
        print("No results found.")
        return
        
    print("Loaded Results:")
    print(df.groupby(['Model', 'Seq Len', 'Key Len'])['Test Acc'].mean())
    
    df.to_csv(os.path.join(args.save_dir, "summary.csv"), index=False)
    plot_results(df, args.save_dir)

if __name__ == "__main__":
    main()

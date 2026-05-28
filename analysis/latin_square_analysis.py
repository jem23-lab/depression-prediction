"""
Latin Square Design Analysis
Проверка распределения selected_use_case_name для каждого (user_id, paragraph_id) пары.
Подсчет сколько раз каждая пара use cases оценивается.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
LOGS_DIR = Path(__file__).parent.parent / "logs"
EVALUATION_FILE = LOGS_DIR / "evaluation_records.csv"
OUTPUT_DIR = Path(__file__).parent / "analysis_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load evaluation records."""
    df = pd.read_csv(EVALUATION_FILE)
    print(f"Loaded {len(df)} records")
    return df


def analyze_latin_square(df):
    """Analyze Latin square design."""

    print("\n" + "="*80)
    print("LATIN SQUARE DESIGN ANALYSIS")
    print("="*80)

    # 1. Check how many use cases per (user_id, paragraph_id)
    print("\n1. USE CASES PER (USER_ID, PARAGRAPH_ID) PAIR:")
    print("-"*80)

    user_para_cases = df.groupby(['user_id', 'paragraph_id'])['selected_use_case_name'].apply(
        lambda x: list(x.dropna().unique())
    )

    case_counts = user_para_cases.apply(len)
    print(f"\nDistribution of use case counts per (user_id, paragraph_id):")
    print(case_counts.value_counts().sort_index())

    print(f"\nTotal (user_id, paragraph_id) pairs: {len(user_para_cases)}")
    print(f"Pairs with exactly 2 use cases: {(case_counts == 2).sum()}")
    print(f"Pairs with exactly 4 use cases: {(case_counts == 4).sum()}")

    # 2. Get all possible use cases
    all_use_cases = sorted(df['selected_use_case_name'].dropna().unique())
    print(f"\nAll use cases ({len(all_use_cases)}): {', '.join(all_use_cases)}")

    # 3. Count use case pairs
    print("\n2. USE CASE PAIRS FREQUENCIES:")
    print("-"*80)

    pair_counts = {}

    for user_para, cases in user_para_cases.items():
        if len(cases) >= 2:
            # Get all combinations of cases
            for pair in combinations(sorted(cases), 2):
                pair_key = tuple(sorted(pair))
                pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    # Sort by frequency
    sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTotal unique use case pairs: {len(pair_counts)}")
    print("\nPair frequencies:")
    for pair, count in sorted_pairs:
        print(f"  {pair[0]} ← → {pair[1]}: {count} times")

    # Check if balanced
    frequencies = [count for _, count in sorted_pairs]
    if len(set(frequencies)) == 1:
        print(f"\n✅ BALANCED: All pairs appear exactly {frequencies[0]} times")
    else:
        print(f"\n⚠️ UNBALANCED: Pairs appear {min(frequencies)} to {max(frequencies)} times")
        print(f"   Range: {max(frequencies) - min(frequencies)}")

    # 4. User statistics
    print("\n3. USER STATISTICS:")
    print("-"*80)

    user_counts = df.groupby('user_id').size()
    print(f"\nRecords per user:")
    print(f"  Mean: {user_counts.mean():.2f}")
    print(f"  Median: {user_counts.median():.2f}")
    print(f"  Min: {user_counts.min()}")
    print(f"  Max: {user_counts.max()}")

    unique_users = df['user_id'].nunique()
    unique_paras = df['paragraph_id'].nunique()
    unique_cases = df['selected_use_case_name'].nunique()

    print(f"\nUnique values:")
    print(f"  Users: {unique_users}")
    print(f"  Paragraphs: {unique_paras}")
    print(f"  Use cases: {unique_cases}")

    # 5. Use case statistics
    print("\n4. USE CASE DISTRIBUTION:")
    print("-"*80)

    case_counts_total = df['selected_use_case_name'].value_counts()
    print("\nTotal records per use case:")
    for case, count in case_counts_total.items():
        pct = 100 * count / len(df)
        print(f"  {case}: {count} ({pct:.1f}%)")

    # 6. Check for specific patterns
    print("\n5. DESIGN PATTERNS:")
    print("-"*80)

    # Count how many user-paragraph pairs have each specific pair
    pair_by_user_para = {}
    for (user, para), cases in user_para_cases.items():
        if len(cases) >= 2:
            pair = tuple(sorted(cases[:2]))
            key = (user, para)
            pair_by_user_para[key] = pair

    # Check if same pair appears for multiple user-paragraph combinations
    print("\nUser-paragraph combinations per pair:")
    for pair, count in sorted_pairs:
        matching = sum(1 for p in pair_by_user_para.values() if p == pair)
        print(f"  {pair[0]} ← → {pair[1]}: {matching} (user, paragraph) pairs")

    return {
        'user_para_cases': user_para_cases,
        'pair_counts': pair_counts,
        'all_use_cases': all_use_cases,
        'unique_users': unique_users,
        'unique_paras': unique_paras
    }


def create_heatmap(pair_counts, all_use_cases, output_dir):
    """Create heatmap of use case pair frequencies."""

    # Create matrix
    n = len(all_use_cases)
    matrix = pd.DataFrame(0, index=all_use_cases, columns=all_use_cases)

    for (case1, case2), count in pair_counts.items():
        matrix.loc[case1, case2] = count
        matrix.loc[case2, case1] = count

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                cbar_kws={"label": "Frequency"},
                ax=ax, linewidths=1, linecolor='gray')

    ax.set_title('Latin Square Design: Use Case Pair Frequencies\n(How often each pair is evaluated together)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Use Case', fontsize=12)
    ax.set_ylabel('Use Case', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_dir / "latin_square_pair_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: latin_square_pair_heatmap.png")
    plt.close()


def create_summary_report(analysis_result, output_dir):
    """Create a summary report."""

    report_path = output_dir / "latin_square_analysis.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("LATIN SQUARE DESIGN ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("OVERVIEW:\n")
        f.write("-"*80 + "\n")
        f.write(f"Users: {analysis_result['unique_users']}\n")
        f.write(f"Paragraphs: {analysis_result['unique_paras']}\n")
        f.write(f"Use cases: {len(analysis_result['all_use_cases'])}\n")
        f.write(f"Use cases: {', '.join(analysis_result['all_use_cases'])}\n\n")

        f.write("USE CASE PAIR FREQUENCIES:\n")
        f.write("-"*80 + "\n")

        sorted_pairs = sorted(analysis_result['pair_counts'].items(),
                            key=lambda x: x[1], reverse=True)

        for pair, count in sorted_pairs:
            f.write(f"{pair[0]} ← → {pair[1]}: {count}\n")

        frequencies = [count for _, count in sorted_pairs]
        min_freq = min(frequencies)
        max_freq = max(frequencies)

        f.write(f"\nMin frequency: {min_freq}\n")
        f.write(f"Max frequency: {max_freq}\n")
        f.write(f"Range: {max_freq - min_freq}\n")

        if len(set(frequencies)) == 1:
            f.write(f"\n✅ STATUS: BALANCED DESIGN\n")
            f.write(f"All pairs appear exactly {frequencies[0]} times\n")
        else:
            f.write(f"\n⚠️ STATUS: UNBALANCED DESIGN\n")
            f.write(f"Imbalance range: {max_freq - min_freq}\n")
            imbalance_pct = 100 * (max_freq - min_freq) / min_freq if min_freq > 0 else 0
            f.write(f"Imbalance percentage: {imbalance_pct:.1f}%\n")

    print(f"✓ Saved: latin_square_analysis.txt")


def main():
    """Main analysis function."""
    print("Loading evaluation records...")
    df = load_data()

    if df.empty:
        print("ERROR: No data found!")
        return

    # Run analysis
    analysis_result = analyze_latin_square(df)

    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    create_heatmap(analysis_result['pair_counts'],
                   analysis_result['all_use_cases'],
                   OUTPUT_DIR)

    # Create report
    create_summary_report(analysis_result, OUTPUT_DIR)

    print("\n" + "="*80)
    print("✓ LATIN SQUARE ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()


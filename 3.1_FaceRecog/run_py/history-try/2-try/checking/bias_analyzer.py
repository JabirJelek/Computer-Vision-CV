import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class BiasAnalyzer:
    def __init__(self, dataset_json: str):
        self.dataset = self._load_dataset(dataset_json)
        self.analysis_results = {}
    
    def _load_dataset(self, dataset_json: str):
        with open(dataset_json, 'r') as f:
            return json.load(f)
    
    def analyze_embedding_distribution(self):
        """Analyze how embeddings are distributed across persons"""
        persons = self.dataset["persons"]
        
        embedding_counts = []
        for person_id, person_data in persons.items():
            embedding_counts.append({
                'person_id': person_id,
                'name': person_data['display_name'],
                'count': person_data['successful_embeddings'],
                'folder': person_data['folder_name']
            })
        
        # Sort by count (descending)
        embedding_counts.sort(key=lambda x: x['count'], reverse=True)
        
        print("📊 EMBEDDING DISTRIBUTION ANALYSIS")
        print("=" * 60)
        
        for i, person in enumerate(embedding_counts[:10]):  # Top 10
            print(f"{i+1:2d}. {person['name']:20} - {person['count']:3d} embeddings")
        
        # Statistics
        counts = [p['count'] for p in embedding_counts]
        avg_embeddings = np.mean(counts)
        std_embeddings = np.std(counts)
        
        print(f"\n📈 Distribution Statistics:")
        print(f"   Average embeddings per person: {avg_embeddings:.1f}")
        print(f"   Standard deviation: {std_embeddings:.1f}")
        print(f"   Most over-represented: {embedding_counts[0]['name']} ({embedding_counts[0]['count']} embeddings)")
        print(f"   Most under-represented: {embedding_counts[-1]['name']} ({embedding_counts[-1]['count']} embeddings)")
        
        # Identify potential bias
        bias_threshold = avg_embeddings + std_embeddings
        over_represented = [p for p in embedding_counts if p['count'] > bias_threshold]
        
        if over_represented:
            print(f"\n⚠️  POTENTIAL BIAS DETECTED:")
            for person in over_represented:
                print(f"   {person['name']}: {person['count']} embeddings "
                      f"(>{bias_threshold:.1f} threshold)")
        
        self.analysis_results['distribution'] = {
            'counts': embedding_counts,
            'stats': {
                'mean': avg_embeddings,
                'std': std_embeddings,
                'bias_threshold': bias_threshold
            },
            'over_represented': over_represented
        }
        
        return embedding_counts
    
    def plot_embedding_distribution(self):
        """Visualize the embedding distribution"""
        if 'distribution' not in self.analysis_results:
            self.analyze_embedding_distribution()
        
        distribution = self.analysis_results['distribution']
        counts = [p['count'] for p in distribution['counts']]
        names = [p['name'] for p in distribution['counts']]
        
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Bar chart
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(counts)), counts, color='skyblue', alpha=0.7)
        
        # Color over-represented persons in red
        bias_threshold = distribution['stats']['bias_threshold']
        for i, count in enumerate(counts):
            if count > bias_threshold:
                bars[i].set_color('red')
        
        plt.axhline(y=bias_threshold, color='red', linestyle='--', 
                   label=f'Bias Threshold ({bias_threshold:.1f})')
        plt.axhline(y=distribution['stats']['mean'], color='blue', linestyle='--',
                   label=f'Average ({distribution["stats"]["mean"]:.1f})')
        
        plt.xlabel('Persons')
        plt.ylabel('Number of Embeddings')
        plt.title('Embedding Distribution Across Persons')
        plt.legend()
        plt.xticks(rotation=45)
        
        # Plot 2: Histogram
        plt.subplot(1, 2, 2)
        plt.hist(counts, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.axvline(distribution['stats']['mean'], color='blue', linestyle='--', 
                   label=f'Mean: {distribution["stats"]["mean"]:.1f}')
        plt.xlabel('Embeddings per Person')
        plt.ylabel('Frequency')
        plt.title('Distribution of Embedding Counts')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def suggest_balancing_strategy(self):
        """Suggest strategies to balance the dataset"""
        if 'distribution' not in self.analysis_results:
            self.analyze_embedding_distribution()
        
        distribution = self.analysis_results['distribution']
        stats = distribution['stats']
        
        print("\n🎯 DATASET BALANCING STRATEGIES")
        print("=" * 50)
        
        if distribution['over_represented']:
            print("🔴 ISSUE: Some persons are over-represented")
            print("💡 SOLUTIONS:")
            print("   1. Limit maximum embeddings per person")
            print("   2. Remove some embeddings from over-represented persons")
            print("   3. Add more embeddings to under-represented persons")
            print("   4. Use weighted similarity scoring")
        else:
            print("✅ Dataset is reasonably balanced")
        
        # Specific recommendations
        max_recommended = int(stats['mean'] + stats['std'])
        min_recommended = max(3, int(stats['mean'] - stats['std']))
        
        print(f"\n📝 SPECIFIC RECOMMENDATIONS:")
        print(f"   Target range: {min_recommended} - {max_recommended} embeddings per person")
        print(f"   Current range: {min([p['count'] for p in distribution['counts']])} - {max([p['count'] for p in distribution['counts']])}")

def main():
    # Analyze your dataset
    dataset_json = input("Enter dataset json path: ").strip()  # Your dataset path
    
    analyzer = BiasAnalyzer(dataset_json)
    
    # Run analysis
    analyzer.analyze_embedding_distribution()
    
    # Show visualization
    analyzer.plot_embedding_distribution()
    
    # Get recommendations
    analyzer.suggest_balancing_strategy()

if __name__ == "__main__":
    main()
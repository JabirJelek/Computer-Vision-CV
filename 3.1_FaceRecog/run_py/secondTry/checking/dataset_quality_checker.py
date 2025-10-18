import json
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

class DatasetQualityChecker:
    def __init__(self, dataset_json: str):
        self.dataset = self._load_dataset(dataset_json)
        self.quality_report = {}
    
    def _load_dataset(self, dataset_json: str):
        with open(dataset_json, 'r') as f:
            return json.load(f)
    
    def analyze_embedding_sufficiency(self):
        """
        Determine if each person has enough embeddings for reliable recognition
        """
        persons = self.dataset["persons"]
        
        analysis = {
            'embedding_counts': [],
            'quality_categories': {'excellent': [], 'good': [], 'minimal': [], 'insufficient': []},
            'recommendations': []
        }
        
        for person_id, person_data in persons.items():
            count = person_data['successful_embeddings']
            
            # Categorize based on embedding count
            if count >= 15:
                category = 'excellent'
                recommendation = 'More than enough for robust recognition'
            elif count >= 8:
                category = 'good' 
                recommendation = 'Sufficient for reliable recognition'
            elif count >= 3:
                category = 'minimal'
                recommendation = 'Bare minimum - consider adding more variety'
            else:
                category = 'insufficient'
                recommendation = 'Too few - will likely cause recognition issues'
            
            analysis['embedding_counts'].append({
                'person_id': person_id,
                'name': person_data['display_name'],
                'count': count,
                'category': category,
                'recommendation': recommendation
            })
            
            analysis['quality_categories'][category].append(person_id)
        
        # Sort by count
        analysis['embedding_counts'].sort(key=lambda x: x['count'], reverse=True)
        
        return analysis
    
    def calculate_intra_person_variance(self):
        """
        Calculate how consistent each person's embeddings are
        Low variance = very similar embeddings (maybe too similar!)
        High variance = diverse representations (good for robustness)
        """
        variance_analysis = {}
        
        for person_id, person_data in self.dataset["persons"].items():
            embeddings = [np.array(emb["vector"]) for emb in person_data["embeddings"]]
            
            if len(embeddings) > 1:
                # Calculate pairwise distances between embeddings of same person
                centroid = np.mean(embeddings, axis=0)
                distances = [np.linalg.norm(emb - centroid) for emb in embeddings]
                variance = np.var(distances)
                
                # Normalize variance for interpretation
                avg_distance = np.mean(distances)
                
                variance_analysis[person_id] = {
                    'variance': float(variance),
                    'avg_distance_from_centroid': float(avg_distance),
                    'embedding_count': len(embeddings),
                    'consistency_level': 'high' if variance < 0.1 else 'medium' if variance < 0.3 else 'low'
                }
        
        return variance_analysis
    
    def check_diversity_quality(self):
        """
        Check if embeddings are diverse enough (not just duplicates)
        """
        diversity_scores = {}
        
        for person_id, person_data in self.dataset["persons"].items():
            embeddings = [np.array(emb["vector"]) for emb in person_data["embeddings"]]
            
            if len(embeddings) < 2:
                diversity_scores[person_id] = {'score': 0, 'status': 'unknown'}
                continue
            
            # Calculate minimum similarity between embeddings (shouldn't be too high)
            min_similarity = 1.0
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    similarity = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                    min_similarity = min(min_similarity, similarity)
            
            diversity_scores[person_id] = {
                'min_similarity': float(min_similarity),
                'status': 'diverse' if min_similarity < 0.7 else 'redundant'
            }
        
        return diversity_scores
    
    def generate_quality_report(self):
        """
        Generate comprehensive quality report
        """
        print("🔍 DATASET QUALITY ANALYSIS REPORT")
        print("=" * 60)
        
        # 1. Basic statistics
        total_persons = len(self.dataset["persons"])
        total_embeddings = self.dataset["metadata"]["total_embeddings"]
        avg_embeddings = self.dataset["metadata"]["average_embeddings_per_person"]
        
        print(f"\n📊 BASIC STATISTICS:")
        print(f"   Total persons: {total_persons}")
        print(f"   Total embeddings: {total_embeddings}")
        print(f"   Average embeddings per person: {avg_embeddings:.1f}")
        
        # 2. Sufficiency analysis
        sufficiency = self.analyze_embedding_sufficiency()
        
        print(f"\n🎯 EMBEDDING SUFFICIENCY:")
        categories = sufficiency['quality_categories']
        for category, persons in categories.items():
            print(f"   {category.capitalize():>12}: {len(persons)} persons")
        
        # 3. Show extremes
        counts = sufficiency['embedding_counts']
        if counts:
            print(f"\n📈 EXTREMES:")
            print(f"   Most represented: {counts[0]['name']} ({counts[0]['count']} embeddings)")
            print(f"   Least represented: {counts[-1]['name']} ({counts[-1]['count']} embeddings)")
        
        # 4. Variance analysis
        variance_data = self.calculate_intra_person_variance()
        
        print(f"\n📏 EMBEDDING DIVERSITY (Variance Analysis):")
        consistency_levels = {'high': 0, 'medium': 0, 'low': 0}
        for person_id, data in variance_data.items():
            consistency_levels[data['consistency_level']] += 1
        
        for level, count in consistency_levels.items():
            print(f"   {level.capitalize():>6} consistency: {count} persons")
        
        # 5. Diversity quality
        diversity_data = self.check_diversity_quality()
        diverse_count = sum(1 for data in diversity_data.values() if data.get('status') == 'diverse')
        redundant_count = sum(1 for data in diversity_data.values() if data.get('status') == 'redundant')
        
        print(f"\n🔄 EMBEDDING QUALITY:")
        print(f"   Diverse embeddings: {diverse_count} persons")
        print(f"   Redundant embeddings: {redundant_count} persons")
        
        # 6. Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Check for imbalance
        if len(categories['insufficient']) > 0:
            print(f"   ⚠️  Add more images for {len(categories['insufficient'])} under-represented persons")
        
        if len(categories['excellent']) > len(categories['minimal']) + len(categories['insufficient']):
            print(f"   ⚠️  Consider reducing embeddings for over-represented persons")
        
        if redundant_count > diverse_count:
            print(f"   ⚠️  Improve embedding diversity - too many similar faces")
        
        if avg_embeddings >= 8:
            print(f"   ✅ Your dataset size is sufficient for research")
        else:
            print(f"   🔄 Consider collecting 2-3 more images per person")
        
        # Store comprehensive report
        self.quality_report = {
            'sufficiency_analysis': sufficiency,
            'variance_analysis': variance_data,
            'diversity_analysis': diversity_data,
            'summary': {
                'total_persons': total_persons,
                'total_embeddings': total_embeddings,
                'quality_score': self._calculate_quality_score(sufficiency, variance_data, diversity_data)
            }
        }
        
        return self.quality_report
    
    def _calculate_quality_score(self, sufficiency, variance, diversity):
        """Calculate overall dataset quality score (0-100)"""
        # Simple heuristic scoring
        score = 0
        
        # Sufficiency score (40%)
        categories = sufficiency['quality_categories']
        total_persons = len(sufficiency['embedding_counts'])
        
        sufficiency_score = (
            (len(categories['excellent']) * 1.0 +
             len(categories['good']) * 0.8 +
             len(categories['minimal']) * 0.5 +
             len(categories['insufficient']) * 0.1) / total_persons * 40
        )
        
        # Diversity score (30%)
        diverse_count = sum(1 for data in diversity.values() if data.get('status') == 'diverse')
        diversity_score = (diverse_count / total_persons) * 30 if total_persons > 0 else 0
        
        # Balance score (30%)
        counts = [p['count'] for p in sufficiency['embedding_counts']]
        if counts:
            balance_ratio = min(counts) / max(counts) if max(counts) > 0 else 0
            balance_score = balance_ratio * 30
        else:
            balance_score = 0
        
        return round(sufficiency_score + diversity_score + balance_score, 1)
    
    def visualize_quality_metrics(self):
        """Create visualization of dataset quality"""
        if not self.quality_report:
            self.generate_quality_report()
        
        report = self.quality_report
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Dataset Quality Analysis', fontsize=16)
        
        # Plot 1: Embedding distribution
        counts = [p['count'] for p in report['sufficiency_analysis']['embedding_counts']]
        names = [p['name'] for p in report['sufficiency_analysis']['embedding_counts']]
        
        axes[0, 0].bar(range(len(counts)), counts, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Embeddings per Person')
        axes[0, 0].set_xlabel('Persons')
        axes[0, 0].set_ylabel('Number of Embeddings')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Quality categories
        categories = report['sufficiency_analysis']['quality_categories']
        category_counts = [len(persons) for persons in categories.values()]
        category_names = list(categories.keys())
        
        colors = ['green', 'lightgreen', 'orange', 'red']
        axes[0, 1].pie(category_counts, labels=category_names, colors=colors, autopct='%1.1f%%')
        axes[0, 1].set_title('Embedding Sufficiency Categories')
        
        # Plot 3: Variance distribution
        if report['variance_analysis']:
            variances = [data['variance'] for data in report['variance_analysis'].values()]
            axes[1, 0].hist(variances, bins=15, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 0].set_title('Embedding Variance Distribution')
            axes[1, 0].set_xlabel('Variance')
            axes[1, 0].set_ylabel('Frequency')
        
        # Plot 4: Quality score
        quality_score = report['summary']['quality_score']
        axes[1, 1].barh(['Overall Quality'], [quality_score], color='skyblue')
        axes[1, 1].set_xlim(0, 100)
        axes[1, 1].set_title(f'Overall Quality Score: {quality_score}/100')
        axes[1, 1].axvline(x=80, color='green', linestyle='--', alpha=0.7, label='Good (80+)')
        axes[1, 1].axvline(x=60, color='orange', linestyle='--', alpha=0.7, label='Acceptable (60+)')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    # Analyze your dataset quality
    dataset_json = "simple_face_dataset.json"  # Your dataset path
    
    checker = DatasetQualityChecker(dataset_json)
    
    # Generate comprehensive report
    report = checker.generate_quality_report()
    
    # Show visualizations
    checker.visualize_quality_metrics()
    
    # Print detailed recommendations
    print("\n" + "="*60)
    print("🎯 ACTIONABLE RECOMMENDATIONS")
    print("="*60)
    
    sufficiency = report['sufficiency_analysis']
    
    # Check for specific issues
    insufficient = sufficiency['quality_categories']['insufficient']
    if insufficient:
        print(f"\n🔴 CRITICAL: {len(insufficient)} persons have insufficient embeddings:")
        for person_id in insufficient[:5]:  # Show first 5
            person_data = checker.dataset['persons'][person_id]
            print(f"   - {person_data['display_name']}: {person_data['successful_embeddings']} embeddings")
    
    # Check for over-representation
    excellent = sufficiency['quality_categories']['excellent']
    minimal_plus_insufficient = len(sufficiency['quality_categories']['minimal']) + len(sufficiency['quality_categories']['insufficient'])
    
    if len(excellent) > minimal_plus_insufficient:
        print(f"\n⚠️  IMBALANCE: {len(excellent)} over-represented vs {minimal_plus_insufficient} under-represented persons")
        print("   Consider reducing embeddings for over-represented persons")

if __name__ == "__main__":
    main()
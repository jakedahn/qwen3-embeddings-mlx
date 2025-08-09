#!/usr/bin/env python3
"""
Generate thematic text embeddings and export for TensorFlow Embedding Projector

This script:
1. Creates diverse thematic text samples
2. Generates embeddings using each model
3. Exports in TSV format for visualization
4. Validates embedding quality through similarity metrics
"""

import json
import time
import requests
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

# Thematic text samples organized by category
THEMATIC_TEXTS = {
    "technology": [
        "Machine learning algorithms revolutionize data analysis",
        "Artificial intelligence transforms modern computing",
        "Neural networks mimic human brain processing",
        "Deep learning enables complex pattern recognition",
        "Quantum computing promises exponential speedup",
        "Cloud infrastructure scales enterprise applications",
        "Edge computing brings processing closer to data",
        "Blockchain technology ensures data immutability"
    ],
    "nature": [
        "Majestic mountains pierce the morning clouds",
        "Ocean waves crash against rocky coastlines",
        "Ancient forests harbor diverse ecosystems",
        "Desert landscapes showcase nature's resilience",
        "Coral reefs teem with vibrant marine life",
        "Arctic tundra endures extreme conditions",
        "Rainforests produce Earth's vital oxygen",
        "Rivers carve valleys through solid rock"
    ],
    "emotions": [
        "Joy spreads through genuine human connections",
        "Sadness teaches us about loss and growth",
        "Anger signals boundaries need protection",
        "Fear keeps us safe from potential danger",
        "Love binds families and communities together",
        "Hope sustains us through difficult times",
        "Gratitude enhances mental wellbeing significantly",
        "Empathy bridges cultural and social divides"
    ],
    "science": [
        "Gravity warps spacetime around massive objects",
        "Evolution shapes life through natural selection",
        "Atoms form the building blocks of matter",
        "DNA encodes genetic information in sequences",
        "Photosynthesis converts sunlight into energy",
        "Chemical reactions transform molecular structures",
        "Physics governs fundamental universal laws",
        "Biology explores living organism complexity"
    ],
    "food": [
        "Fresh pasta absorbs rich tomato sauce perfectly",
        "Sushi combines fresh fish with seasoned rice",
        "Chocolate melts smoothly on the tongue",
        "Coffee aroma fills morning kitchens worldwide",
        "Bread rises through yeast fermentation process",
        "Spices transform simple ingredients dramatically",
        "Wine complexity develops through careful aging",
        "Cheese varieties reflect regional traditions"
    ],
    "sports": [
        "Athletes push physical limits in competition",
        "Basketball requires teamwork and individual skill",
        "Marathon runners demonstrate incredible endurance",
        "Soccer unites fans across global cultures",
        "Tennis matches showcase strategic mental games",
        "Swimming builds full-body strength and stamina",
        "Gymnastics combines grace with raw power",
        "Baseball strategy unfolds over nine innings"
    ],
    "art": [
        "Paintings capture moments frozen in time",
        "Sculpture transforms raw materials into meaning",
        "Music evokes powerful emotional responses",
        "Poetry distills language to its essence",
        "Dance expresses stories through movement",
        "Photography documents reality and imagination",
        "Theater brings written words to life",
        "Architecture shapes how we inhabit spaces"
    ],
    "travel": [
        "Exploring new cultures broadens perspectives immensely",
        "Mountain trails reward hikers with stunning vistas",
        "City streets pulse with urban energy",
        "Beach sunsets create unforgettable memories",
        "Ancient ruins tell stories of past civilizations",
        "Local cuisine reveals cultural heritage deeply",
        "Train journeys offer scenic countryside views",
        "Backpacking adventures foster personal growth"
    ]
}

# Add some outlier/noise samples to test clustering
OUTLIER_TEXTS = [
    "xyzabc random gibberish text qwerty",
    "The quick brown fox jumps over the lazy dog",
    "Lorem ipsum dolor sit amet consectetur",
    "1234567890 numeric sequence test pattern",
    "!@#$%^&* special characters punctuation marks"
]

class EmbeddingGenerator:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def generate_embeddings(self, texts: List[str], model: str = "small") -> np.ndarray:
        """Generate embeddings for a list of texts"""
        response = self.session.post(
            f"{self.base_url}/embed_batch",
            json={"texts": texts, "model": model}
        )
        response.raise_for_status()
        result = response.json()
        return np.array(result["embeddings"])
    
    def check_server(self) -> bool:
        """Check if server is available"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False

def prepare_samples() -> Tuple[List[str], List[str], List[str]]:
    """Prepare text samples with labels and categories"""
    texts = []
    labels = []
    categories = []
    
    # Add thematic texts
    for category, samples in THEMATIC_TEXTS.items():
        for i, text in enumerate(samples):
            texts.append(text)
            labels.append(f"{category}_{i+1}")
            categories.append(category)
    
    # Add outliers
    for i, text in enumerate(OUTLIER_TEXTS):
        texts.append(text)
        labels.append(f"outlier_{i+1}")
        categories.append("outlier")
    
    return texts, labels, categories

def validate_embeddings(embeddings: np.ndarray, labels: List[str], categories: List[str]) -> Dict:
    """Validate embedding quality through similarity analysis"""
    results = {}
    
    # Calculate intra-category similarities
    category_sims = {}
    unique_categories = list(set(categories))
    
    for cat in unique_categories:
        if cat == "outlier":
            continue
        indices = [i for i, c in enumerate(categories) if c == cat]
        if len(indices) > 1:
            cat_embeddings = embeddings[indices]
            sims = cosine_similarity(cat_embeddings)
            # Get average similarity excluding diagonal
            mask = ~np.eye(sims.shape[0], dtype=bool)
            category_sims[cat] = float(np.mean(sims[mask]))
    
    results["intra_category_similarity"] = category_sims
    
    # Calculate inter-category similarities (should be lower)
    inter_sims = []
    for i, cat1 in enumerate(unique_categories[:-1]):
        if cat1 == "outlier":
            continue
        for cat2 in unique_categories[i+1:]:
            if cat2 == "outlier":
                continue
            indices1 = [j for j, c in enumerate(categories) if c == cat1]
            indices2 = [j for j, c in enumerate(categories) if c == cat2]
            if indices1 and indices2:
                sims = cosine_similarity(embeddings[indices1], embeddings[indices2])
                inter_sims.append(float(np.mean(sims)))
    
    results["avg_inter_category_similarity"] = float(np.mean(inter_sims)) if inter_sims else 0
    
    # Check embedding norms (should be ~1.0 for normalized)
    norms = np.linalg.norm(embeddings, axis=1)
    results["avg_norm"] = float(np.mean(norms))
    results["norm_std"] = float(np.std(norms))
    
    return results

def export_for_projector(embeddings: np.ndarray, labels: List[str], categories: List[str], 
                         texts: List[str], model_name: str, output_dir: Path):
    """Export embeddings in TSV format for TensorFlow Embedding Projector"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save embeddings
    vectors_file = output_dir / f"vectors_{model_name}.tsv"
    np.savetxt(vectors_file, embeddings, delimiter='\t', fmt='%.6f')
    
    # Save metadata
    metadata_file = output_dir / f"metadata_{model_name}.tsv"
    with open(metadata_file, 'w') as f:
        f.write("Label\tCategory\tText\n")
        for label, category, text in zip(labels, categories, texts):
            # Escape tabs and newlines in text
            clean_text = text.replace('\t', ' ').replace('\n', ' ')
            f.write(f"{label}\t{category}\t{clean_text}\n")
    
    # Save config for projector
    config = {
        "embeddings": [
            {
                "tensorName": f"Qwen3 {model_name} Embeddings",
                "tensorShape": list(embeddings.shape),
                "tensorPath": vectors_file.name,
                "metadataPath": metadata_file.name
            }
        ]
    }
    
    config_file = output_dir / f"config_{model_name}.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return vectors_file, metadata_file, config_file

def analyze_model_differences(all_embeddings: Dict[str, np.ndarray], texts: List[str]):
    """Analyze how different models represent the same text"""
    console.print("\n[bold cyan]📊 Model Comparison Analysis[/bold cyan]")
    console.print("[dim]Note: Cannot directly compare embeddings of different dimensions[/dim]")
    console.print("[dim]Small: 1024d, Medium: 2560d, Large: 4096d[/dim]")

def main():
    parser = argparse.ArgumentParser(description="Generate and visualize embeddings")
    parser.add_argument("--models", nargs="+", choices=["small", "medium", "large"], 
                       default=["small", "medium", "large"],
                       help="Which models to test")
    parser.add_argument("--output", default="embeddings_export",
                       help="Output directory for TSV files")
    parser.add_argument("--url", default="http://localhost:8000",
                       help="Server URL")
    args = parser.parse_args()
    
    console.print("[bold cyan]🎯 Thematic Embedding Generator & Validator[/bold cyan]")
    console.print("=" * 60)
    
    # Check server
    generator = EmbeddingGenerator(args.url)
    if not generator.check_server():
        console.print("[red]❌ Server not running! Start with: python server.py[/red]")
        return
    
    console.print("[green]✓ Server connected[/green]")
    
    # Prepare samples
    texts, labels, categories = prepare_samples()
    console.print(f"\n📝 Prepared {len(texts)} text samples across {len(set(categories))} categories")
    
    # Show category distribution
    cat_counts = {}
    for cat in categories:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Count", justify="right")
    for cat, count in sorted(cat_counts.items()):
        table.add_row(cat, str(count))
    console.print(table)
    
    # Generate embeddings for each model
    all_embeddings = {}
    all_validations = {}
    
    for model in args.models:
        console.print(f"\n[bold yellow]🚀 Testing {model.upper()} model[/bold yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console
        ) as progress:
            task = progress.add_task(f"Generating embeddings...", total=1)
            
            start_time = time.time()
            embeddings = generator.generate_embeddings(texts, model=model)
            elapsed = time.time() - start_time
            
            progress.update(task, completed=1)
        
        console.print(f"  ✓ Generated {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
        console.print(f"  ⏱️  Time: {elapsed:.2f}s ({len(texts)/elapsed:.1f} texts/sec)")
        
        # Validate embeddings
        validation = validate_embeddings(embeddings, labels, categories)
        all_embeddings[model] = embeddings
        all_validations[model] = validation
        
        # Print validation results
        console.print(f"\n  [cyan]Validation Results:[/cyan]")
        console.print(f"    • Avg norm: {validation['avg_norm']:.3f} ± {validation['norm_std']:.3f}")
        console.print(f"    • Inter-category similarity: {validation['avg_inter_category_similarity']:.3f}")
        
        console.print(f"    • Intra-category similarities:")
        for cat, sim in validation['intra_category_similarity'].items():
            console.print(f"      - {cat}: {sim:.3f}")
        
        # Export for projector
        output_dir = Path(args.output)
        vectors_file, metadata_file, config_file = export_for_projector(
            embeddings, labels, categories, texts, model, output_dir
        )
        
        console.print(f"\n  [green]✓ Exported to:[/green]")
        console.print(f"    • Vectors: {vectors_file}")
        console.print(f"    • Metadata: {metadata_file}")
        console.print(f"    • Config: {config_file}")
    
    # Compare models if multiple were tested
    if len(all_embeddings) > 1:
        analyze_model_differences(all_embeddings, texts)
    
    # Print summary
    console.print("\n[bold green]✅ Embedding Generation Complete![/bold green]")
    console.print("\n[bold cyan]📊 Summary:[/bold cyan]")
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Model", style="yellow")
    summary_table.add_column("Dimension", justify="right")
    summary_table.add_column("Avg Norm", justify="right")
    summary_table.add_column("Category Cohesion", justify="right")
    
    for model in args.models:
        if model in all_embeddings:
            emb = all_embeddings[model]
            val = all_validations[model]
            avg_intra = np.mean(list(val['intra_category_similarity'].values()))
            
            summary_table.add_row(
                model.upper(),
                str(emb.shape[1]),
                f"{val['avg_norm']:.3f}",
                f"{avg_intra:.3f}"
            )
    
    console.print(summary_table)
    
    console.print("\n[bold yellow]📈 To visualize:[/bold yellow]")
    console.print("1. Go to: https://projector.tensorflow.org/")
    console.print(f"2. Click 'Load' and upload the TSV files from '{args.output}/'")
    console.print("3. Or use the standalone version with the config.json files")
    console.print("\n[dim]Note: Higher category cohesion (intra-category similarity) indicates[/dim]")
    console.print("[dim]better semantic understanding. Values above 0.7 are excellent.[/dim]")

if __name__ == "__main__":
    main()
"""
Graph-Based Learning Objective Generation with Llama 3 70B
Extracts concepts → Builds knowledge graph → Generates LOs
Using Together AI API
"""

import json
import os
import glob
import time
from typing import List, Dict, Any
import requests
from dotenv import load_dotenv
import pdfplumber
from tqdm import tqdm
import networkx as nx
from collections import defaultdict

# ==================== CONFIGURATION ====================
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError("TOGETHER_API_KEY not found in .env file!")

# Paths
SLIDE_DECKS_FOLDER = "../../raw-data/osn_lecs"
COURSE_TITLE = "Advanced Operating Systems"
COURSE_CODE = "CS3.304"
OUTPUT_FILE = "../../datasets/slide_based_los_graph_method.json"
GRAPH_OUTPUT = "../../datasets/graphs/slides_concept_graph.json"
PROGRESS_FILE = "../../datasets/graphs/slides_graph_extraction_progress.json"

# Model
MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct-Turbo"

# Rate limiting settings
SLIDES_PER_BATCH = 25
DELAY_BETWEEN_CALLS = 2

# API endpoint
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"


# ==================== SLIDE EXTRACTION ====================
def extract_slides_from_pdf(file_path: str) -> List[Dict]:
    """Extract text from PDF slides."""
    slides = []
    
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                slides.append({
                    "slide_number": i + 1,
                    "source_file": os.path.basename(file_path),
                    "content": text.strip()
                })
    return slides


def extract_all_slides(folder_path: str) -> List[Dict]:
    """Extract slides from all PDFs in folder."""
    all_slides = []
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    
    print(f"\n📚 Extracting slides from {len(pdf_files)} PDF files...")
    
    for path in tqdm(pdf_files, desc="Processing PDFs"):
        slides = extract_slides_from_pdf(path)
        all_slides.extend(slides)
    
    print(f"✓ Total slides extracted: {len(all_slides)}")
    return all_slides


# ==================== RESUME CAPABILITY ====================
def load_progress() -> Dict:
    """Load progress from previous run."""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"concepts": [], "relationships": [], "batches_processed": 0}
    return {"concepts": [], "relationships": [], "batches_processed": 0}


def save_progress(progress: Dict):
    """Save progress for resume capability."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ==================== TOGETHER AI API WRAPPER ====================
def generate_with_llama(prompt: str, json_mode: bool = True, max_tokens: int = 3000) -> str:
    """
    Generate response using Llama 3 via Together AI.
    """
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert educational curriculum designer and concept extraction specialist."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    if json_mode:
        data["response_format"] = {"type": "json_object"}
    
    try:
        response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Together AI API Error: {str(e)}")


def extract_json_from_response(text: str) -> Dict:
    """
    Extract JSON from response (handles markdown code blocks).
    """
    # Try to find JSON in markdown code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        json_str = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        json_str = text[start:end].strip()
    else:
        json_str = text.strip()
    
    # Remove any leading/trailing whitespace
    json_str = json_str.strip()
    
    # Parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to find JSON object in text
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            return json.loads(json_str[start_idx:end_idx])
        raise e


# ==================== CONCEPT EXTRACTION ====================
def create_concept_extraction_prompt(slides_batch: List[Dict]) -> str:
    """
    Create prompt for extracting concepts from a batch of slides.
    """
    slides_text = ""
    for slide in slides_batch:
        slides_text += f"\n--- Slide {slide['slide_number']} ({slide['source_file']}) ---\n"
        slides_text += slide['content'][:1000]
        slides_text += "\n"
    
    return f"""You are an expert in educational concept extraction and knowledge graph construction.

**Task**: Analyze these slides and extract key concepts with relationships.

**Slides Content**:
{slides_text}

**Instructions**:
1. Identify 5-12 KEY CONCEPTS (technical terms, not trivial)
2. For each concept, determine:
   - **Importance**: critical/high/medium/low
   - **Bloom Level**: remember/understand/apply/analyze/evaluate/create
   - **Slides**: Which slide numbers discuss this (MUST be integers, not objects)

3. Identify RELATIONSHIPS:
   - **prerequisite_for**: A must be learned before B
   - **is_a**: Taxonomic (e.g., "DFA is_a Finite Automaton")
   - **part_of**: Compositional
   - **enables**: Learning A enables B

**CRITICAL**: slide_numbers MUST be an array of integers like [5, 6, 7], NOT objects!

**Output Format** (JSON ONLY):
{{
  "concepts": [
    {{
      "name": "Process Scheduling",
      "importance": "critical",
      "bloom_level": "analyze",
      "slide_numbers": [5, 6, 7],
      "definition": "Brief 1-sentence definition"
    }}
  ],
  "relationships": [
    {{
      "source": "Process Control Block",
      "target": "Process Scheduling",
      "type": "enables",
      "strength": 0.9
    }}
  ]
}}

**Quality Guidelines**:
- Focus on CORE technical concepts only
- Ensure logical prerequisite chains
- Use strength 0.7-1.0 for strong, 0.4-0.6 for weak
- Skip generic/obvious concepts
- slide_numbers must be simple integer arrays

Return ONLY valid JSON."""


def extract_concepts_with_retry(slides: List[Dict], max_retries: int = 3) -> Dict:
    """
    Extract concepts from all slides with smart batching and retry logic.
    """
    print(f"\n🧠 Extracting concepts from {len(slides)} slides using Llama 3 70B...")
    
    # Load previous progress
    progress = load_progress()
    all_concepts = progress.get("concepts", [])
    all_relationships = progress.get("relationships", [])
    batches_done = progress.get("batches_processed", 0)
    
    # Calculate batches
    total_batches = (len(slides) + SLIDES_PER_BATCH - 1) // SLIDES_PER_BATCH
    
    print(f"  Processing in {total_batches} batches ({SLIDES_PER_BATCH} slides each)")
    print(f"  Resuming from batch {batches_done + 1}")
    
    for batch_idx in range(batches_done, total_batches):
        start_idx = batch_idx * SLIDES_PER_BATCH
        end_idx = min(start_idx + SLIDES_PER_BATCH, len(slides))
        batch = slides[start_idx:end_idx]
        
        prompt = create_concept_extraction_prompt(batch)
        
        # Retry logic
        for attempt in range(max_retries):
            try:
                print(f"\n  Batch {batch_idx + 1}/{total_batches} (slides {start_idx+1}-{end_idx})...", end=" ")
                
                response_text = generate_with_llama(prompt, json_mode=True, max_tokens=3000)
                result = json.loads(response_text)
                
                batch_concepts = result.get("concepts", [])
                batch_relationships = result.get("relationships", [])
                
                # Validate and fix slide_numbers format
                for concept in batch_concepts:
                    if "slide_numbers" in concept:
                        slide_nums = concept["slide_numbers"]
                        if isinstance(slide_nums, list):
                            # Ensure all elements are integers
                            concept["slide_numbers"] = [
                                int(x) if isinstance(x, (int, float, str)) and str(x).isdigit() else x
                                for x in slide_nums
                                if isinstance(x, (int, float)) or (isinstance(x, str) and x.isdigit())
                            ]
                
                all_concepts.extend(batch_concepts)
                all_relationships.extend(batch_relationships)
                
                print(f"✓ (+{len(batch_concepts)} concepts)")
                
                # Save progress
                progress = {
                    "concepts": all_concepts,
                    "relationships": all_relationships,
                    "batches_processed": batch_idx + 1
                }
                save_progress(progress)
                
                # Rate limiting
                time.sleep(DELAY_BETWEEN_CALLS)
                break
                
            except Exception as e:
                error_str = str(e)
                
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = 60 * (attempt + 1)
                    print(f"\n  ⚠️  Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    
                    if attempt == max_retries - 1:
                        print(f"\n  ❌ Failed after {max_retries} attempts. Progress saved.")
                        print(f"  📊 Processed {batch_idx} of {total_batches} batches so far.")
                        print(f"\n  💡 Run the script again to resume from batch {batch_idx + 1}")
                        return merge_concepts({
                            "concepts": all_concepts,
                            "relationships": all_relationships
                        })
                else:
                    print(f"\n  ⚠️  Error: {error_str}")
                    if attempt == max_retries - 1:
                        print(f"\n  ⚠️  Skipping batch after {max_retries} attempts")
                        continue
    
    print(f"\n✓ Concept extraction complete!")
    
    # Clean up progress file
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)
    
    return merge_concepts({
        "concepts": all_concepts,
        "relationships": all_relationships
    })


def merge_concepts(raw_graph: Dict) -> Dict:
    """Merge duplicate concepts and relationships."""
    
    # Merge concepts by name
    concept_map = {}
    for c in raw_graph["concepts"]:
        name = c["name"]
        if name in concept_map:
            # Merge slide numbers - ensure they're all integers
            existing = concept_map[name]
            existing_slides = existing.get("slide_numbers", [])
            new_slides = c.get("slide_numbers", [])
            
            # Filter to only include integers
            all_slides = []
            for item in existing_slides + new_slides:
                if isinstance(item, int):
                    all_slides.append(item)
                elif isinstance(item, (float, str)) and str(item).replace('.', '').isdigit():
                    all_slides.append(int(float(item)))
            
            existing["slide_numbers"] = list(set(all_slides))
        else:
            concept_map[name] = c
    
    # Remove duplicate relationships
    unique_rels = {}
    for r in raw_graph["relationships"]:
        key = (r["source"], r["target"], r["type"])
        if key not in unique_rels:
            unique_rels[key] = r
    
    return {
        "concepts": list(concept_map.values()),
        "relationships": list(unique_rels.values())
    }


# ==================== GRAPH ANALYSIS ====================
def analyze_graph(graph: Dict) -> Dict:
    """Analyze concept graph using centrality metrics."""
    
    G = nx.DiGraph()
    
    # Add nodes
    for concept in graph["concepts"]:
        G.add_node(
            concept["name"],
            importance=concept.get("importance", "medium"),
            bloom=concept.get("bloom_level", "understand")
        )
    
    # Add edges
    for rel in graph["relationships"]:
        G.add_edge(
            rel["source"],
            rel["target"],
            type=rel["type"],
            weight=rel.get("strength", 0.5)
        )
    
    # Compute metrics
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except:
        pagerank = {n: 1.0 for n in G.nodes()}
    
    # Rank concepts
    analysis = []
    for concept in graph["concepts"]:
        name = concept["name"]
        
        importance_score = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }.get(concept.get("importance", "medium"), 0.5)
        
        composite_score = (
            pagerank.get(name, 0) * 0.5 +
            importance_score * 0.5
        )
        
        analysis.append({
            "name": name,
            "score": composite_score,
            "importance": concept.get("importance"),
            "bloom": concept.get("bloom_level"),
            "slide_count": len(concept.get("slide_numbers", []))
        })
    
    analysis.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "ranked_concepts": analysis,
        "graph_stats": {
            "total_concepts": len(G.nodes()),
            "total_relationships": len(G.edges()),
            "density": nx.density(G) if len(G.nodes()) > 0 else 0
        }
    }


# ==================== LO GENERATION FROM GRAPH ====================
def create_lo_prompt_from_graph(graph: Dict, analysis: Dict) -> str:
    """Generate LOs based on the knowledge graph."""
    
    top_concepts = [c["name"] for c in analysis["ranked_concepts"][:15]]
    
    # Group by Bloom level
    bloom_groups = defaultdict(list)
    for c in analysis["ranked_concepts"][:20]:
        bloom_groups[c["bloom"]].append(c["name"])
    
    return f"""You are an expert Educational Curriculum Designer.

**Task**: Generate 6-7 Learning Outcomes for this course based on the concept knowledge graph.

**Course**: {COURSE_TITLE} ({COURSE_CODE})

**Knowledge Graph Summary**:
- Total Concepts Identified: {analysis['graph_stats']['total_concepts']}
- Total Relationships: {analysis['graph_stats']['total_relationships']}

**Top 15 Concepts (by importance & centrality)**:
{json.dumps(top_concepts, indent=2)}

**Concepts by Bloom's Taxonomy Level**:
{json.dumps(dict(bloom_groups), indent=2)}

**Instructions**:
Generate 6-7 learning outcomes that:
1. Cover the most important concepts from the graph
2. Follow Bloom's Taxonomy progression (lower → higher)
3. Are specific and measurable
4. Reflect the prerequisite relationships

**Bloom's Taxonomy Verbs**:
- Remember: Define, List, Identify, Recall
- Understand: Explain, Describe, Summarize, Interpret
- Apply: Apply, Implement, Execute, Demonstrate
- Analyze: Analyze, Examine, Differentiate, Compare
- Evaluate: Evaluate, Assess, Critique, Judge
- Create: Design, Construct, Develop, Formulate

**Output Format** (JSON array ONLY):
[
  "CO-1: [Bloom verb] [specific technical content]",
  "CO-2: [Bloom verb] [specific technical content]",
  "CO-3: [Bloom verb] [specific technical content]",
  "CO-4: [Bloom verb] [specific technical content]",
  "CO-5: [Bloom verb] [specific technical content]",
  "CO-6: [Bloom verb] [specific technical content]",
  "CO-7: [Bloom verb] [specific technical content]"
]

**Requirements**:
✓ Start each with a Bloom verb
✓ Be specific to OS concepts
✓ Progress from foundational to advanced
✓ Cover breadth of topics
✓ No redundancy

Return ONLY the JSON array."""


def generate_los_from_graph(graph: Dict, analysis: Dict) -> List[str]:
    """Generate learning objectives from concept graph."""
    
    print("\n📝 Generating learning objectives from knowledge graph...")
    
    prompt = create_lo_prompt_from_graph(graph, analysis)
    
    try:
        response_text = generate_with_llama(prompt, json_mode=True, max_tokens=1500)
        los = json.loads(response_text)
        
        if isinstance(los, list) and 5 <= len(los) <= 8:
            print(f"✓ Generated {len(los)} learning objectives")
            return los
        else:
            print("⚠️  Invalid LO format, retrying...")
            time.sleep(3)
            response_text = generate_with_llama(prompt, json_mode=True, max_tokens=1500)
            los = json.loads(response_text)
            return los
            
    except Exception as e:
        print(f"❌ Error generating LOs: {e}")
        return []


# ==================== MAIN PIPELINE ====================
def main():
    print("="*70)
    print("  GRAPH-BASED LEARNING OBJECTIVE GENERATION")
    print("  Using Llama 3 70B via Together AI")
    print("  Slides → Concept Graph → Learning Objectives")
    print("="*70)
    
    # CREATE REQUIRED DIRECTORIES
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(GRAPH_OUTPUT), exist_ok=True)
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    
    # Step 1: Extract slides
    all_slides = extract_all_slides(SLIDE_DECKS_FOLDER)
    
    if not all_slides:
        print("❌ No slides extracted!")
        return
    
    # Step 2: Build concept graph (with resume capability)
    concept_graph = extract_concepts_with_retry(all_slides)
    
    if not concept_graph["concepts"]:
        print("\n❌ No concepts extracted. Check your API quota.")
        return
    
    # Save graph
    with open(GRAPH_OUTPUT, 'w') as f:
        json.dump(concept_graph, f, indent=2)
    print(f"\n✓ Saved concept graph: {GRAPH_OUTPUT}")
    
    # Step 3: Analyze graph
    print("\n📊 Analyzing concept graph...")
    analysis = analyze_graph(concept_graph)
    
    print(f"  - Concepts: {analysis['graph_stats']['total_concepts']}")
    print(f"  - Relationships: {analysis['graph_stats']['total_relationships']}")
    print(f"  - Density: {analysis['graph_stats']['density']:.3f}")
    print(f"\n  Top 5 Concepts:")
    for i, c in enumerate(analysis['ranked_concepts'][:5], 1):
        print(f"    {i}. {c['name']} (importance: {c['importance']}, Bloom: {c['bloom']})")
    
    # Step 4: Generate LOs
    learning_objectives = generate_los_from_graph(concept_graph, analysis)
    
    if not learning_objectives:
        print("❌ Failed to generate LOs")
        return
    
    # Step 5: Save final output
    output = {
        "course_title": COURSE_TITLE,
        "course_code": COURSE_CODE,
        "metadata": {
            "total_slides": len(all_slides),
            "source_folder": SLIDE_DECKS_FOLDER,
            "generation_method": "Hierarchical Concept Dependency Graph",
            "model_used": MODEL_NAME
        },
        "concept_graph_summary": {
            "total_concepts": len(concept_graph["concepts"]),
            "total_relationships": len(concept_graph["relationships"]),
            "top_10_concepts": [c["name"] for c in analysis["ranked_concepts"][:10]],
            "graph_density": analysis["graph_stats"]["density"]
        },
        "learning_objectives": learning_objectives
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved output: {OUTPUT_FILE}")
    
    # Display results
    print("\n" + "="*70)
    print("  GENERATED LEARNING OBJECTIVES")
    print("="*70)
    for lo in learning_objectives:
        print(f"\n{lo}")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
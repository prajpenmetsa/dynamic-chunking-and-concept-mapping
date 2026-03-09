"""
Coursera Course Data Collector

Collects course descriptions and learning outcomes from Coursera.

Requirements:
    pip install requests beautifulsoup4 tqdm python-dotenv

Usage:
    python coursera_collector.py --output ../../datasets/large_scale_collection/raw/coursera/batch_1.json --limit 1000

Methods:
1. Coursera Catalog API (if available)
2. Web scraping (fallback)
3. Process pre-provided data files
"""

import json
import requests
import time
import os
import argparse
from datetime import datetime
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
COURSERA_API_BASE = "https://api.coursera.org/api"
COURSERA_WEB_BASE = "https://www.coursera.org"
RATE_LIMIT_DELAY = 2  # seconds between requests
MAX_RETRIES = 3
RETRY_BACKOFF = 5

class CourseraCollector:
    """Collects course data from Coursera"""
    
    def __init__(self, output_file: str, use_api: bool = True):
        self.output_file = output_file
        self.use_api = use_api
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.collected_data = []
        
    def collect_from_api(self, limit: int = 1000) -> List[Dict]:
        """
        Collect courses using Coursera API (if available)

        Note: Coursera's public API may require authentication.
        Check: https://api.coursera.org/api/courses.v1
        """
        print("🔍 Attempting to collect from Coursera API...")

        endpoint = f"{COURSERA_API_BASE}/courses.v1"
        page_size = 100
        all_courses = []
        start = 0

        # Verify endpoint is reachable first
        try:
            test_response = self.session.get(endpoint, params={
                'limit': 1,
                'fields': 'name,description,learningObjectives,partners'
            }, timeout=30)
            if test_response.status_code != 200:
                print("❌ API access not available. Consider web scraping or pre-provided data.")
                return []
            print("✅ API access confirmed. Starting paginated collection...")
        except Exception as e:
            print(f"❌ API endpoint unreachable: {e}")
            return []

        with tqdm(total=limit, desc="Collecting courses") as pbar:
            while len(all_courses) < limit:
                fetch = min(page_size, limit - len(all_courses))
                params = {
                    'limit': fetch,
                    'start': start,
                    'fields': 'name,description,learningObjectives,partners'
                }
                try:
                    response = self.session.get(endpoint, params=params, timeout=30)
                    if response.status_code != 200:
                        print(f"⚠️ Unexpected status {response.status_code} at start={start}")
                        break

                    data = response.json()
                    elements = data.get('elements', [])
                    if not elements:
                        break  # No more results

                    batch = self._parse_api_response(data)
                    all_courses.extend(batch)
                    pbar.update(len(batch))

                    # Save progress every 500 courses
                    if len(all_courses) % 500 == 0:
                        self.save_progress(all_courses)

                    # Check if there are more pages
                    paging = data.get('paging', {})
                    if 'next' not in paging:
                        break  # Reached last page

                    start += fetch
                    time.sleep(RATE_LIMIT_DELAY)

                except Exception as e:
                    print(f"⚠️ Error at start={start}: {e}")
                    break

        print(f"✅ Collected {len(all_courses)} courses total")
        return all_courses
    
    def _parse_cml_lo(self, lo_item) -> Optional[str]:
        """Extract plain text from a Coursera CML learning objective object"""
        if isinstance(lo_item, str):
            return lo_item.strip() or None

        if not isinstance(lo_item, dict):
            return None

        # Try renderableHtml first (cleanest path)
        try:
            html = lo_item['definition']['renderableHtmlWithMetadata']['renderableHtml']
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=' ').strip()
            if text:
                return text
        except (KeyError, TypeError):
            pass

        # Fallback: strip tags from raw CML value
        try:
            raw = lo_item['definition']['value']
            soup = BeautifulSoup(raw, 'html.parser')
            text = soup.get_text(separator=' ').strip()
            if text:
                return text
        except (KeyError, TypeError):
            pass

        return None

    def _parse_api_response(self, api_data: Dict) -> List[Dict]:
        """Parse API response into standardized format"""
        courses = []

        for course in api_data.get('elements', []):
            raw_los = course.get('learningObjectives', [])
            parsed_los = [
                text for lo in raw_los
                if (text := self._parse_cml_lo(lo))
            ]
            standardized = self._standardize_course(
                course_id=course.get('id'),
                title=course.get('name'),
                description=course.get('description'),
                learning_outcomes=parsed_los,
                institution=self._extract_institution(course),
                metadata=course
            )
            courses.append(standardized)

        return courses
    
    def collect_from_web(self, start_page: int = 1, num_pages: int = 10) -> List[Dict]:
        """
        Collect courses by scraping Coursera web pages
        
        WARNING: This may violate Coursera's ToS. Use with caution.
        Prefer API or pre-provided data.
        """
        print("🌐 Collecting from Coursera web pages...")
        print("⚠️ Note: Ensure this complies with Coursera's Terms of Service")
        
        courses = []
        
        for page in tqdm(range(start_page, start_page + num_pages)):
            try:
                # Coursera course catalog URL
                url = f"{COURSERA_WEB_BASE}/courses"
                params = {'page': page}
                
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    page_courses = self._extract_courses_from_page(soup)
                    courses.extend(page_courses)
                    
                    # Save progress
                    if len(courses) % 100 == 0:
                        self.save_progress(courses)
                
                # Respect rate limits
                time.sleep(RATE_LIMIT_DELAY)
                
            except Exception as e:
                print(f"⚠️ Error on page {page}: {e}")
                continue
        
        return courses
    
    def _extract_courses_from_page(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract course data from a Coursera page"""
        courses = []
        
        # Note: These selectors may need updating based on Coursera's current HTML structure
        course_cards = soup.find_all('div', class_=['course-card', 'cds-ProductCard-container'])
        
        for card in course_cards:
            try:
                title = card.find('h3', class_=['course-title', 'cds-119']).text.strip()
                description = card.find('p', class_=['course-description', 'cds-119']).text.strip()
                
                # Extract course link to get more details
                link = card.find('a', href=True)
                course_url = COURSERA_WEB_BASE + link['href'] if link else None
                
                if course_url:
                    # Visit individual course page for LOs
                    course_details = self._get_course_details(course_url)
                    
                    standardized = self._standardize_course(
                        course_id=self._extract_course_id(course_url),
                        title=title,
                        description=description,
                        learning_outcomes=course_details.get('learning_outcomes', []),
                        institution=course_details.get('institution'),
                        metadata={'url': course_url}
                    )
                    courses.append(standardized)
                    
            except Exception as e:
                print(f"⚠️ Error extracting course from card: {e}")
                continue
        
        return courses
    
    def _get_course_details(self, course_url: str) -> Dict:
        """Get detailed course information including learning outcomes"""
        try:
            response = self.session.get(course_url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract learning outcomes (usually in "What you'll learn" section)
            los = []
            lo_section = soup.find('div', class_=['syllabus', 'what-you-will-learn'])
            if lo_section:
                lo_items = lo_section.find_all('li')
                los = [item.text.strip() for item in lo_items]
            
            # Extract institution
            institution = soup.find('div', class_='partner-name')
            institution = institution.text.strip() if institution else 'Unknown'
            
            time.sleep(RATE_LIMIT_DELAY)  # Be respectful
            
            return {
                'learning_outcomes': los,
                'institution': institution
            }
            
        except Exception as e:
            print(f"⚠️ Error getting course details: {e}")
            return {}
    
    def process_provided_data(self, input_file: str) -> List[Dict]:
        """
        Process pre-provided Coursera data file
        
        Assumes the professor provided data in some format (CSV, JSON, etc.)
        """
        print(f"📂 Processing provided data from: {input_file}")
        
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            return []
        
        courses = []
        
        # Handle different file formats
        if input_file.endswith('.json'):
            with open(input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                courses = self._standardize_provided_data(raw_data)
        
        elif input_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(input_file)
            courses = self._standardize_provided_data(df.to_dict('records'))
        
        print(f"✅ Processed {len(courses)} courses from provided data")
        return courses
    
    def _standardize_provided_data(self, raw_data: any) -> List[Dict]:
        """Standardize provided data to our schema"""
        courses = []
        
        # Handle list of courses
        if isinstance(raw_data, list):
            for item in raw_data:
                standardized = self._standardize_course(
                    course_id=item.get('id', item.get('course_id')),
                    title=item.get('title', item.get('name', item.get('course_name'))),
                    description=item.get('description', item.get('overview')),
                    learning_outcomes=self._extract_los_from_item(item),
                    institution=item.get('institution', item.get('partner', item.get('university'))),
                    metadata=item
                )
                courses.append(standardized)
        
        return courses
    
    def _extract_los_from_item(self, item: Dict) -> List[str]:
        """Extract learning outcomes from various possible field names"""
        possible_fields = [
            'learning_outcomes', 'learningOutcomes', 'outcomes',
            'learning_objectives', 'learningObjectives', 'objectives',
            'what_you_will_learn', 'skills', 'skills_you_will_gain'
        ]
        
        for field in possible_fields:
            if field in item and item[field]:
                value = item[field]
                # Handle string (comma-separated or newline-separated)
                if isinstance(value, str):
                    if ',' in value:
                        return [lo.strip() for lo in value.split(',')]
                    elif '\n' in value:
                        return [lo.strip() for lo in value.split('\n') if lo.strip()]
                    else:
                        return [value]
                # Handle list
                elif isinstance(value, list):
                    return value
        
        return []
    
    def _standardize_course(self, course_id: str, title: str, description: str,
                           learning_outcomes: List[str], institution: str,
                           metadata: Dict) -> Dict:
        """Convert to standardized schema"""
        return {
            "id": f"coursera_{course_id}",
            "source": "coursera",
            "course_code": None,  # Coursera doesn't use traditional course codes
            "title": title,
            "description": description,
            "learning_outcomes": learning_outcomes,
            "metadata": {
                "institution": institution,
                "level": metadata.get('level', metadata.get('difficulty')),
                "category": metadata.get('category', metadata.get('subject')),
                "language": metadata.get('language', 'English'),
                "date_collected": datetime.now().isoformat(),
                "url": metadata.get('url', f"{COURSERA_WEB_BASE}/learn/{course_id}")
            }
        }
    
    def _extract_institution(self, course_data: Dict) -> str:
        """Extract institution name from course data"""
        partners = course_data.get('partners', [])
        if partners:
            return partners[0].get('name', 'Unknown')
        return course_data.get('institution', 'Unknown')
    
    def _extract_course_id(self, url: str) -> str:
        """Extract course ID from URL"""
        parts = url.rstrip('/').split('/')
        return parts[-1] if parts else 'unknown'
    
    def save_progress(self, courses: List[Dict]):
        """Save collected data incrementally"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(courses, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved {len(courses)} courses to {self.output_file}")
    
    def validate_data(self, courses: List[Dict]) -> Dict:
        """Validate collected data quality"""
        total = len(courses)
        with_los = sum(1 for c in courses if c['learning_outcomes'])
        with_description = sum(1 for c in courses if len(c.get('description', '')) > 50)
        
        stats = {
            'total_courses': total,
            'with_learning_outcomes': with_los,
            'with_description': with_description,
            'lo_coverage': round(with_los / total * 100, 2) if total > 0 else 0,
            'avg_los_per_course': round(
                sum(len(c['learning_outcomes']) for c in courses) / total, 2
            ) if total > 0 else 0
        }
        
        return stats

def main():
    parser = argparse.ArgumentParser(description='Collect Coursera course data')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--method', choices=['api', 'web', 'provided'], default='api',
                       help='Collection method')
    parser.add_argument('--input', help='Input file for provided data method')
    parser.add_argument('--limit', type=int, default=1000, help='Number of courses to collect')
    
    args = parser.parse_args()
    
    collector = CourseraCollector(args.output)
    
    print("🚀 Starting Coursera data collection...")
    print(f"   Method: {args.method}")
    print(f"   Output: {args.output}")
    
    courses = []
    
    if args.method == 'api':
        courses = collector.collect_from_api(limit=args.limit)
    elif args.method == 'web':
        num_pages = args.limit // 20  # Assume ~20 courses per page
        courses = collector.collect_from_web(num_pages=num_pages)
    elif args.method == 'provided':
        if not args.input:
            print("❌ --input required for 'provided' method")
            return
        courses = collector.process_provided_data(args.input)
    
    if courses:
        collector.save_progress(courses)
        stats = collector.validate_data(courses)
        
        print("\n" + "="*60)
        print("📊 COLLECTION STATISTICS")
        print("="*60)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print("="*60)
        
        # Save statistics
        stats_file = args.output.replace('.json', '_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\n✅ Statistics saved to: {stats_file}")
    else:
        print("\n❌ No courses collected. Check method and connectivity.")

if __name__ == "__main__":
    main()

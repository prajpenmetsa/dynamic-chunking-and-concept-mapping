# Data Collection Module

Scripts for collecting course descriptions and learning outcomes from multiple sources at scale.

## Quick Start

```bash
# Install dependencies
pip install -r requirements_data_collection.txt

# Collect from Coursera (if you have provided data)
python coursera_collector.py \
  --method provided \
  --input /path/to/coursera_data.json \
  --output ../../datasets/large_scale_collection/raw/coursera/batch_1.json

# Or try Coursera API
python coursera_collector.py \
  --method api \
  --output ../../datasets/large_scale_collection/raw/coursera/batch_1.json \
  --limit 5000

# Collect from MIT
python university_collector.py \
  --university mit \
  --output ../../datasets/large_scale_collection/raw/universities/mit.json

# Collect from Udemy  
python udemy_collector.py \
  --method api \
  --output ../../datasets/large_scale_collection/raw/udemy/batch_1.json \
  --limit 10000
```

## Available Collectors

### 1. coursera_collector.py
Collects from Coursera using API or web scraping.

**Methods:**
- `api` - Use Coursera API (fast, requires auth)
- `web` - Web scraping (slower, no auth needed)
- `provided` - Process pre-provided data files

**Example:**
```bash
python coursera_collector.py --method provided --input coursera_data.csv --output batch_1.json
```

### 2. udemy_collector.py
Collects from Udemy using Affiliate API or web scraping.

**Setup:**
1. Get API credentials from https://www.udemy.com/developers/affiliate/
2. Add to `.env`:
   ```
   UDEMY_CLIENT_ID=your_id
   UDEMY_CLIENT_SECRET=your_secret
   ```

**Example:**
```bash
python udemy_collector.py --method api --limit 10000 --output batch_1.json
```

### 3. university_collector.py
Collects from university course catalogs.

**Supported Universities:**
- MIT OpenCourseWare (mit)
- Stanford (stanford)
- UC Berkeley (berkeley)
- Carnegie Mellon (cmu)
- IIIT Hyderabad (iiit_hyderabad)
- IIT Bombay (iit_bombay)
- IIT Delhi (iit_delhi)
- IIT Madras (iit_madras)

**Example:**
```bash
python university_collector.py --university mit --output mit.json
```

## Data Format

All collectors output standardized JSON:

```json
{
  "id": "unique_identifier",
  "source": "coursera|udemy|university_name",
  "course_code": "CS101 or null",
  "title": "Course Title",
  "description": "Course description text...",
  "learning_outcomes": [
    "Students will be able to...",
    "Demonstrate understanding of..."
  ],
  "metadata": {
    "institution": "University Name",
    "level": "undergraduate|graduate|professional",
    "category": "Computer Science",
    "language": "English",
    "date_collected": "2026-03-07T12:00:00",
    "url": "source_url"
  }
}
```

## Output Structure

```
datasets/large_scale_collection/
  raw/
    coursera/
      batch_1.json
      batch_1_stats.json
    udemy/
      batch_1.json
      batch_1_stats.json
    universities/
      mit.json
      mit_stats.json
      stanford.json
      iiit_hyderabad.json
```

## Statistics

Each collector generates statistics:

```json
{
  "total_courses": 5000,
  "with_learning_outcomes": 4500,
  "with_description": 4950,
  "lo_coverage": 90.0,
  "avg_los_per_course": 5.2
}
```

## Rate Limiting

All collectors respect rate limits:
- Coursera: 2 seconds between requests
- Udemy: 3 seconds between requests
- Universities: 2 seconds between requests

Adjust `RATE_LIMIT_DELAY` in scripts if needed.

## Error Handling

- Automatic retries with exponential backoff
- Progress saved incrementally (every 100-500 courses)
- Graceful degradation (continues on errors)
- Detailed error logging

## Tips

1. **Start small:** Test with `--limit 10` first
2. **Run in parallel:** Use multiple terminals for different sources
3. **Save often:** Collectors auto-save progress
4. **Check stats:** Review `*_stats.json` files for quality
5. **Be respectful:** Don't set rate limits too low

## Troubleshooting

**"API authentication failed"**
- Check API keys in `.env` file
- Verify credentials are still valid

**"Rate limited"**
- Increase `RATE_LIMIT_DELAY` in script
- Run overnight with lower rate

**"No courses collected"**
- Check internet connection
- Verify source URL is accessible
- Try with `--limit 10` to test

**"Script crashes"**
- Check error message
- Last successful save in output file
- Re-run from where it stopped

## Next Steps

After collection:
1. Merge datasets: `python merge_datasets.py`
2. Validate quality: `python validate_data.py`
3. Generate statistics: `python generate_stats.py`
4. Send progress report to professor

## Support

For issues, check:
1. Error logs in terminal
2. `*_stats.json` files for data quality
3. Source website accessibility
4. API rate limits/quotas

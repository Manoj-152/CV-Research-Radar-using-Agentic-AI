import arxiv
from datetime import datetime, timedelta, timezone


def fetch_papers_for_date(date, papers_per_day):
    date_str = date.strftime('%Y%m%d')
    client = arxiv.Client()
    search = arxiv.Search(
        query=f"cat:cs.CV AND submittedDate:[{date_str} TO {date_str}]",
        max_results=papers_per_day,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    papers = []
    for result in client.results(search):
        papers.append({
            "id":        result.entry_id.split('/')[-1],
            "title":     result.title,
            "authors":   [author.name for author in result.authors],
            "abstract":  result.summary.replace('\n', ' '),
            "pdf_url":   result.pdf_url,
            "published": result.published.strftime('%Y-%m-%d')
        })

    print(f"  {date.strftime('%Y-%m-%d')}: {len(papers)} papers fetched.")
    return papers


def fetch_recent_papers(days_back=2, total_papers=50):
    papers_per_day = total_papers // days_back
    print(f"Fetching up to {papers_per_day} CV papers per day for the last {days_back} days (total target: {total_papers}).")

    today = datetime.now(timezone.utc)
    all_papers = []

    for i in range(1, days_back + 1):
        date = today - timedelta(days=i)
        papers = fetch_papers_for_date(date, papers_per_day)
        all_papers.extend(papers)

    print(f"Total: {len(all_papers)} papers to evaluate.")
    return all_papers


if __name__ == '__main__':
    TOTAL_PAPERS = 50
    DAYS_BACK = 2

    papers = fetch_recent_papers(days_back=DAYS_BACK, total_papers=TOTAL_PAPERS)
    for p in papers:
        print(f"\nTitle    : {p['title']}")
        print(f"Published: {p['published']}")
        print(f"URL      : {p['pdf_url']}")

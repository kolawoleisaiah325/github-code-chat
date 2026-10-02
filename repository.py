"""Read public repositories at a pinned commit; never run their code."""
import json
import re
import urllib.error
import urllib.request
from urllib.parse import quote, urlsplit

from chunk_repo import chunk_repository


def validate_repository_url(url):
    parts = urlsplit(url.strip())
    if parts.scheme != "https" or parts.netloc != "github.com" or parts.query or parts.fragment:
        raise ValueError("Use a public repository URL such as https://github.com/owner/repository.")
    pieces = parts.path.strip("/").split("/")
    if len(pieces) != 2 or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) or part in {".", ".."} for part in pieces):
        raise ValueError("Enter the repository root URL, without a branch, file, or extra path.")
    owner, name = pieces
    name = name.removesuffix(".git")
    if not name:
        raise ValueError("The repository name is empty.")
    return f"https://github.com/{owner}/{name}"


def github_json(path):
    request = urllib.request.Request("https://api.github.com/" + path,
                                     headers={"Accept": "application/vnd.github+json", "User-Agent": "GitHub-Code-Chat"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"GitHub returned HTTP {exc.code}. Check that the repository is public; API rate limits may also apply.") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ValueError("Could not reach GitHub. Try the bundled example or retry later.") from exc


def ingest_repository(url):
    from gitingest import ingest
    canonical = validate_repository_url(url)
    slug = canonical.removeprefix("https://github.com/")
    info = github_json(f"repos/{slug}")
    if info.get("private"):
        raise ValueError("This demo accepts public repositories only.")
    ref = info["default_branch"]
    commit = github_json(f"repos/{slug}/commits/{quote(ref, safe='')}")["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("GitHub did not return a valid commit identifier.")
    summary, tree, content = ingest(f"{canonical}/tree/{commit}", max_file_size=100_000)
    if len(content) > 3_000_000:
        raise ValueError("The extracted text exceeds this demo's 3 MB limit. Choose a smaller repository.")
    chunks = chunk_repository(content)
    metadata = {"url": canonical, "name": slug, "commit": commit, "ref": ref, "kind": "public GitHub snapshot"}
    return summary, tree, content, chunks, metadata
